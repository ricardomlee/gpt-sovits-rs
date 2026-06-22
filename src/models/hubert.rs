//! Hubert Feature Extractor with ONNX Runtime (optional)

#[cfg(feature = "onnx")]
use ort::{ep, session::Session, value::Value, inputs};

#[cfg(not(feature = "onnx"))]
use candle_core::{Device, Tensor};
#[cfg(feature = "onnx")]
use candle_core::{Tensor, Device};
use crate::Result;
use std::path::Path;

/// Hubert model for audio feature extraction
pub struct HubertModel {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(not(feature = "onnx"))]
    _marker: std::marker::PhantomData<()>,
    device: String,
    sampling_rate: u32,
}

impl HubertModel {
    /// Load Hubert model from ONNX file
    #[cfg(feature = "onnx")]
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, "cpu")
    }

    #[cfg(feature = "onnx")]
    pub fn load_with_device(path: &str, device: &str) -> Result<Self> {
        let session = if device == "cuda" {
            Session::builder()?
                .with_execution_providers([ep::CUDA::default().build()])
                .map_err(|e| crate::Error::ModelLoadError(format!("Failed to configure CUDA EP: {}", e)))?
                .commit_from_file(path)
        } else {
            Session::builder()?
                .commit_from_file(path)
        }
        .map_err(|e| crate::Error::ModelLoadError(format!("Failed to load ONNX: {}", e)))?;

        Ok(Self {
            session,
            device: device.to_string(),
            sampling_rate: 16000,
        })
    }

    #[cfg(not(feature = "onnx"))]
    pub fn load(_path: &str) -> Result<Self> {
        Self::load_with_device(_path, "cpu")
    }

    #[cfg(not(feature = "onnx"))]
    pub fn load_with_device(_path: &str, device: &str) -> Result<Self> {
        Ok(Self {
            _marker: std::marker::PhantomData,
            device: device.to_string(),
            sampling_rate: 16000,
        })
    }

    /// Extract Hubert features from audio file
    #[cfg(feature = "onnx")]
    pub fn extract<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<Tensor> {
        let audio_data = self.load_audio(audio_path)?;
        let seq_len = audio_data.len();
        let input_array = (vec![1i64, seq_len as i64], audio_data);

        let inputs = inputs! {
            "input_values" => Value::from_array(input_array)?,
        };

        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        let output_value = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("features"))
            .or_else(|| outputs.get("hidden_states"))
            .ok_or_else(|| crate::Error::InferenceError("No output from Hubert".to_string()))?;

        let (shape, data) = output_value.try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::InferenceError(format!("Extract error: {}", e)))?;

        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        Tensor::from_vec(data.to_vec(), candle_shape.as_slice(), &Device::Cpu)
            .map_err(|e| e.into())
    }

    #[cfg(not(feature = "onnx"))]
    pub fn extract<P: AsRef<Path>>(&mut self, _audio_path: P) -> Result<Tensor> {
        // Return dummy features: [batch=1, time=100, hidden=768]
        Tensor::zeros((1, 100, 768), candle_core::DType::F32, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Extract features from raw audio samples
    #[cfg(feature = "onnx")]
    pub fn extract_from_samples(&mut self, samples: &[f32]) -> Result<Tensor> {
        let seq_len = samples.len();
        let input_array = (vec![1i64, seq_len as i64], samples.to_vec());

        let inputs = inputs! {
            "input_values" => Value::from_array(input_array)?,
        };

        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        let output_value = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("features"))
            .or_else(|| outputs.get("hidden_states"))
            .ok_or_else(|| crate::Error::InferenceError("No output from Hubert".to_string()))?;

        let (shape, data) = output_value.try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::InferenceError(format!("Extract error: {}", e)))?;

        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        Tensor::from_vec(data.to_vec(), candle_shape.as_slice(), &Device::Cpu)
            .map_err(|e| e.into())
    }

    #[cfg(not(feature = "onnx"))]
    pub fn extract_from_samples(&mut self, _samples: &[f32]) -> Result<Tensor> {
        Tensor::zeros((1, 100, 768), candle_core::DType::F32, &Device::Cpu)
            .map_err(|e| e.into())
    }

    #[cfg(feature = "onnx")]
    fn load_audio<P: AsRef<Path>>(&self, path: P) -> Result<Vec<f32>> {
        use hound::WavReader;

        let mut reader = WavReader::open(path)
            .map_err(|e| crate::Error::AudioError(format!("Failed to open audio: {}", e)))?;

        let spec = reader.spec();
        tracing::debug!("HuBERT load_audio sr={}, channels={}, bits={}", spec.sample_rate, spec.channels, spec.bits_per_sample);

        // Load as float32, mixing down to mono if needed
        let all_samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => match spec.bits_per_sample {
                32 => reader.samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i32::MAX as f32)
                    .collect(),
                _ => reader.samples::<i16>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i16::MAX as f32)
                    .collect(),
            },
            hound::SampleFormat::Float => reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
        };

        // Mix down to mono by averaging channels
        let channels = spec.channels as usize;
        let samples: Vec<f32> = if channels > 1 {
            all_samples.chunks_exact(channels)
                .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            all_samples
        };

        if spec.sample_rate != self.sampling_rate {
            // Resample with sinc filter then add Python's 0.6s silence padding
            let mut resampled = self.resample_sinc(&samples, spec.sample_rate, self.sampling_rate)?;
            // Python: wav16k = np.concatenate([wav16k, np.zeros(int(sr * 0.6))])
            let pad = (self.sampling_rate as f32 * 0.6) as usize;
            resampled.resize(resampled.len() + pad, 0.0);
            Ok(resampled)
        } else {
            // Already at target rate — assume caller pre-processed (including silence if needed)
            Ok(samples)
        }
    }

    /// Resample using libsoxr (HQ quality), matching librosa's default resampler exactly.
    #[cfg(feature = "onnx")]
    fn resample_sinc(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        use soxr::{Soxr, format::Mono};

        let ratio = to_rate as f64 / from_rate as f64;
        let out_capacity = (samples.len() as f64 * ratio).ceil() as usize + 64;
        let mut output = vec![0.0f32; out_capacity];

        let mut resampler = Soxr::<Mono<f32>>::new(from_rate as f64, to_rate as f64)
            .map_err(|e| crate::Error::AudioError(format!("soxr init: {}", e)))?;

        let proc = resampler.process(samples, &mut output)
            .map_err(|e| crate::Error::AudioError(format!("soxr process: {}", e)))?;

        let mut tail = vec![0.0f32; out_capacity];
        let tail_frames = resampler.drain(&mut tail)
            .map_err(|e| crate::Error::AudioError(format!("soxr drain: {}", e)))?;

        output.truncate(proc.output_frames);
        output.extend_from_slice(&tail[..tail_frames]);
        Ok(output)
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
}

impl crate::models::Model for HubertModel {
    #[cfg(feature = "onnx")]
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    #[cfg(not(feature = "onnx"))]
    fn load(_path: &str) -> Result<Self> {
        Self::load("placeholder")
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn to_device(&mut self, device: &str) -> Result<()> {
        self.device = device.to_string();
        Ok(())
    }
}
