//! Hubert Feature Extractor with ONNX Runtime support
//!
//! Compile with --features onnx to enable ONNX Runtime

#[cfg(feature = "onnx")]
use ort::{Session, Value, inputs};

use candle_core::{Device, Tensor, DType};
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
        let session = Session::builder()?
            .commit_from_file(path)
            .map_err(|e| crate::Error::ModelLoadError(format!("Failed to load ONNX: {}", e)))?;

        Ok(Self {
            session,
            device: "cpu".to_string(),
            sampling_rate: 16000,
        })
    }

    /// Load Hubert model (placeholder when ONNX feature is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn load(_path: &str) -> Result<Self> {
        Ok(Self {
            _marker: std::marker::PhantomData,
            device: "cpu".to_string(),
            sampling_rate: 16000,
        })
    }

    /// Extract Hubert features from audio file
    #[cfg(feature = "onnx")]
    pub fn extract<P: AsRef<Path>>(&self, audio_path: P) -> Result<Tensor> {
        let audio_data = self.load_audio(audio_path)?;

        let inputs = inputs! {
            "source" => Value::from_array(audio_data)?,
        }.map_err(|e| crate::Error::InferenceError(format!("ONNX inputs error: {}", e)))?;

        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        let output_tensor = outputs["features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::InferenceError(format!("Extract error: {}", e)))?;

        let shape: Vec<usize> = output_tensor.shape().dims().iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = output_tensor.iter().copied().collect();

        Tensor::from_vec(data, &shape, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Extract Hubert features from audio file (placeholder when ONNX feature is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn extract<P: AsRef<Path>>(&self, _audio_path: P) -> Result<Tensor> {
        Tensor::zeros((1, 100, 768), DType::F32, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Extract features from raw audio samples
    #[cfg(feature = "onnx")]
    pub fn extract_from_samples(&self, samples: &[f32]) -> Result<Tensor> {
        let inputs = inputs! {
            "source" => Value::from_array(samples.to_vec())?,
        }.map_err(|e| crate::Error::InferenceError(format!("ONNX inputs error: {}", e)))?;

        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        let output_tensor = outputs["features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::InferenceError(format!("Extract error: {}", e)))?;

        let shape: Vec<usize> = output_tensor.shape().dims().iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = output_tensor.iter().copied().collect();

        Tensor::from_vec(data, &shape, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Extract features from raw audio samples (placeholder)
    #[cfg(not(feature = "onnx"))]
    pub fn extract_from_samples(&self, _samples: &[f32]) -> Result<Tensor> {
        Tensor::zeros((1, 100, 768), DType::F32, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Load audio file and return normalized samples
    #[cfg(feature = "onnx")]
    fn load_audio<P: AsRef<Path>>(&self, path: P) -> Result<Vec<f32>> {
        use hound::WavReader;

        let reader = WavReader::open(path)
            .map_err(|e| crate::Error::AudioError(format!("Failed to open audio: {}", e)))?;

        let spec = reader.spec();
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                reader.read_samples::<i16>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 / i16::MAX as f32)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader.read_samples::<f32>()
                    .filter_map(|s| s.ok())
                    .collect()
            }
        };

        if spec.sample_rate != self.sampling_rate {
            Ok(self.resample(&samples, spec.sample_rate, self.sampling_rate))
        } else {
            Ok(samples)
        }
    }

    /// Simple linear interpolation resampler
    #[cfg(feature = "onnx")]
    fn resample(&self, samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let ratio = from_rate as f64 / to_rate as f64;
        let new_len = (samples.len() as f64 / ratio) as usize;

        (0..new_len)
            .map(|i| {
                let pos = (i as f64 * ratio) as usize;
                let frac = (i as f64 * ratio) - pos as f64;
                if pos + 1 < samples.len() {
                    (samples[pos] * (1.0 - frac as f32)) + (samples[pos + 1] * frac as f32)
                } else {
                    samples.get(pos).copied().unwrap_or(0.0)
                }
            })
            .collect()
    }

    /// Get model device
    pub fn device(&self) -> &str {
        &self.device
    }

    /// Get sampling rate
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
