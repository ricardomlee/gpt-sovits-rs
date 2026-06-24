//! HuBERT feature extractor — pure candle (Wav2Vec2) implementation.

use candle_core::{Device, Tensor};
use crate::Result;
use std::path::Path;

fn device_str(dev: &Device) -> &'static str {
    match dev {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "mps",
    }
}

pub struct HubertModel {
    model: super::wav2vec2::Wav2Vec2Model,
    device: &'static str,
    sampling_rate: u32,
    candle_device: Device,
}

impl HubertModel {
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, candle_core::DType::F32)
    }

    pub fn load_bf16(path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, candle_core::DType::BF16)
    }

    fn load_with_dtype(path: &str, device: &Device, dtype: candle_core::DType) -> Result<Self> {
        let model = super::wav2vec2::Wav2Vec2Model::load_from_file_with_dtype(Path::new(path), device, dtype)?;
        Ok(Self { model, device: device_str(device), sampling_rate: 16000, candle_device: device.clone() })
    }

    pub fn extract<P: AsRef<Path>>(&mut self, audio_path: P) -> Result<Tensor> {
        let samples = self.load_audio(audio_path)?;
        self.extract_from_samples(&samples)
    }

    pub fn extract_from_samples(&mut self, samples: &[f32]) -> Result<Tensor> {
        let n = samples.len();
        let audio = Tensor::from_vec(samples.to_vec(), (1, n), &self.candle_device)?;
        self.model.forward(&audio).map_err(|e| e.into())
    }

    fn load_audio<P: AsRef<Path>>(&self, path: P) -> Result<Vec<f32>> {
        use hound::WavReader;

        let mut reader = WavReader::open(path)
            .map_err(|e| crate::Error::AudioError(format!("Failed to open audio: {}", e)))?;

        let spec = reader.spec();
        tracing::debug!("HuBERT load_audio sr={}, channels={}, bits={}", spec.sample_rate, spec.channels, spec.bits_per_sample);

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

        let channels = spec.channels as usize;
        let samples: Vec<f32> = if channels > 1 {
            all_samples.chunks_exact(channels)
                .map(|ch| ch.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            all_samples
        };

        // Python always appends 0.3s at 32kHz (= 9600 samples) as a zero-pad,
        // which at 16kHz is equivalent to 0.6s. Match this exactly.
        let pad = (self.sampling_rate as f32 * 0.6) as usize;
        if spec.sample_rate != self.sampling_rate {
            let mut resampled = self.resample_sinc(&samples, spec.sample_rate, self.sampling_rate)?;
            resampled.resize(resampled.len() + pad, 0.0);
            Ok(resampled)
        } else {
            let mut padded = samples;
            padded.resize(padded.len() + pad, 0.0);
            Ok(padded)
        }
    }

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

    pub fn device(&self) -> &str { self.device }
    pub fn sampling_rate(&self) -> u32 { self.sampling_rate }
}

impl crate::models::Model for HubertModel {
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    fn device(&self) -> &str { self.device }

    fn to_device(&mut self, _device: &str) -> Result<()> {
        Ok(())
    }
}
