//! Hubert Feature Extractor (placeholder - ONNX Runtime disabled)

use candle_core::{Device, Tensor, DType};
use crate::Result;
use std::path::Path;

/// Hubert model for audio feature extraction (placeholder)
pub struct HubertModel {
    device: String,
}

impl HubertModel {
    /// Load Hubert model from ONNX file (placeholder)
    pub fn load(_path: &str) -> Result<Self> {
        // Placeholder - ONNX Runtime disabled until protoc is installed
        Ok(Self {
            device: "cpu".to_string(),
        })
    }

    /// Extract Hubert features from audio file (placeholder)
    pub fn extract<P: AsRef<Path>>(&self, _audio_path: P) -> Result<Tensor> {
        // Return dummy features
        let features = Tensor::zeros((1, 100, 768), DType::F32, &Device::Cpu)?;
        Ok(features)
    }

    /// Extract features from raw audio samples (placeholder)
    pub fn extract_from_samples(&self, _samples: &[f32], _sample_rate: u32) -> Result<Tensor> {
        let features = Tensor::zeros((1, 100, 768), DType::F32, &Device::Cpu)?;
        Ok(features)
    }

    /// Get model device
    pub fn device(&self) -> &str {
        &self.device
    }
}

impl crate::models::Model for HubertModel {
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
