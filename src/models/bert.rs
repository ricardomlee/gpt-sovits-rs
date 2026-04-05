//! BERT Feature Extractor (placeholder - ONNX Runtime disabled)

use candle_core::{Device, Tensor, DType};
use crate::Result;

/// BERT model for text feature extraction (placeholder)
pub struct BertModel {
    device: String,
}

impl BertModel {
    /// Load BERT model from ONNX file (placeholder)
    pub fn load(_path: &str) -> Result<Self> {
        // Placeholder - ONNX Runtime disabled until protoc is installed
        Ok(Self {
            device: "cpu".to_string(),
        })
    }

    /// Extract BERT features from text (placeholder)
    pub fn extract(&self, _text: &str) -> Result<Tensor> {
        // Return dummy features for now
        let features = Tensor::zeros((1, 768, 10), DType::F32, &Device::Cpu)?;
        Ok(features)
    }

    /// Get model device
    pub fn device(&self) -> &str {
        &self.device
    }
}

impl crate::models::Model for BertModel {
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
