//! BERT Feature Extractor with ONNX Runtime support
//!
//! Compile with --features onnx to enable ONNX Runtime

#[cfg(feature = "onnx")]
use ort::{Session, Value, inputs};

use candle_core::{Device, Tensor, DType};
use crate::Result;

/// BERT model for text feature extraction
pub struct BertModel {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(not(feature = "onnx"))]
    _marker: std::marker::PhantomData<()>,
    device: String,
    #[cfg(feature = "onnx")]
    max_length: usize,
}

impl BertModel {
    /// Load BERT model from ONNX file
    #[cfg(feature = "onnx")]
    pub fn load(path: &str) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .map_err(|e| crate::Error::ModelLoadError(format!("Failed to load ONNX: {}", e)))?;

        Ok(Self {
            session,
            device: "cpu".to_string(),
            max_length: 512,
        })
    }

    /// Load BERT model (placeholder when ONNX feature is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn load(_path: &str) -> Result<Self> {
        Ok(Self {
            _marker: std::marker::PhantomData,
            device: "cpu".to_string(),
        })
    }

    /// Extract BERT features from text
    #[cfg(feature = "onnx")]
    pub fn extract(&self, text: &str) -> Result<Tensor> {
        // Tokenize input (simple word-level tokenization for now)
        let tokens: Vec<&str> = text.split_whitespace().take(self.max_length).collect();
        let input_ids: Vec<i64> = tokens.iter().map(|t| t.len() as i64).collect();

        // Create ONNX inputs
        let inputs = inputs! {
            "input_ids" => Value::from_array(input_ids)?,
        }.map_err(|e| crate::Error::InferenceError(format!("ONNX inputs error: {}", e)))?;

        // Run inference
        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        // Extract features from output
        let output_tensor = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::InferenceError(format!("Extract error: {}", e)))?;

        // Convert to Candle tensor
        let shape: Vec<usize> = output_tensor.shape().dims().iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = output_tensor.iter().copied().collect();

        Tensor::from_vec(data, &shape, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Extract BERT features from text (placeholder when ONNX feature is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn extract(&self, _text: &str) -> Result<Tensor> {
        // Return dummy features for now
        // Shape: [batch=1, hidden=768, seq_len=10]
        Tensor::zeros((1, 768, 10), DType::F32, &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Get model device
    pub fn device(&self) -> &str {
        &self.device
    }
}

impl crate::models::Model for BertModel {
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
