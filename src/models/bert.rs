//! BERT Feature Extractor with ONNX Runtime

use ort::{session::Session, value::Value, inputs};
use candle_core::{Device, Tensor};
use crate::Result;

/// BERT model for text feature extraction
pub struct BertModel {
    session: Session,
    device: String,
    max_length: usize,
}

impl BertModel {
    /// Load BERT model from ONNX file
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

    /// Extract BERT features from text
    pub fn extract(&mut self, text: &str) -> Result<Tensor> {
        // Simple tokenization - convert chars to IDs
        let chars: Vec<char> = text.chars().take(self.max_length).collect();
        let input_ids: Vec<i64> = chars.iter().map(|&c| c as i64).collect();
        let seq_len = input_ids.len();

        // Create attention mask (all 1s for valid tokens)
        let attention_mask: Vec<i64> = vec![1; seq_len];

        // Create ONNX inputs - need (shape, data) tuple format
        let input_ids_array = (vec![1i64, seq_len as i64], input_ids);
        let attention_mask_array = (vec![1i64, seq_len as i64], attention_mask);

        let inputs = inputs! {
            "input_ids" => Value::from_array(input_ids_array)?,
            "attention_mask" => Value::from_array(attention_mask_array)?,
        };

        // Run inference
        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        // Extract features from output - use known output name
        let output_value = outputs.get("last_hidden_state")
            .ok_or_else(|| crate::Error::InferenceError("No 'last_hidden_state' output from BERT".to_string()))?;

        // Extract tensor data: try_extract_tensor returns (&Shape, &[T])
        let (shape, data) = output_value.try_extract_tensor::<f32>()
            .map_err(|e| crate::Error::InferenceError(format!("Extract error: {}", e)))?;

        // Convert shape to Candle format (Shape derefs to [i64])
        let candle_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        Tensor::from_vec(data.to_vec(), candle_shape.as_slice(), &Device::Cpu)
            .map_err(|e| e.into())
    }

    /// Get model device
    pub fn device(&self) -> &str {
        &self.device
    }
}

impl crate::models::Model for BertModel {
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn to_device(&mut self, device: &str) -> Result<()> {
        self.device = device.to_string();
        Ok(())
    }
}
