//! BERT Feature Extractor with ONNX Runtime (optional)

#[cfg(feature = "onnx")]
use ort::{ep, session::Session, value::Value, inputs};
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;

#[cfg(not(feature = "onnx"))]
use candle_core::{Device, Tensor, DType};
#[cfg(feature = "onnx")]
use candle_core::{Tensor, Device};
use crate::Result;

/// BERT model for text feature extraction
pub struct BertModel {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(feature = "onnx")]
    tokenizer: Tokenizer,
    #[cfg(not(feature = "onnx"))]
    _marker: std::marker::PhantomData<()>,
    device: String,
    #[allow(dead_code)]
    max_length: usize,
}

impl BertModel {
    /// Load BERT model from ONNX file
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

        // Load tokenizer from the same directory as the ONNX model
        let model_dir = std::path::Path::new(path).parent()
            .ok_or_else(|| crate::Error::ModelLoadError("Invalid model path".to_string()))?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| crate::Error::ModelLoadError(format!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e)))?;

        Ok(Self {
            session,
            tokenizer,
            device: device.to_string(),
            max_length: 512,
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
            max_length: 512,
        })
    }

    /// Extract BERT features from text
    #[cfg(feature = "onnx")]
    pub fn extract(&mut self, text: &str) -> Result<Tensor> {
        // Use HuggingFace tokenizer for proper subword tokenization
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| crate::Error::InferenceError(format!("Tokenizer error: {}", e)))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq_len = input_ids.len();

        let input_ids_array = (vec![1i64, seq_len as i64], input_ids);
        let attention_mask_array = (vec![1i64, seq_len as i64], attention_mask);

        let inputs = inputs! {
            "input_ids" => Value::from_array(input_ids_array)?,
            "attention_mask" => Value::from_array(attention_mask_array)?,
        };

        // Run inference
        let outputs = self.session.run(inputs)
            .map_err(|e| crate::Error::InferenceError(format!("ONNX run error: {}", e)))?;

        // Extract features from output — re-exported BERT uses "hidden_state_neg3" (hidden_states[-3] layer)
        let output_value = outputs.get("hidden_state_neg3")
            .or_else(|| outputs.get("last_hidden_state"))
            .ok_or_else(|| crate::Error::InferenceError("No BERT output found".to_string()))?;

        // Try float32 first, fall back to float16
        let candle_shape: Vec<usize>;
        let tensor = if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
            candle_shape = shape.iter().map(|&d| d as usize).collect();
            Tensor::from_vec(data.to_vec(), candle_shape.as_slice(), &Device::Cpu)?
        } else if let Ok((shape, data)) = output_value.try_extract_tensor::<half::f16>() {
            candle_shape = shape.iter().map(|&d| d as usize).collect();
            let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
            Tensor::from_vec(f32_data, candle_shape.as_slice(), &Device::Cpu)?
        } else {
            return Err(crate::Error::InferenceError("Failed to extract BERT output tensor".to_string()));
        };

        Ok(tensor)
    }

    #[cfg(not(feature = "onnx"))]
    pub fn extract(&mut self, _text: &str) -> Result<Tensor> {
        // Return dummy features: [batch=1, hidden=768, seq_len=10]
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
