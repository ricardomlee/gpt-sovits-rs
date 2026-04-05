//! Model Weights Loader
//!
//! Utilities for loading and converting model weights from safetensors format

use candle_core::{DType, Device, Tensor};
use safetensors::{SafeTensors, tensor::Dtype};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{Error, Result};

/// Load weights from a safetensors file
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();

    // Read file contents
    let mut file = File::open(path)
        .map_err(|e| Error::ModelLoadError(format!("Failed to open file: {}", e)))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| Error::ModelLoadError(format!("Failed to read file: {}", e)))?;

    // Parse safetensors
    let safetensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::ModelLoadError(format!("Failed to parse safetensors: {}", e)))?;

    // Convert to Candle tensors
    let mut weights = HashMap::new();
    let device = Device::Cpu;

    for name in safetensors.names() {
        let view = safetensors
            .tensor(name)
            .map_err(|e| Error::ModelLoadError(format!("Failed to get tensor {}: {}", name, e)))?;

        let dtype = match view.dtype() {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::I64 => DType::I64,
            other => {
                return Err(Error::ModelLoadError(format!(
                    "Unsupported dtype: {:?}",
                    other
                )))
            }
        };

        let tensor = Tensor::from_raw_buffer(
            view.data(),
            dtype,
            view.shape(),
            &device,
        ).map_err(|e| Error::ModelLoadError(format!("Failed to create tensor {}: {}", name, e)))?;

        weights.insert(name.to_string(), tensor);
    }

    Ok(weights)
}

/// Extract weights with a specific prefix
pub fn extract_prefix(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
    weights
        .iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .map(|(k, v)| {
            let new_key = k.strip_prefix(prefix).unwrap_or(k).to_string();
            (new_key, v.clone())
        })
        .collect()
}

/// Rename keys in weights map
pub fn rename_keys(
    weights: HashMap<String, Tensor>,
    renames: &HashMap<String, String>,
) -> HashMap<String, Tensor> {
    weights
        .into_iter()
        .map(|(k, v)| {
            let new_key = renames.get(&k).cloned().unwrap_or(k);
            (new_key, v)
        })
        .collect()
}

/// Model state dict wrapper
#[derive(Debug, Clone)]
pub struct StateDict {
    data: HashMap<String, Tensor>,
}

impl StateDict {
    pub fn new(data: HashMap<String, Tensor>) -> Self {
        Self { data }
    }

    pub fn get(&self, key: &str) -> Result<&Tensor> {
        self.data.get(key).ok_or_else(|| {
            Error::ModelLoadError(format!("Key '{}' not found in state dict", key))
        })
    }

    pub fn remove(&mut self, key: &str) -> Result<Tensor> {
        self.data.remove(key).ok_or_else(|| {
            Error::ModelLoadError(format!("Key '{}' not found in state dict", key))
        })
    }

    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Get a tensor and reshape it
    pub fn get_reshaped(&self, key: &str, shape: &[usize]) -> Result<Tensor> {
        let tensor = self.get(key)?;
        tensor.reshape(shape).map_err(|e| {
            Error::ModelLoadError(format!("Failed to reshape {}: {}", key, e))
        })
    }

    /// Get embedding weights
    pub fn get_embedding(&self, key: &str) -> Result<Embedding> {
        let weight = self.get(key)?.clone();
        Ok(Embedding::new(weight))
    }

    /// Get linear layer weights
    pub fn get_linear(&self, prefix: &str) -> Result<Linear> {
        let weight = self.get(&format!("{}.weight", prefix))?.clone();
        let bias = self.get(&format!("{}.bias", prefix)).ok().cloned();
        Ok(Linear::new(weight, bias))
    }

    /// Get layer norm weights
    pub fn get_layer_norm(&self, prefix: &str) -> Result<LayerNorm> {
        let weight = self.get(&format!("{}.weight", prefix))?.clone();
        let bias = self.get(&format!("{}.bias", prefix))?.clone();
        Ok(LayerNorm::new(weight, bias))
    }
}

/// Embedding layer
#[derive(Debug, Clone)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simple embedding lookup
        self.weight.index_select(input, 0).map_err(|e| e.into())
    }

    pub fn num_embeddings(&self) -> usize {
        self.weight.dims()[0]
    }

    pub fn embedding_dim(&self) -> usize {
        self.weight.dims()[1]
    }
}

/// Linear layer
#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Matrix multiplication: input @ weight.T
        let weight_t = self.weight.t()?;
        let output = input.matmul(&weight_t)?;

        match &self.bias {
            Some(bias) => {
                let bias_len = bias.dims()[0];
                let bias_reshaped = bias.reshape(&[1, 1, bias_len])?;
                Ok(output.broadcast_add(&bias_reshaped)?)
            }
            None => Ok(output),
        }
    }

    pub fn in_features(&self) -> usize {
        let dims = self.weight.dims();
        if dims.len() >= 2 {
            dims[1]
        } else {
            0
        }
    }

    pub fn out_features(&self) -> usize {
        self.weight.dims()[0]
    }
}

/// Layer normalization
#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self {
            weight,
            bias,
            eps: 1e-5,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Normalize: (input - mean) / sqrt(var + eps)
        let mean = input.mean_keepdim(candle_core::D::Minus1)?;
        let centered = input.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let eps = Tensor::full(self.eps, var.dims(), &var.device())?;
        let std = var.add(&eps)?.sqrt()?;
        let normalized = centered.broadcast_div(&std)?;

        // Apply weight and bias - reshape to match last dimension
        let last_dim = input.rank();
        let mut shape = vec![1; last_dim];
        shape[last_dim - 1] = self.weight.dims()[0];

        // Convert weight and bias to match normalized tensor dtype
        let weight = self.weight.to_dtype(normalized.dtype())?;
        let bias = self.bias.to_dtype(normalized.dtype())?;
        let weight_reshaped = weight.reshape(&*shape)?;
        let bias_reshaped = bias.reshape(&*shape)?;

        let output = normalized.broadcast_mul(&weight_reshaped)?;
        output.broadcast_add(&bias_reshaped).map_err(|e| e.into())
    }
}

/// 2D Convolution layer
#[derive(Debug, Clone)]
pub struct Conv1d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl Conv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.conv1d(
            &self.weight,
            self.padding,
            self.stride,
            self.dilation,
            1, // groups
        )?;

        match &self.bias {
            Some(bias) => {
                let bias_len = bias.dims()[0];
                let bias_reshaped = bias.reshape(&[1, bias_len, 1])?;
                Ok(output.broadcast_add(&bias_reshaped)?)
            }
            None => Ok(output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dict() {
        let device = Device::Cpu;
        let mut data = HashMap::new();
        data.insert("layer.weight".to_string(), Tensor::ones((10, 5), DType::F32, &device).unwrap());
        data.insert("layer.bias".to_string(), Tensor::zeros(5, DType::F32, &device).unwrap());

        let sd = StateDict::new(data);
        assert!(sd.contains("layer.weight"));
        assert!(sd.contains("layer.bias"));
        assert!(!sd.contains("nonexistent"));
    }
}
