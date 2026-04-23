//! Model Weights Loader
//!
//! Utilities for loading and converting model weights from safetensors format

use candle_core::{DType, Device, Tensor, D};
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

    /// Get weight-parameterized conv1d for BigVGAN/SoVITS
    pub fn get_conv1d_weight_norm(&self, prefix: &str) -> Result<Conv1dWeightNorm> {
        let weight_g = self.get(&format!("{}.weight_g", prefix))?.clone();
        let weight_v = self.get(&format!("{}.weight_v", prefix))?.clone();
        let bias = self.get(&format!("{}.bias", prefix)).ok().cloned();

        // Infer kernel size from weight_v shape [out_channels, in_channels, kernel_size]
        let weight_v_shape = weight_v.dims();
        let kernel_size = if weight_v_shape.len() >= 3 {
            weight_v_shape[2]
        } else {
            1
        };

        // Calculate padding to maintain sequence length: padding = (kernel_size - 1) / 2
        let padding = (kernel_size - 1) / 2;

        Ok(Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, padding, 1)?)
    }

    /// Get internal HashMap for VarBuilder
    pub fn as_hash_map(&self) -> &HashMap<String, Tensor> {
        &self.data
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
        self.embedding(input)
    }

    /// Lookup embeddings for input indices
    /// Handles both 1D [seq_len] and 2D [batch, seq_len] inputs
    /// Output: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim] for 1D input
    pub fn embedding(&self, input: &Tensor) -> Result<Tensor> {
        let input_dims = input.dims();

        if input_dims.len() == 1 {
            // 1D input: [seq_len] -> [seq_len, hidden_dim]
            let seq_len = input_dims[0];
            let indices: Vec<i64> = input.to_vec1()?;
            let mut result = Vec::with_capacity(seq_len);
            for &idx in &indices {
                let emb = self.weight.get(idx as usize)?;
                result.push(emb);
            }
            Ok(Tensor::stack(&result, 0)?)
        } else if input_dims.len() == 2 {
            // 2D input: [batch, seq_len] -> [batch, seq_len, hidden_dim]
            let batch = input_dims[0];
            let seq_len = input_dims[1];
            let mut result = Vec::with_capacity(batch);
            for b in 0..batch {
                let batch_indices: Vec<i64> = input.narrow(0, b, 1)?.squeeze(0)?.to_vec1()?;
                let mut batch_embs = Vec::with_capacity(seq_len);
                for &idx in &batch_indices {
                    let emb = self.weight.get(idx as usize)?;
                    batch_embs.push(emb);
                }
                let batch_tensor = Tensor::stack(&batch_embs, 0)?;
                result.push(batch_tensor);
            }
            Ok(Tensor::stack(&result, 0)?)
        } else {
            use candle_core::Shape;
            Err(candle_core::Error::UnexpectedShape {
                msg: "Embedding input must be 1D or 2D".to_string(),
                expected: Shape::from(&[1usize]),
                got: Shape::from(input.dims()),
            }.into())
        }
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
        // Handle both 2D [batch, hidden] and 3D [batch, seq, hidden] inputs
        let input_dims = input.dims();
        let weight_t = self.weight.t()?;

        let output = if input_dims.len() == 3 {
            // 3D input: [batch, seq, hidden] -> reshape to [batch*seq, hidden] -> matmul -> reshape back
            let (batch, seq, hidden) = (input_dims[0], input_dims[1], input_dims[2]);
            let flat = input.reshape((batch * seq, hidden))?;
            let result = flat.matmul(&weight_t)?;
            result.reshape((batch, seq, self.weight.dims()[0]))?
        } else {
            // 2D input: [batch, hidden]
            input.matmul(&weight_t)?
        };

        match &self.bias {
            Some(bias) => {
                let bias_len = bias.dims()[0];
                // Convert bias to match output dtype
                let bias_converted = bias.to_dtype(output.dtype())?;
                if output.dims().len() == 3 {
                    let bias_reshaped = bias_converted.reshape(&[1, 1, bias_len])?;
                    Ok(output.broadcast_add(&bias_reshaped)?)
                } else {
                    let bias_reshaped = bias_converted.reshape(&[1, bias_len])?;
                    Ok(output.broadcast_add(&bias_reshaped)?)
                }
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
        // Auto-detect format based on input shape vs weight dimension
        let input_dims = input.dims();
        let rank = input_dims.len();
        let norm_dim = self.weight.dims()[0];

        // Detect format:
        // - [batch, seq_len, channels] (Transformer): last dim == norm_dim → normalize over D::Minus1
        // - [batch, channels, seq_len] (SoVITS): middle dim == norm_dim → normalize over D::Minus2
        let is_last_dim = rank >= 2 && input_dims[rank - 1] == norm_dim;
        let is_middle_dim = rank >= 3 && input_dims[1] == norm_dim;

        let dim = if is_last_dim {
            candle_core::D::Minus1  // Transformer format
        } else if is_middle_dim {
            candle_core::D::Minus2  // SoVITS format
        } else {
            // Fallback: try last dimension (standard PyTorch behavior)
            candle_core::D::Minus1
        };

        // Mean/var over the detected dimension
        let mean = input.mean_keepdim(dim)?;
        let centered = input.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(dim)?;

        // Convert eps to match input dtype
        let eps_val = self.eps as f32;
        let eps = Tensor::full(eps_val, var.dims(), &var.device())?;
        let eps = eps.to_dtype(var.dtype())?;
        let std = var.add(&eps)?.sqrt()?;
        let normalized = centered.broadcast_div(&std)?;

        // Apply weight and bias with proper shape
        let mut shape = vec![1; rank];
        if is_last_dim {
            shape[rank - 1] = norm_dim;  // Transformer: [1, 1, channels]
        } else {
            shape[1] = norm_dim;  // SoVITS: [1, channels, 1]
        }

        let weight = self.weight.to_device(&input.device())?.to_dtype(normalized.dtype())?;
        let bias = self.bias.to_device(&input.device())?.to_dtype(normalized.dtype())?;
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

/// Weight-parameterized Conv1d for BigVGAN/SoVITS
/// Uses weight_g (norm) and weight_v (direction) decomposition
/// Caches pre-computed weight to avoid recomputing norm on every forward pass
#[derive(Debug, Clone)]
pub struct Conv1dWeightNorm {
    pub weight_g: Tensor,
    pub weight_v: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    /// Pre-computed normalized weight (computed once during loading)
    cached_weight: Option<Tensor>,
}

impl Conv1dWeightNorm {
    pub fn new(
        weight_g: Tensor,
        weight_v: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            weight_g,
            weight_v,
            bias,
            stride,
            padding,
            dilation,
            cached_weight: None,
        }
    }

    /// Create with pre-computed weight (preferred for inference)
    pub fn new_with_cached(
        weight_g: Tensor,
        weight_v: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        // Compute weight once during construction
        let v_squared = weight_v.sqr()?;
        let v_norm = v_squared.sum(D::Minus1)?.sum(D::Minus1)?.sqrt()?;
        let out_channels = weight_v.dims()[0];
        let v_norm_reshaped = v_norm.reshape((out_channels, 1, 1))?;
        let v_normalized = weight_v.broadcast_div(&v_norm_reshaped)?;
        let weight = v_normalized.broadcast_mul(&weight_g)?;

        Ok(Self {
            weight_g,
            weight_v,
            bias,
            stride,
            padding,
            dilation,
            cached_weight: Some(weight),
        })
    }

    /// Compute actual weight from g/v decomposition
    pub fn get_weight(&self) -> Result<Tensor> {
        // Use cached weight if available
        if let Some(ref w) = self.cached_weight {
            return Ok(w.clone());
        }

        // weight = (weight_v / ||weight_v||) * weight_g
        // weight_g: [out_channels, in_channels, kernel] (same shape as weight_v)
        // weight_v: [out_channels, in_channels, kernel]
        // Need to compute norm per output channel

        // Compute ||weight_v|| per output channel
        // weight_v: [out_ch, in_ch, kernel] -> norm: [out_ch]
        let v_squared = self.weight_v.sqr()?;
        // Sum over in_channels and kernel dimensions (dims 1 and 2)
        let v_sum_in = v_squared.sum(D::Minus1)?; // [out_ch, in_ch] -> [out_ch]
        let v_sum = v_sum_in.sum(D::Minus1)?; // [out_ch]
        let v_norm = v_sum.sqrt()?; // [out_ch]

        // Reshape v_norm to [out_ch, 1, 1] for broadcasting
        let weight_v_dims = self.weight_v.dims();
        let out_channels = weight_v_dims[0];
        let v_norm_reshaped = v_norm.reshape((out_channels, 1, 1))?;

        // weight = weight_v / v_norm * weight_g
        let v_normalized = self.weight_v.broadcast_div(&v_norm_reshaped)?;
        Ok(v_normalized.broadcast_mul(&self.weight_g)?)
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.get_weight()?;
        let output = input.conv1d(
            &weight,
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
