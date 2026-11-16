//! BERT feature extractor — pure candle (chinese-roberta-wwm-ext-large).
//!
//! `BertModel` provides shared ownership and tokenizer discovery;
//! `BertCandleModel` adapts converted weights and tokenization to Candle's BERT.

use crate::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{
    BertModel as CandleBertModel, Config, HiddenAct, PositionEmbeddingType,
};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub const DTYPE_F32: DType = DType::F32;
pub const DTYPE_BF16: DType = DType::BF16;

fn device_str(dev: &Device) -> &'static str {
    match dev {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "mps",
    }
}

#[derive(Clone)]
pub struct BertModel {
    model: Arc<BertCandleModel>,
    device: &'static str,
}

impl BertModel {
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load from safetensors path. Tokenizer is looked up as `{dir}/tokenizer.json`
    /// or `models/bert/tokenizer.json` as fallback.
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, DType::F32)
    }

    pub fn load_bf16(path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, DType::BF16)
    }

    fn load_with_dtype(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let weights_path = std::path::Path::new(path);
        let tokenizer_path = weights_path.with_file_name("tokenizer.json");
        let tokenizer_path = if tokenizer_path.exists() {
            tokenizer_path
        } else {
            std::path::PathBuf::from("models/bert/tokenizer.json")
        };
        let model = BertCandleModel::load_with_dtype(weights_path, &tokenizer_path, device, dtype)?;
        Ok(Self {
            model: Arc::new(model),
            device: device_str(device),
        })
    }

    pub fn extract(&mut self, text: &str) -> Result<Tensor> {
        self.model.extract(text)
    }

    pub fn device(&self) -> &str {
        self.device
    }
}

impl crate::models::Model for BertModel {
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    fn device(&self) -> &str {
        self.device
    }
}

/// Exported BERT-large configuration, using the third-from-last hidden state.
fn chinese_roberta_large_config() -> Config {
    Config {
        vocab_size: 21128,
        hidden_size: 1024,
        num_hidden_layers: 22,
        num_attention_heads: 16,
        intermediate_size: 4096,
        hidden_act: HiddenAct::Gelu,
        hidden_dropout_prob: 0.0,
        max_position_embeddings: 512,
        type_vocab_size: 2,
        initializer_range: 0.02,
        layer_norm_eps: 1e-12,
        pad_token_id: 0,
        position_embedding_type: PositionEmbeddingType::Absolute,
        use_cache: false,
        classifier_dropout: None,
        model_type: Some("bert".to_string()),
    }
}

pub struct BertCandleModel {
    model: CandleBertModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    max_length: usize,
    vocab_size: usize,
}

fn validate_token_ids(ids: &[u32], max_length: usize, vocab_size: usize) -> Result<()> {
    if ids.len() > max_length {
        return Err(crate::Error::InferenceError(format!(
            "BERT input has {} tokens; maximum is {max_length}; split the text into shorter chunks",
            ids.len()
        )));
    }
    if let Some(id) = ids.iter().find(|id| **id as usize >= vocab_size) {
        return Err(crate::Error::InferenceError(format!(
            "BERT token ID {id} is outside vocabulary size {vocab_size}; check tokenizer.json"
        )));
    }
    Ok(())
}

impl BertCandleModel {
    pub fn load(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
    ) -> Result<Self> {
        Self::load_with_dtype(weights_path, tokenizer_path, device, DType::F32)
    }

    pub fn load_bf16(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
    ) -> Result<Self> {
        Self::load_with_dtype(weights_path, tokenizer_path, device, DType::BF16)
    }

    pub fn load_with_dtype(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Converted checkpoints use lowercase layer_norm; Candle expects LayerNorm.
        let weights = candle_core::safetensors::load(weights_path, device)?
            .into_iter()
            .map(|(k, v)| (k.replace("layer_norm", "LayerNorm"), v))
            .collect::<std::collections::HashMap<_, _>>();

        let vb = VarBuilder::from_tensors(weights, dtype, device);
        let config = chinese_roberta_large_config();
        let model = CandleBertModel::load(vb, &config)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| crate::Error::ModelLoadError(format!("tokenizer: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
            max_length: config.max_position_embeddings,
            vocab_size: config.vocab_size,
        })
    }

    /// Returns [1, seq_len, 1024] in F32 (caller always expects F32).
    pub fn extract(&self, text: &str) -> Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| crate::Error::InferenceError(format!("tokenize: {e}")))?;

        let ids: Vec<u32> = enc.get_ids().to_vec();
        // Validate on the host: CUDA index errors can poison subsequent requests.
        validate_token_ids(&ids, self.max_length, self.vocab_size)?;
        let mask: Vec<u32> = enc.get_attention_mask().to_vec();
        let seq = ids.len();

        let input_ids = Tensor::from_vec(ids, (1, seq), &self.device)?;
        let token_type_ids = Tensor::zeros((1, seq), DType::U32, &self.device)?;
        let attn_mask = Tensor::from_vec(mask, (1, seq), &self.device)?;

        let out = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attn_mask))?;

        if self.dtype != DType::F32 {
            Ok(out.to_dtype(DType::F32)?)
        } else {
            Ok(out)
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_exports_preserve_the_bert_api() {
        let _: fn(&str) -> Result<crate::models::BertModel> = BertModel::load;
        let _: fn(
            &std::path::Path,
            &std::path::Path,
            &Device,
        ) -> Result<crate::models::BertCandleModel> =
            crate::models::bert_candle::BertCandleModel::load;
        assert_eq!(crate::models::bert_candle::DTYPE_F32, DTYPE_F32);
        assert_eq!(crate::models::bert_candle::DTYPE_BF16, DTYPE_BF16);
    }

    #[test]
    fn validates_position_and_vocabulary_bounds_before_gpu_indexing() {
        assert!(validate_token_ids(&vec![21127; 512], 512, 21128).is_ok());
        let long = validate_token_ids(&vec![0; 513], 512, 21128).unwrap_err();
        assert!(long.to_string().contains("513 tokens; maximum is 512"));
        let invalid = validate_token_ids(&[101, 21128, 102], 512, 21128).unwrap_err();
        assert!(invalid.to_string().contains("outside vocabulary"));
    }
}
