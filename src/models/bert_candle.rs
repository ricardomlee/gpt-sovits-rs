//! BERT feature extractor — delegates to candle-transformers' BertModel.
//!
//! Architecture: chinese-roberta-wwm-ext-large (BERT-large)
//!   vocab_size=21128, hidden=1024, 16 heads, 22 layers (exported up to layer 21,
//!   i.e. the 3rd-from-last hidden state of the full 24-layer model).
//!
//! Weight key compat: our safetensors uses `layer_norm` (lowercase) while
//! candle-transformers expects `LayerNorm`. Keys are renamed at load time.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, PositionEmbeddingType};
use tokenizers::Tokenizer;

pub const DTYPE_F32: DType = DType::F32;
pub const DTYPE_BF16: DType = DType::BF16;

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
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl BertCandleModel {
    pub fn load(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
    ) -> crate::Result<Self> {
        Self::load_with_dtype(weights_path, tokenizer_path, device, DType::F32)
    }

    pub fn load_bf16(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
    ) -> crate::Result<Self> {
        Self::load_with_dtype(weights_path, tokenizer_path, device, DType::BF16)
    }

    pub fn load_with_dtype(
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> crate::Result<Self> {
        // Rename layer_norm → LayerNorm to match HuggingFace canonical key names
        let weights = candle_core::safetensors::load(weights_path, device)?
            .into_iter()
            .map(|(k, v)| (k.replace("layer_norm", "LayerNorm"), v))
            .collect::<std::collections::HashMap<_, _>>();

        let vb = VarBuilder::from_tensors(weights, dtype, device);
        let config = chinese_roberta_large_config();
        let model = BertModel::load(vb, &config)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| crate::Error::ModelLoadError(format!("tokenizer: {e}")))?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    /// Returns [1, seq_len, 1024] in F32 (caller always expects F32).
    pub fn extract(&self, text: &str) -> crate::Result<Tensor> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| crate::Error::InferenceError(format!("tokenize: {e}")))?;

        let ids: Vec<u32> = enc.get_ids().to_vec();
        let mask: Vec<u32> = enc.get_attention_mask().iter().map(|&m| m as u32).collect();
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
