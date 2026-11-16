//! Models Module
//!
//! Neural network models for GPT-SoVITS

pub mod bert;
/// Compatibility exports; BERT implementation lives in [`bert`].
pub mod bert_candle {
    pub use super::bert::{BertCandleModel, DTYPE_BF16, DTYPE_F32};
}
pub mod bigvgan;
pub mod gpt;
pub mod hubert;
pub mod mrte;
pub mod semantic_tokenizer;
pub mod sovits;
pub mod sovits_decoder;
pub mod sovits_encp;
pub mod sovits_encq;
pub mod sovits_flow;
pub mod sovits_ref_enc;
pub mod transformer;
pub mod wav2vec2;

pub use bert::{BertCandleModel, BertModel};
pub use bigvgan::BigVGAN;
pub use gpt::GPTModel;
pub use hubert::HubertModel;
pub use mrte::MRTE;
pub use semantic_tokenizer::SemanticTokenizer;
pub use sovits::SoVITSModel;
pub use sovits_encq::EncQ;
pub use transformer::{MultiHeadAttention, Transformer, TransformerBlock, TransformerConfig};
pub use wav2vec2::Wav2Vec2Model;

use crate::{Error, Result};

/// Trait for all models
pub trait Model: Send + Sync {
    /// Load model from file
    fn load(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// Get model device
    fn device(&self) -> &str;

    /// Accept the current device as a no-op; runtime weight migration is unsupported.
    /// Load a fresh model on the requested device instead.
    fn to_device(&mut self, device: &str) -> Result<()> {
        let current = self.device();
        if device.eq_ignore_ascii_case(current) {
            return Ok(());
        }
        Err(Error::ConfigError(format!(
            "Runtime model device migration from {current} to {device} is not supported; reload the model on the requested device"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDevice(&'static str);

    impl Model for FixedDevice {
        fn load(_: &str) -> Result<Self> {
            Ok(Self("cpu"))
        }

        fn device(&self) -> &str {
            self.0
        }
    }

    #[test]
    fn device_policy_does_not_initialize_or_relabel_any_backend() {
        for current in ["cpu", "cuda", "mps"] {
            let mut model = FixedDevice(current);
            model.to_device(current).unwrap();
            model.to_device(&current.to_ascii_uppercase()).unwrap();
            for requested in ["cpu", "cuda", "mps", "auto", "invalid", "cuda:1", ""] {
                if requested != current {
                    let error = model.to_device(requested).unwrap_err();
                    assert!(matches!(error, Error::ConfigError(_)), "{error}");
                    assert!(error.to_string().contains("reload the model"));
                }
                assert_eq!(model.device(), current);
            }
        }
    }
}
