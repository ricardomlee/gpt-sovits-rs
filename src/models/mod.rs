//! Models Module
//!
//! Neural network models for GPT-SoVITS

pub mod bert;
pub mod bert_candle;
pub mod bigvgan;
pub mod gpt;
pub mod hubert;
pub mod wav2vec2;
pub mod mrte;
pub mod sovits;
pub mod sovits_encp;
pub mod sovits_encq;
pub mod sovits_flow;
pub mod sovits_decoder;
pub mod sovits_ref_enc;
pub mod transformer;
pub mod semantic_tokenizer;

pub use bert::BertModel;
pub use bert_candle::BertCandleModel;
pub use bigvgan::BigVGAN;
pub use gpt::GPTModel;
pub use hubert::HubertModel;
pub use wav2vec2::Wav2Vec2Model;
pub use mrte::MRTE;
pub use sovits::SoVITSModel;
pub use sovits_encq::EncQ;
pub use transformer::{Transformer, TransformerConfig, TransformerBlock, MultiHeadAttention};
pub use semantic_tokenizer::SemanticTokenizer;

use crate::Result;

/// Trait for all models
pub trait Model: Send + Sync {
    /// Load model from file
    fn load(path: &str) -> Result<Self>
    where
        Self: Sized;

    /// Get model device
    fn device(&self) -> &str;

    /// Move model to device
    fn to_device(&mut self, device: &str) -> Result<()>;
}
