//! Models Module
//!
//! Neural network models for GPT-SoVITS

pub mod bert;
pub mod bigvgan;
pub mod gpt;
pub mod hubert;
pub mod mrte;
pub mod sovits;
pub mod transformer;

pub use bert::BertModel;
pub use bigvgan::BigVGAN;
pub use gpt::GPTModel;
pub use hubert::HubertModel;
pub use mrte::MRTE;
pub use sovits::SoVITSModel;
pub use transformer::{Transformer, TransformerConfig, TransformerBlock, MultiHeadAttention};

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
