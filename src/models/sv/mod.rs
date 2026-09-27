//! Native v2Pro speaker verification features. No Python or ONNX runtime.

mod audio;
pub mod features;
mod network;

use crate::{Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::{path::Path, sync::Arc};

pub const EMBEDDING_DIM: usize = 20_480;

/// Validate all inference weight names/shapes without running the model.
pub fn validate_weights(weights: std::collections::HashMap<String, Tensor>) -> Result<()> {
    network::Network::load(VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu))?;
    Ok(())
}

/// Shared immutable ERes2NetV2 weights, evaluated in F32 on the selected device.
#[derive(Clone)]
pub struct SvModel {
    network: Arc<network::Network>,
    device: Device,
}

impl SvModel {
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let weights = crate::utils::load_safetensors(path)?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, device);
        Ok(Self {
            network: Arc::new(network::Network::load(vb)?),
            device: device.clone(),
        })
    }

    /// Extract from a WAV using the upstream reference path: mono -> SoVITS rate -> 16 kHz.
    pub fn extract(&self, path: impl AsRef<Path>, sovits_sample_rate: u32) -> Result<Tensor> {
        self.extract_from_samples(&audio::load_reference(path.as_ref(), sovits_sample_rate)?)
    }

    /// Extract from already prepared 16 kHz mono samples (no padding or peak boost).
    pub fn extract_from_samples(&self, samples: &[f32]) -> Result<Tensor> {
        self.forward_features(&features::fbank(samples, &self.device)?)
    }

    /// Compute forward3, not the 192-dimensional speaker-classification head.
    pub fn forward_features(&self, features: &Tensor) -> Result<Tensor> {
        match features.dims() {
            [batch, frames, 80] if *batch > 0 && *frames > 0 => {}
            shape => {
                return Err(Error::ModelLoadError(format!(
                    "SV fbank must be [batch, frames, 80], got {shape:?}"
                )))
            }
        }
        Ok(self
            .network
            .forward(&features.to_device(&self.device)?.to_dtype(DType::F32)?)?)
    }
}
