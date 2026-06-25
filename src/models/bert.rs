//! BERT feature extractor — pure candle (chinese-roberta-wwm-ext-large).

use candle_core::{Device, Tensor};
use crate::Result;

fn device_str(dev: &Device) -> &'static str {
    match dev {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "mps",
    }
}

pub struct BertModel {
    model: super::bert_candle::BertCandleModel,
    device: &'static str,
    #[allow(dead_code)]
    max_length: usize,
}

impl BertModel {
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load from safetensors path. Tokenizer is looked up as `{dir}/tokenizer.json`
    /// or `models/bert/tokenizer.json` as fallback.
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, candle_core::DType::F32)
    }

    pub fn load_bf16(path: &str, device: &Device) -> Result<Self> {
        Self::load_with_dtype(path, device, candle_core::DType::BF16)
    }

    fn load_with_dtype(path: &str, device: &Device, dtype: candle_core::DType) -> Result<Self> {
        let weights_path = std::path::Path::new(path);
        let tokenizer_path = weights_path.with_file_name("tokenizer.json");
        let tokenizer_path = if tokenizer_path.exists() {
            tokenizer_path
        } else {
            std::path::PathBuf::from("models/bert/tokenizer.json")
        };
        let model = super::bert_candle::BertCandleModel::load_with_dtype(weights_path, &tokenizer_path, device, dtype)?;
        Ok(Self { model, device: device_str(device), max_length: 512 })
    }

    pub fn extract(&mut self, text: &str) -> Result<Tensor> {
        self.model.extract(text)
    }

    pub fn device(&self) -> &str { self.device }
}

impl crate::models::Model for BertModel {
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    fn device(&self) -> &str { self.device }

    fn to_device(&mut self, _device: &str) -> Result<()> {
        Ok(())
    }
}
