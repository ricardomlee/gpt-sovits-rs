pub mod checkpoint;

use candle_core::{safetensors, Device, Tensor};
use gpt_sovits_rs::{
    models::{BertCandleModel, GPTModel, SoVITSModel},
    text_frontend::TextFrontend,
    AudioBuffer, Error, Language, Result,
};
use std::collections::HashMap;

fn get<'a>(inputs: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    inputs
        .get(name)
        .ok_or_else(|| Error::InferenceError(format!("Missing input tensor: {name}")))
}

fn ids(input: &Tensor) -> Result<Vec<usize>> {
    input
        .to_vec1::<i64>()?
        .into_iter()
        .map(|id| {
            usize::try_from(id)
                .map_err(|_| Error::InferenceError(format!("Invalid token ID: {id}")))
        })
        .collect()
}

fn reference_stft(inputs: &HashMap<String, Tensor>) -> Result<&Tensor> {
    let reference = get(inputs, "reference_stft")?;
    if !matches!(reference.dims(), [1, 704, frames] if *frames > 0) {
        return Err(Error::InferenceError(
            "Expected reference STFT [1, 704, frames]".into(),
        ));
    }
    Ok(reference)
}

pub fn render(model: &SoVITSModel, case: &[u8]) -> Result<AudioBuffer> {
    let inputs = safetensors::load_buffer(case, &Device::Cpu)?;
    let samples = model.synthesize_with_speed_and_sv(
        &ids(get(&inputs, "semantic")?)?,
        &ids(get(&inputs, "phonemes")?)?,
        Some(reference_stft(&inputs)?),
        inputs.get("sv_embedding"),
        0.0,
        1.0,
    )?;
    Ok(AudioBuffer::new(samples, model.sampling_rate(), 1))
}

pub struct Voice {
    gpt: GPTModel,
    sovits: SoVITSModel,
    bert: BertCandleModel,
    frontend: TextFrontend,
    prompt_tokens: Vec<usize>,
    prompt_phones: Vec<usize>,
    prompt_bert: Tensor,
    stft: Tensor,
    sv: Option<Tensor>,
}

impl Voice {
    pub fn new(
        gpt: GPTModel,
        sovits: SoVITSModel,
        bert: BertCandleModel,
        reference: &[u8],
    ) -> Result<Self> {
        let inputs = safetensors::load_buffer(reference, &Device::Cpu)?;
        let prompt_tokens = ids(get(&inputs, "prompt_tokens")?)?;
        let prompt_phones = ids(get(&inputs, "prompt_phonemes")?)?;
        let prompt_bert = get(&inputs, "prompt_bert")?;
        if prompt_tokens.is_empty()
            || prompt_phones.is_empty()
            || prompt_bert.dims() != [1, prompt_phones.len(), 512]
        {
            return Err(Error::InferenceError(
                "Expected a reference with semantic tokens, phones and aligned BERT".into(),
            ));
        }
        Ok(Self {
            gpt,
            sovits,
            bert,
            frontend: TextFrontend::new()?,
            prompt_tokens,
            prompt_phones,
            prompt_bert: prompt_bert.clone(),
            stft: reference_stft(&inputs)?.clone(),
            sv: inputs.get("sv_embedding").cloned(),
        })
    }

    pub fn synthesize(&self, text: &str) -> Result<(Vec<usize>, AudioBuffer)> {
        let text = text.trim();
        if text.is_empty() || text.chars().count() > 80 {
            return Err(Error::InferenceError("Enter 1 to 80 characters".into()));
        }
        let (phones, word2ph, normalized) = self
            .frontend
            .process_with_word2ph_and_text(text, Language::Chinese)?;
        if phones.is_empty() {
            return Err(Error::InferenceError("Text produced no phones".into()));
        }
        let bert = self.gpt.project_and_align_bert(
            &self.bert.extract(&normalized)?,
            &word2ph,
            phones.len(),
        )?;
        let aligned = Tensor::cat(&[&self.prompt_bert, &bert], 1)?;
        let all_phones: Vec<_> = self.prompt_phones.iter().chain(&phones).copied().collect();

        const SEED: u64 = 42;
        const MAX_TOKENS: usize = 256;
        let capacity = all_phones.len() + self.prompt_tokens.len() + MAX_TOKENS + 32;
        self.gpt.set_sampling_seed(Some(SEED))?;
        let tokens = self.gpt.generate_with_static_kv(
            &all_phones,
            &self.prompt_tokens,
            Some(&aligned),
            15,
            0.95,
            0.8,
            1.35,
            MAX_TOKENS,
            capacity,
        )?;
        if tokens.len() >= MAX_TOKENS {
            return Err(Error::InferenceError(
                "Generation reached the token limit; use a shorter sentence".into(),
            ));
        }
        self.sovits
            .set_sampling_seed(Some(SEED ^ 0x9e37_79b9_7f4a_7c15))?;
        let samples = self.sovits.synthesize_with_speed_and_sv(
            &tokens,
            &phones,
            Some(&self.stft),
            self.sv.as_ref(),
            0.5,
            1.0,
        )?;
        Ok((
            tokens,
            AudioBuffer::new(samples, self.sovits.sampling_rate(), 1),
        ))
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    use crate::checkpoint::{header_end, Checkpoint};
    use candle_core::DType;
    use gpt_sovits_rs::utils::StateDict;
    use wasm_bindgen::prelude::*;

    type JsResult<T> = core::result::Result<T, JsError>;

    #[wasm_bindgen]
    pub fn checkpoint_header_end(prefix: &[u8], size: u32) -> JsResult<u32> {
        Ok(header_end(prefix, size)?)
    }

    #[wasm_bindgen]
    pub struct CheckpointLoader(Checkpoint);

    #[wasm_bindgen]
    impl CheckpointLoader {
        #[wasm_bindgen(constructor)]
        pub fn new(header: &[u8], size: u32) -> JsResult<CheckpointLoader> {
            Ok(Self(Checkpoint::new(header, size)?))
        }

        pub fn next_range(&self) -> Option<Vec<u32>> {
            self.0.next_range().map(|(start, end)| vec![start, end])
        }

        pub fn push(&mut self, bytes: &[u8]) -> JsResult<()> {
            Ok(self.0.push(bytes)?)
        }
    }

    #[wasm_bindgen]
    pub struct AcousticModel(SoVITSModel);

    #[wasm_bindgen]
    impl AcousticModel {
        #[wasm_bindgen(constructor)]
        pub fn new(weights: &[u8]) -> JsResult<AcousticModel> {
            Ok(Self(SoVITSModel::from_safetensors(weights)?))
        }

        pub fn from_checkpoint(loader: CheckpointLoader) -> JsResult<AcousticModel> {
            let weights = StateDict::new(loader.0.finish()?);
            Ok(Self(SoVITSModel::from_state_dict(
                &weights,
                &Device::Cpu,
                DType::F32,
            )?))
        }

        pub fn render(&self, case: &[u8]) -> JsResult<Vec<u8>> {
            Ok(super::render(&self.0, case)?.to_wav_bytes()?)
        }
    }

    #[wasm_bindgen]
    pub struct TextVoice(Voice);

    #[wasm_bindgen]
    impl TextVoice {
        #[wasm_bindgen(constructor)]
        pub fn new(
            sovits: AcousticModel,
            gpt: CheckpointLoader,
            bert: CheckpointLoader,
            tokenizer: &[u8],
            reference: &[u8],
        ) -> JsResult<TextVoice> {
            let gpt = GPTModel::from_state_dict(
                &StateDict::new(gpt.0.finish()?),
                &Device::Cpu,
                DType::F32,
            )?;
            let bert = BertCandleModel::from_cpu_weights(bert.0.finish()?, tokenizer)?;
            Ok(Self(Voice::new(gpt, sovits.0, bert, reference)?))
        }

        pub fn render(&self, text: &str) -> JsResult<Vec<u8>> {
            Ok(self.0.synthesize(text)?.1.to_wav_bytes()?)
        }
    }
}
