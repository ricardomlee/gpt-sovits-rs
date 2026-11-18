use candle_core::{safetensors, Device, Tensor};
use gpt_sovits_rs::{models::SoVITSModel, AudioBuffer, Error, Result};

pub fn render(model: &SoVITSModel, case: &[u8]) -> Result<AudioBuffer> {
    let inputs = safetensors::load_buffer(case, &Device::Cpu)?;
    let get = |name: &str| {
        inputs
            .get(name)
            .ok_or_else(|| Error::InferenceError(format!("Missing case tensor: {name}")))
    };
    let ids = |tensor: &Tensor| -> Result<Vec<usize>> {
        tensor
            .to_vec1::<i64>()?
            .into_iter()
            .map(|id| {
                usize::try_from(id)
                    .map_err(|_| Error::InferenceError(format!("Invalid token ID: {id}")))
            })
            .collect()
    };
    let reference = get("reference_stft")?;
    if !matches!(reference.dims(), [1, 704, frames] if *frames > 0) {
        return Err(Error::InferenceError(
            "Expected reference STFT [1, 704, frames]".into(),
        ));
    }
    let samples = model.synthesize_with_speed_and_sv(
        &ids(get("semantic")?)?,
        &ids(get("phonemes")?)?,
        Some(reference),
        inputs.get("sv_embedding"),
        0.0,
        1.0,
    )?;
    Ok(AudioBuffer::new(samples, model.sampling_rate(), 1))
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct AcousticModel(SoVITSModel);

    #[wasm_bindgen]
    impl AcousticModel {
        #[wasm_bindgen(constructor)]
        pub fn new(weights: &[u8]) -> core::result::Result<AcousticModel, JsError> {
            Ok(Self(SoVITSModel::from_safetensors(weights)?))
        }

        pub fn render(&self, case: &[u8]) -> core::result::Result<Vec<u8>, JsError> {
            Ok(super::render(&self.0, case)?.to_wav_bytes()?)
        }
    }
}
