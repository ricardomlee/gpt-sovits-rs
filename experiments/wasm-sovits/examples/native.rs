use anyhow::{bail, Context, Result};
use candle_core::{safetensors, Device, Tensor};
use gpt_sovits_rs::{
    model_paths::{ModelPathOverrides, ModelPaths},
    models::SoVITSModel,
    utils::SpectrogramExtractor,
    voice::{LoadedVoiceProfile, VoiceDefaults},
    AudioBuffer, Config, Language, Pipeline,
};
use std::{collections::HashMap, fs, path::Path, time::Instant};

fn trace_tokens(dir: &Path, name: &str) -> Result<Tensor> {
    let file = fs::read_to_string(dir.join(format!("sovits_debug_{name}.txt")))?;
    let mut lines = file.lines();
    let count: usize = lines.next().context("Missing trace count")?.parse()?;
    let ids = lines.map(str::parse).collect::<Result<Vec<i64>, _>>()?;
    if ids.len() != count || count == 0 {
        bail!(
            "Invalid {name} trace length: {} (expected {count})",
            ids.len()
        );
    }
    Ok(Tensor::from_vec(ids, count, &Device::Cpu)?)
}

fn prepare(trace: &str, reference: &str, sv: &str, output: &str) -> Result<()> {
    let audio = AudioBuffer::load(reference)?;
    if audio.sample_rate != 32_000 || audio.channels != 1 {
        bail!("Use a 32 kHz mono reference WAV");
    }
    let stft = SpectrogramExtractor::new(32_000, 2048, 640, 100)
        .extract_spectrogram_batched(&audio.samples, &Device::Cpu)?
        .narrow(1, 0, 704)?;
    let trace = Path::new(trace);
    let mut tensors = HashMap::from([
        (
            "semantic".to_string(),
            trace_tokens(trace, "semantic_tokens")?,
        ),
        ("phonemes".to_string(), trace_tokens(trace, "text_tokens")?),
        ("reference_stft".to_string(), stft),
    ]);
    if sv != "-" {
        let embedding = safetensors::load(sv, &Device::Cpu)?
            .remove("sv_embedding")
            .context("Missing sv_embedding tensor")?;
        tensors.insert("sv_embedding".to_string(), embedding);
    }
    safetensors::save(&tensors, output)?;
    Ok(())
}

fn prepare_reference(models: &Path, voices: &Path, name: &str, output: &Path) -> Result<()> {
    let voice = LoadedVoiceProfile::load(name, voices).map_err(anyhow::Error::msg)?;
    let weights = voice.model_paths(models);
    let paths = ModelPaths::discover(
        models,
        ModelPathOverrides {
            gpt: weights.gpt,
            sovits: weights.sovits,
            ..Default::default()
        },
    )
    .map_err(anyhow::Error::msg)?;
    let defaults = VoiceDefaults::from_profile(Some(&voice.profile));
    let mut options = defaults.to_inference_options(Language::Chinese, Default::default());
    options.sv_embedding = voice.sv_embedding_path();

    let mut pipeline = Pipeline::new(Config::builder().with_device("cpu").build())?;
    pipeline.load_gpt(paths.gpt)?;
    pipeline.load_sovits(paths.sovits)?;
    pipeline.load_bert(paths.bert.context("BERT weights required")?)?;
    pipeline.load_hubert(paths.hubert.context("HuBERT weights required")?)?;
    pipeline.export_reference_features(
        &voice
            .reference_audio_path()
            .context("Voice reference audio required")?,
        voice
            .reference_text()
            .filter(|text| !text.trim().is_empty())
            .context("Voice reference transcript required")?,
        &options,
        output,
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, models, voices, name, output] if mode == "prepare-reference" => {
            prepare_reference(Path::new(models), Path::new(voices), name, Path::new(output))
        }
        [mode, trace, reference, sv, output] if mode == "prepare" => {
            prepare(trace, reference, sv, output)
        }
        [mode, weights, case, output] if mode == "render" => {
            let start = Instant::now();
            let model = SoVITSModel::from_safetensors(&fs::read(weights)?)?;
            let load_ms = start.elapsed().as_millis();
            let case = fs::read(case)?;
            let start = Instant::now();
            let audio = wasm_sovits::render(&model, &case)?;
            let wav = audio.to_wav_bytes()?;
            let render_ms = start.elapsed().as_millis();
            fs::write(output, wav)?;
            println!("load_ms={load_ms} render_ms={render_ms} samples={}", audio.len());
            Ok(())
        }
        _ => bail!("native prepare-reference MODELS_DIR VOICES_DIR VOICE OUTPUT.safetensors\nnative prepare TRACE_DIR REF_32K_MONO.wav SV.safetensors|- CASE.safetensors\nnative render WEIGHTS.safetensors CASE.safetensors OUTPUT.wav"),
    }
}
