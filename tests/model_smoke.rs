//! Optional real-model smoke test.
//!
//! This test is skipped by default. Run it before releases with:
//!
//! GPT_SOVITS_RUN_MODEL_SMOKE=1 cargo test --test model_smoke -- --nocapture

use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::voice::{LoadedVoiceProfile, VoiceDefaults};
use gpt_sovits_rs::{Config, Language, Pipeline};
use std::path::PathBuf;

fn optional_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[test]
fn real_model_inference_smoke_when_enabled() {
    if std::env::var("GPT_SOVITS_RUN_MODEL_SMOKE").as_deref() != Ok("1") {
        eprintln!("skipping real model smoke; set GPT_SOVITS_RUN_MODEL_SMOKE=1 to enable");
        return;
    }

    let models_dir = optional_path("GPT_SOVITS_MODELS_DIR").unwrap_or_else(|| "models".into());
    let voices_dir = optional_path("GPT_SOVITS_VOICES_DIR").unwrap_or_else(|| "voices".into());
    let voice_name = std::env::var("GPT_SOVITS_SMOKE_VOICE").unwrap_or_else(|_| "demo".into());
    let device = std::env::var("GPT_SOVITS_SMOKE_DEVICE").unwrap_or_else(|_| "auto".into());
    let text = std::env::var("GPT_SOVITS_SMOKE_TEXT")
        .unwrap_or_else(|_| "你好，这是发布前的真实模型冒烟测试。".into());

    let paths = ModelPaths::discover(
        &models_dir,
        ModelPathOverrides {
            gpt: optional_path("GPT_SOVITS_GPT_MODEL"),
            sovits: optional_path("GPT_SOVITS_SOVITS_MODEL"),
            bert: optional_path("GPT_SOVITS_BERT_MODEL"),
            hubert: optional_path("GPT_SOVITS_HUBERT_MODEL"),
        },
    )
    .expect("model discovery should succeed when smoke test is enabled");

    let voice = LoadedVoiceProfile::load(&voice_name, &voices_dir)
        .expect("voice profile should load when smoke test is enabled");
    let defaults = VoiceDefaults::from_profile(Some(&voice.profile));
    let language = Language::parse(&defaults.language).unwrap_or(Language::Chinese);
    let reference_audio = voice
        .reference_audio_path()
        .expect("voice profile should provide reference_audio");
    let reference_text = voice
        .reference_text()
        .expect("voice profile should provide reference_text");

    let config = Config::builder().with_device(&device).build();
    let mut pipeline = Pipeline::new(config).expect("pipeline should initialize");
    pipeline
        .load_gpt(&paths.gpt)
        .expect("GPT model should load");
    pipeline
        .load_sovits(&paths.sovits)
        .expect("SoVITS model should load");
    if let Some(path) = paths.bert.as_ref() {
        pipeline.load_bert(path).expect("BERT model should load");
    }
    if let Some(path) = paths.hubert.as_ref() {
        pipeline
            .load_hubert(path)
            .expect("HuBERT model should load");
    }

    let mut options = defaults.to_inference_options(language, Default::default());
    options.max_tokens = std::env::var("GPT_SOVITS_SMOKE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(80);

    let audio = pipeline
        .inference_with_mode(
            &defaults.mode,
            &text,
            &reference_audio,
            reference_text,
            &options,
        )
        .expect("real model smoke inference should succeed");

    assert!(
        audio.duration() > 0.1,
        "smoke output should contain audible duration"
    );
    assert!(
        audio.samples.iter().all(|sample| sample.is_finite()),
        "smoke output should not contain non-finite samples"
    );
}
