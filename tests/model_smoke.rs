//! Optional real-model smoke test.
//!
//! This test is skipped by default. Run it before releases with:
//!
//! GPT_SOVITS_RUN_MODEL_SMOKE=1 cargo test --test model_smoke -- --nocapture

use gpt_sovits_rs::audio_checks::{
    validate_audio_quality, AudioQualityMetrics, AudioQualityThresholds,
};
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::voice::{LoadedVoiceProfile, VoiceDefaults};
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::PathBuf;

fn optional_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

#[test]
fn debug_capture_preserves_seeded_speech_when_enabled() {
    if std::env::var("GPT_SOVITS_RUN_MODEL_SMOKE").as_deref() != Ok("1") {
        return;
    }
    let models = optional_path("GPT_SOVITS_MODELS_DIR").unwrap_or_else(|| "models".into());
    let voices = optional_path("GPT_SOVITS_VOICES_DIR").unwrap_or_else(|| "voices".into());
    let models = models.canonicalize().unwrap();
    let voices = voices.canonicalize().unwrap();
    let voice = std::env::var("GPT_SOVITS_SMOKE_VOICE").unwrap_or_else(|_| "demo".into());
    let device = std::env::var("GPT_SOVITS_SMOKE_DEVICE").unwrap_or_else(|_| "auto".into());
    let directory = tempfile::tempdir().unwrap();
    let run = |debug: bool| {
        let output = directory
            .path()
            .join(if debug { "trace.wav" } else { "normal.wav" });
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_gpt-sovits"));
        command
            .current_dir(directory.path())
            .env_remove("SOVITS_DEBUG")
            .args([
                "--mode",
                "cuda-graph",
                "--device",
                &device,
                "--voice",
                &voice,
            ])
            .arg("--models-dir")
            .arg(&models)
            .arg("--voices-dir")
            .arg(&voices)
            .args(["--seed", "42", "--speed", "1.25", "--max-tokens", "150"])
            .args([
                "--text",
                "这是固定种子的合成测试，声音和语速都应该保持一致。",
            ])
            .arg("--output")
            .arg(&output);
        for (variable, flag) in [
            ("GPT_SOVITS_GPT_MODEL", "--gpt-model"),
            ("GPT_SOVITS_SOVITS_MODEL", "--sovits-model"),
            ("GPT_SOVITS_BERT_MODEL", "--bert-model"),
            ("GPT_SOVITS_HUBERT_MODEL", "--hubert-model"),
        ] {
            if let Some(path) = optional_path(variable) {
                command.arg(flag).arg(path.canonicalize().unwrap());
            }
        }
        if debug {
            command.env("SOVITS_DEBUG", "1");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let audio = gpt_sovits_rs::AudioBuffer::load(&output).unwrap();
        let metrics = AudioQualityMetrics::from_audio(&audio);
        assert!(validate_audio_quality(&metrics, &AudioQualityThresholds::default()).is_empty());
        audio
    };
    let normal = run(false);
    let traced = run(true);
    assert_eq!(normal.sample_rate, traced.sample_rate);
    assert!(
        normal.samples == traced.samples,
        "debug capture changed seeded speech"
    );
    assert!(directory
        .path()
        .join("sovits_debug_dec_input.txt")
        .is_file());
    assert!(directory.path().join("sovits_debug_audio.txt").is_file());
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

    let voice = LoadedVoiceProfile::load(&voice_name, &voices_dir)
        .expect("voice profile should load when smoke test is enabled");
    let voice_models = voice.model_paths(&models_dir);
    let paths = ModelPaths::discover(
        &models_dir,
        ModelPathOverrides {
            gpt: optional_path("GPT_SOVITS_GPT_MODEL").or(voice_models.gpt),
            sovits: optional_path("GPT_SOVITS_SOVITS_MODEL").or(voice_models.sovits),
            bert: optional_path("GPT_SOVITS_BERT_MODEL"),
            hubert: optional_path("GPT_SOVITS_HUBERT_MODEL"),
        },
    )
    .expect("model discovery should succeed when smoke test is enabled");

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
    options.sv_embedding = voice.sv_embedding_path();
    options.max_tokens = std::env::var("GPT_SOVITS_SMOKE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(80);

    pipeline
        .preload_speaker_with_options(&reference_audio, reference_text, &options)
        .expect("speaker features should preload");
    let synthesize = |pipeline: &mut Pipeline, options: &InferenceOptions| {
        if defaults.split_sentences {
            pipeline.inference_split_with_method(
                &text,
                &reference_audio,
                reference_text,
                options,
                &defaults.mode,
                defaults.min_sentence_chars,
                defaults.sentence_gap_ms,
                defaults.sentence_fade_ms,
                defaults.split_method,
            )
        } else {
            pipeline.inference_with_mode(
                &defaults.mode,
                &text,
                &reference_audio,
                reference_text,
                options,
            )
        }
    };
    let audio =
        synthesize(&mut pipeline, &options).expect("real model smoke inference should succeed");

    let metrics = AudioQualityMetrics::from_audio(&audio);
    let issues = validate_audio_quality(&metrics, &AudioQualityThresholds::default());
    assert!(issues.is_empty(), "smoke audio quality issues: {issues:?}");

    if let Ok(seed) = std::env::var("GPT_SOVITS_SMOKE_SEED") {
        let seed: u64 = seed.parse().expect("SMOKE_SEED must be a u64");
        options.seed = Some(seed);
        let first = synthesize(&mut pipeline, &options).unwrap();
        options.seed = Some(seed.wrapping_add(1));
        let different = synthesize(&mut pipeline, &options).unwrap();
        assert_ne!(first.samples, different.samples);
        options.seed = None;
        let unseeded = synthesize(&mut pipeline, &options).unwrap();
        assert_ne!(first.samples, unseeded.samples);
        options.seed = Some(seed);
        let replay = synthesize(&mut pipeline, &options).unwrap();
        for audio in [&first, &different, &unseeded, &replay] {
            let metrics = AudioQualityMetrics::from_audio(audio);
            let issues = validate_audio_quality(&metrics, &AudioQualityThresholds::default());
            assert!(issues.is_empty(), "seeded smoke audio issues: {issues:?}");
        }
        assert_eq!(first.sample_rate, replay.sample_rate);
        assert_eq!(first.samples.len(), replay.samples.len());
        let max_error = first
            .samples
            .iter()
            .zip(&replay.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 1e-6, "seeded replay error: {max_error}");
        eprintln!(
            "Seed {seed} replayed after other seeded and unseeded requests; max error {max_error}"
        );
    }

    options.max_tokens = 4;
    let assert_feature_error = |pipeline: &mut Pipeline,
                                text: &str,
                                audio: &std::path::Path,
                                prompt: &str,
                                stage: &str| {
        let error = pipeline
            .inference_with_mode("kv", text, audio, prompt, &options)
            .expect_err("configured feature failures must stop inference");
        assert!(error.to_string().contains(stage), "{error}");
        eprintln!("Expected feature error: {error}");
    };
    if pipeline.bert_model().is_some() {
        let oversized = if language == Language::English {
            "hello ".repeat(600)
        } else {
            "你好".repeat(300)
        };
        assert_feature_error(
            &mut pipeline,
            &oversized,
            &reference_audio,
            reference_text,
            "Target BERT",
        );
        assert_feature_error(
            &mut pipeline,
            &text,
            &reference_audio,
            &oversized,
            "Reference BERT",
        );
    }
    if pipeline.hubert_model().is_some() {
        let truncated = tempfile::NamedTempFile::new().unwrap();
        let mut bytes = gpt_sovits_rs::AudioBuffer::new(vec![0.0; 16000], 16000, 1)
            .to_wav_bytes()
            .unwrap();
        bytes.pop();
        std::fs::write(truncated.path(), bytes).unwrap();
        assert_feature_error(
            &mut pipeline,
            &text,
            truncated.path(),
            reference_text,
            "Reference HuBERT",
        );

        let incompatible = tempfile::NamedTempFile::new().unwrap();
        let cpu = &candle_core::Device::Cpu;
        let dtype = candle_core::DType::F32;
        let weights = std::collections::HashMap::from([
            (
                "ssl_proj.weight",
                candle_core::Tensor::zeros((1, 1, 1), dtype, cpu).unwrap(),
            ),
            (
                "ssl_proj.bias",
                candle_core::Tensor::zeros(1, dtype, cpu).unwrap(),
            ),
            (
                "quantizer.vq.layers.0._codebook.embed",
                candle_core::Tensor::zeros((2, 1), dtype, cpu).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&weights, incompatible.path()).unwrap();
        pipeline
            .preload_speaker_with_options(&reference_audio, reference_text, &options)
            .expect("populate reference cache before replacing the tokenizer");
        pipeline
            .load_semantic_tokenizer(incompatible.path())
            .unwrap();
        for _ in 0..2 {
            assert_feature_error(
                &mut pipeline,
                &text,
                &reference_audio,
                reference_text,
                "Reference semantic token",
            );
        }
        pipeline.load_semantic_tokenizer(&paths.sovits).unwrap();
        let recovered = pipeline
            .inference_with_mode("kv", &text, &reference_audio, reference_text, &options)
            .expect("failed features must not poison the speaker cache");
        assert!(!recovered.is_empty());
    }

    if pipeline.bert_model().is_some() {
        // Chinese folds "..." into one symbol; English retains all three tokens.
        let boundary_reference = "a...".repeat(150);
        pipeline
            .preload_speaker_with_options(
                &reference_audio,
                &boundary_reference,
                &InferenceOptions {
                    language: Language::English,
                    prompt_language: Language::Chinese,
                    ..Default::default()
                },
            )
            .unwrap();
        let hot = pipeline
            .preload_speaker(&reference_audio, &boundary_reference, Language::English)
            .expect_err("a Chinese cache entry must not bypass English BERT length validation");
        pipeline.clear_speaker_cache();
        let cold = pipeline
            .preload_speaker(&reference_audio, &boundary_reference, Language::English)
            .expect_err("oversized English reference must fail on a cold cache too");
        assert!(hot.to_string().contains("Reference BERT"), "{hot}");
        assert_eq!(hot.to_string(), cold.to_string());
        pipeline
            .preload_speaker(&reference_audio, &boundary_reference, Language::Chinese)
            .unwrap();
    }
}
