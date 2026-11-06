//! Local release benchmark. No HTTP listener or changes to an existing deployment.
mod report;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use gpt_sovits_rs::{
    audio_checks::{validate_audio_quality, AudioQualityMetrics, AudioQualityThresholds},
    model_paths::{ModelPathOverrides, ModelPaths},
    voice::{LoadedVoiceProfile, VoiceDefaults},
    Config, Language, Pipeline,
};
use report::{Attempt, Case, CaseReport, Report};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    voice: String,
    #[arg(long, default_value = "voices")]
    voices_dir: PathBuf,
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    /// Record the actual Cargo profile/features, e.g. "release; cuda".
    #[arg(long)]
    build_label: String,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Omit to honor the voice profile's mode. Does not change sampling parameters.
    #[arg(long, value_parser = ["auto", "kv", "plain", "cuda-graph"])]
    mode: Option<String>,
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=10))]
    warmup: u32,
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(2..=100))]
    repeats: u32,
}

fn hash_file(path: &Path) -> Result<Value> {
    let mut file = fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut size = 0u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
        size += count as u64;
    }
    Ok(json!({"sha256": format!("{:x}", digest.finalize()), "bytes": size}))
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn requires_sv(path: &Path) -> Result<bool> {
    // Inspect only the safetensors header; the pipeline validates and loads the weights.
    let mut file = fs::File::open(path)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    ensure!(length <= 16 * 1024 * 1024, "oversized safetensors header");
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header)?;
    let tensors: serde_json::Map<String, Value> = serde_json::from_slice(&header)?;
    Ok(tensors.contains_key("sv_emb.weight"))
}

fn validate_cases(cases: &[Case]) -> Result<()> {
    ensure!(!cases.is_empty(), "empty corpus");
    let mut ids = HashSet::new();
    for case in cases {
        ensure!(
            !case.id.is_empty()
                && case
                    .id
                    .bytes()
                    .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, b'_' | b'-')),
            "unsafe case ID"
        );
        ensure!(ids.insert(&case.id), "duplicate case ID");
        ensure!(
            !case.text.trim().is_empty() && !case.reference.trim().is_empty(),
            "empty text/reference"
        );
        ensure!(
            matches!(case.language.as_str(), "zh" | "en"),
            "ASR language must be zh or en"
        );
    }
    Ok(())
}

fn benchmark(args: &Args, report: &mut Report) -> Result<()> {
    ensure!(
        cfg!(feature = "cuda"),
        "GPU benchmark requires --features cuda; CPU fallback is not a benchmark pass"
    );
    ensure!(
        std::env::var("GPT_SOVITS_DISABLE_CUDA_GRAPH").as_deref() != Ok("1"),
        "unset GPT_SOVITS_DISABLE_CUDA_GRAPH; use --mode kv explicitly to benchmark KV"
    );
    for variable in [
        "GPT_SOVITS_SYNC_PROFILE",
        "GPT_SOVITS_CUDA_PROFILE",
        "SOVITS_DEBUG",
    ] {
        ensure!(
            std::env::var_os(variable).is_none(),
            "unset {variable} for latency measurements"
        );
    }
    candle_core::Device::new_cuda(0).context("CUDA device 0 must be available")?;
    let voice =
        LoadedVoiceProfile::load(&args.voice, &args.voices_dir).map_err(anyhow::Error::msg)?;
    let models = voice.model_paths(&args.models_dir);
    let paths = ModelPaths::discover(
        &args.models_dir,
        ModelPathOverrides {
            gpt: models.gpt,
            sovits: models.sovits,
            ..Default::default()
        },
    )
    .map_err(anyhow::Error::msg)?;
    let bert = paths
        .bert
        .as_ref()
        .context("BERT required; refusing reduced-quality benchmark")?;
    let hubert = paths
        .hubert
        .as_ref()
        .context("HuBERT required; refusing reduced-quality benchmark")?;
    let reference = voice
        .reference_audio_path()
        .context("reference_audio required")?;
    let reference_text = voice
        .reference_text()
        .filter(|text| !text.trim().is_empty())
        .context("reference_text required")?;
    let defaults = VoiceDefaults::from_profile(Some(&voice.profile));
    let language = Language::parse(&defaults.language).context("invalid voice language")?;
    let mut options = defaults.to_inference_options(language, Default::default());
    options.sv_embedding = voice.sv_embedding_path();
    ensure!(
        !requires_sv(&paths.sovits)? || options.sv_embedding.is_some(),
        "v2Pro benchmark requires the voice's matching SV embedding; refusing zero-SV fallback"
    );
    let mode = args.mode.as_deref().unwrap_or(&defaults.mode);
    for (label, path) in [
        ("gpt", &paths.gpt),
        ("sovits", &paths.sovits),
        ("bert", bert),
        ("hubert", hubert),
        ("reference_audio", &reference),
        ("voice_profile", &voice.dir.join("voice.json")),
        (
            "tokenizer",
            &bert
                .parent()
                .context("BERT directory")?
                .join("tokenizer.json"),
        ),
    ] {
        report.metadata["inputs"][label] = hash_file(path)?;
    }
    if let Some(sv) = &options.sv_embedding {
        report.metadata["inputs"]["sv_embedding"] = hash_file(sv)?;
    }
    report.metadata["settings"] = json!({"voice": args.voice, "mode": mode, "language": defaults.language,
        "dtype": "F32", "top_k": options.top_k, "top_p": options.top_p, "temperature": options.temperature,
        "repetition_penalty": options.repetition_penalty, "max_tokens": options.max_tokens, "speed": options.speed,
        "sv_embedding": options.sv_embedding.is_some(), "split_sentences": defaults.split_sentences,
        "split_method": format!("{:?}", defaults.split_method), "min_sentence_chars": defaults.min_sentence_chars,
        "sentence_gap_ms": defaults.sentence_gap_ms, "sentence_fade_ms": defaults.sentence_fade_ms});
    report.save(&args.output_dir)?;

    let start = Instant::now();
    let mut pipeline = Pipeline::new(Config::builder().with_device("cuda").build())?;
    // Auto can only choose Graph for a real CUDA F32 pipeline, never a CPU fallback.
    ensure!(
        pipeline.automatic_inference_mode() == "cuda-graph",
        "pipeline fell back to CPU"
    );
    pipeline.load_gpt(&paths.gpt)?;
    pipeline.load_sovits(&paths.sovits)?;
    pipeline.load_bert(bert)?;
    pipeline.load_hubert(hubert)?;
    report.model_load_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
    report.metadata["selected_mode"] = json!(if mode == "auto" {
        pipeline.automatic_inference_mode()
    } else {
        mode
    });
    let start = Instant::now();
    pipeline.preload_speaker_with_options(&reference, reference_text, &options)?;
    report.reference_preload_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
    let mut manifest = fs::File::create(args.output_dir.join("asr.jsonl"))?;
    for index in 0..report.cases.len() {
        let case = report.cases[index].input.clone();
        for (warmup, count) in [(true, args.warmup), (false, args.repeats)] {
            for repeat in 1..=count {
                let id = format!("{}-{}{repeat:02}", case.id, if warmup { "w" } else { "r" });
                println!("Synthesizing {id}");
                let start = Instant::now();
                let result = if defaults.split_sentences {
                    pipeline.inference_split_with_method(
                        &case.text,
                        &reference,
                        reference_text,
                        &options,
                        mode,
                        defaults.min_sentence_chars,
                        defaults.sentence_gap_ms,
                        defaults.sentence_fade_ms,
                        defaults.split_method,
                    )
                } else {
                    pipeline.inference_with_mode(
                        mode,
                        &case.text,
                        &reference,
                        reference_text,
                        &options,
                    )
                };
                // AudioBuffer owns host samples: synthesis has completed before stopping the clock.
                let mut attempt = Attempt {
                    elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                    audio_seconds: None,
                    output: None,
                    metrics: None,
                    issues: vec![],
                };
                match result {
                    Ok(audio) => {
                        let metrics = AudioQualityMetrics::from_audio(&audio);
                        attempt.audio_seconds = Some(f64::from(metrics.duration_s));
                        attempt.issues =
                            validate_audio_quality(&metrics, &AudioQualityThresholds::default());
                        attempt.metrics = Some(json!({"rms": metrics.rms, "peak": metrics.peak,
                            "clipping_ratio": metrics.clipping_ratio, "silence_ratio": metrics.silence_ratio,
                            "dc_offset": metrics.dc_offset, "has_non_finite": metrics.has_non_finite,
                            "sample_rate": audio.sample_rate, "channels": audio.channels}));
                        let filename = format!("{id}.wav");
                        match audio.save(args.output_dir.join(&filename)) {
                            Ok(()) => {
                                if !warmup {
                                    writeln!(
                                        manifest,
                                        "{}",
                                        json!({"id": id, "audio": filename,
                                    "reference": case.reference, "language": case.language, "force_language": case.language})
                                    )?;
                                }
                                attempt.output = Some(filename);
                            }
                            Err(error) => attempt.issues.push(format!("saving audio: {error}")),
                        }
                    }
                    Err(error) => attempt.issues.push(error.to_string()),
                }
                if warmup {
                    report.cases[index].warmups.push(attempt);
                } else {
                    report.cases[index].runs.push(attempt);
                }
                report.save(&args.output_dir)?;
            }
        }
    }
    report.completed = true;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.build_label.trim().is_empty(),
        "build label is required"
    );
    let corpus = match &args.corpus {
        Some(path) => fs::read(path)?,
        None => include_bytes!("../speech_regression/corpus.json").to_vec(),
    };
    let cases: Vec<Case> = serde_json::from_slice(&corpus)?;
    validate_cases(&cases)?;
    fs::create_dir(&args.output_dir)
        .context("output directory must be new; its parent must exist")?;
    let log = fs::File::create(args.output_dir.join("inference.log"))?;
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer(log)
        .init();
    let mut report = Report {
        schema_version: 1,
        warmup_count: args.warmup,
        repeat_count: args.repeats,
        completed: false,
        fatal_error: None,
        model_load_ms: None,
        reference_preload_ms: None,
        cases: cases
            .into_iter()
            .map(|input| CaseReport {
                input,
                warmups: vec![],
                runs: vec![],
            })
            .collect(),
        metadata: json!({"started_at_unix": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            "package_version": env!("CARGO_PKG_VERSION"), "build_label": args.build_label,
            "features": {"cuda": cfg!(feature = "cuda"), "mkl": cfg!(feature = "mkl"), "http_api": cfg!(feature = "http-api")},
            "source_commit": command("git", &["rev-parse", "HEAD"]),
            "source_status": command("git", &["status", "--short"]),
            "host_rustc": command("rustc", &["-Vv"]),
            "host": command("uname", &["-srm"]),
            "gpus": command("nvidia-smi", &["--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]),
            "gpu_state_before": command("nvidia-smi", &["--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,pstate", "--format=csv,noheader"]),
            "nvcc": command("nvcc", &["--version"]),
            "cuda_visible_devices": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            "binary": hash_file(&std::env::current_exe()?)?,
            "cargo_lock": hash_file(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
            "corpus_sha256": format!("{:x}", Sha256::digest(&corpus)), "inputs": {}}),
    };
    if let Err(error) = benchmark(&args, &mut report) {
        report.fatal_error = Some(format!("{error:#}"));
    }
    report.metadata["gpu_state_after"] = json!(command(
        "nvidia-smi",
        &[
            "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,pstate",
            "--format=csv,noheader"
        ]
    ));
    report.save(&args.output_dir)?;
    println!("Reports: {}", args.output_dir.display());
    ensure!(
        report.passed(),
        "benchmark incomplete or signal checks failed; inspect report.json and inference.log"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corpus_is_safe_and_failures_are_rejected() {
        let mut cases: Vec<Case> =
            serde_json::from_slice(include_bytes!("../speech_regression/corpus.json")).unwrap();
        validate_cases(&cases).unwrap();
        cases[0].id = "../escape".into();
        assert!(validate_cases(&cases).is_err());
        assert!(validate_cases(&[]).is_err());
    }
    #[test]
    fn requires_real_repeats_and_preserves_existing_outputs() {
        assert!(Args::try_parse_from([
            "bench",
            "--voice",
            "test",
            "--build-label",
            "test",
            "--output-dir",
            "/tmp/test",
            "--repeats",
            "0"
        ])
        .is_err());
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("sentinel"), "keep").unwrap();
        assert!(fs::create_dir(directory.path()).is_err());
        assert_eq!(
            fs::read_to_string(directory.path().join("sentinel")).unwrap(),
            "keep"
        );
    }
    #[test]
    fn fingerprints_use_sha256() {
        let file = tempfile::NamedTempFile::new().unwrap();
        fs::write(file.path(), b"abc").unwrap();
        assert_eq!(
            hash_file(file.path()).unwrap()["sha256"],
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn identifies_sv_conditioned_checkpoints_and_rejects_bad_headers() {
        let file = tempfile::NamedTempFile::new().unwrap();
        for (header, expected) in [(r#"{"sv_emb.weight":{}}"#, true), ("{}", false)] {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            fs::write(file.path(), bytes).unwrap();
            assert_eq!(requires_sv(file.path()).unwrap(), expected);
        }
        fs::write(file.path(), u64::MAX.to_le_bytes()).unwrap();
        assert!(requires_sv(file.path()).is_err());
        fs::write(file.path(), b"trunc").unwrap();
        assert!(requires_sv(file.path()).is_err());
    }
}
