//! Generate a small TTS matrix and report objective audio sanity metrics.

use clap::Parser;
use gpt_sovits_rs::audio_checks::{
    validate_audio_quality, AudioQualityMetrics, AudioQualityThresholds,
};
use gpt_sovits_rs::voice::{LoadedVoiceProfile, VoiceDefaults};
use gpt_sovits_rs::{Config, Language, Pipeline};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "quality_smoke")]
struct Args {
    #[arg(long)]
    voice: String,

    #[arg(long, default_value = "voices")]
    voices_dir: PathBuf,

    #[arg(long)]
    gpt_model: PathBuf,

    #[arg(long)]
    sovits_model: PathBuf,

    #[arg(long)]
    bert_model: Option<PathBuf>,

    #[arg(long)]
    hubert_model: Option<PathBuf>,

    #[arg(long, default_value = "cuda", value_parser = ["cuda", "cpu", "mps"])]
    device: String,

    #[arg(long)]
    half: bool,

    #[arg(long, default_value = "quality_outputs")]
    output_dir: PathBuf,

    #[arg(long)]
    text: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SmokeItem {
    index: usize,
    text: String,
    output: String,
    inference_ms: u128,
    duration_s: f32,
    peak: f32,
    rms: f32,
    clipping_ratio: f32,
    silence_ratio: f32,
    dc_offset: f32,
    has_non_finite: bool,
    issues: Vec<String>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output_dir)?;

    let voice = LoadedVoiceProfile::load(&args.voice, &args.voices_dir)?;
    let defaults = VoiceDefaults::from_profile(Some(&voice.profile));
    let language = Language::from_str(&defaults.language).unwrap_or(Language::Chinese);
    let reference_audio = voice
        .reference_audio_path()
        .ok_or("voice profile missing reference_audio")?;
    let reference_text = voice
        .reference_text()
        .ok_or("voice profile missing reference_text")?;

    let config = Config::builder()
        .with_device(&args.device)
        .with_half_precision(args.half)
        .build();
    let mut pipeline = Pipeline::new(config)?;
    pipeline.load_gpt(&args.gpt_model)?;
    pipeline.load_sovits(&args.sovits_model)?;
    if let Some(path) = args.bert_model.as_ref() {
        pipeline.load_bert(path)?;
    }
    if let Some(path) = args.hubert_model.as_ref() {
        pipeline.load_hubert(path)?;
        pipeline.load_semantic_tokenizer(&args.sovits_model)?;
    }

    let options = defaults.to_inference_options(language, Default::default());

    let texts = if args.text.is_empty() {
        vec![
            "你好，这是自动质量测试。".to_string(),
            "人民，只有人民，才是创造世界历史的动力。".to_string(),
            "请用稳定自然的节奏，说完这一小段话。".to_string(),
        ]
    } else {
        args.text
    };

    pipeline.preload_speaker(&reference_audio, reference_text, language)?;

    let thresholds = AudioQualityThresholds::default();
    let mut report = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        let start = Instant::now();
        let audio = match defaults.mode.as_str() {
            "plain" => pipeline.inference(text, &reference_audio, reference_text, &options)?,
            "kv" => {
                pipeline.inference_kv_cache(text, &reference_audio, reference_text, &options)?
            }
            "cuda-graph" => {
                pipeline.inference_cuda_graph(text, &reference_audio, reference_text, &options)?
            }
            _ => unreachable!("voice profile mode is validated on load"),
        };
        let inference_ms = start.elapsed().as_millis();
        let output_path = args.output_dir.join(format!("{:02}.wav", index));
        audio.save(&output_path)?;

        let metrics = AudioQualityMetrics::from_audio(&audio);
        let issues = validate_audio_quality(&metrics, &thresholds);
        report.push(SmokeItem {
            index,
            text: text.clone(),
            output: output_path.to_string_lossy().into_owned(),
            inference_ms,
            duration_s: metrics.duration_s,
            peak: metrics.peak,
            rms: metrics.rms,
            clipping_ratio: metrics.clipping_ratio,
            silence_ratio: metrics.silence_ratio,
            dc_offset: metrics.dc_offset,
            has_non_finite: metrics.has_non_finite,
            issues,
        });
    }

    let report_path = args.output_dir.join("report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("Wrote quality smoke report: {}", report_path.display());
    let issue_count: usize = report.iter().map(|item| item.issues.len()).sum();
    if issue_count > 0 {
        eprintln!("Quality smoke found {issue_count} issue(s). See report.json.");
        std::process::exit(2);
    }
    Ok(())
}
