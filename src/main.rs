//! GPT-SoVITS CLI - Command line interface for TTS inference

use clap::Parser;
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::PathBuf;
use tracing::{info, error};

#[derive(Parser, Debug)]
#[command(name = "gpt-sovits")]
#[command(author = "GPT-SoVITS Rust Contributors")]
#[command(version = "0.1.0")]
#[command(about = "GPT-SoVITS TTS Inference Engine", long_about = None)]
struct Args {
    /// Input text for synthesis
    #[arg(short, long)]
    text: String,

    /// Path to GPT model file
    #[arg(long, required_unless_present = "http")]
    gpt_model: Option<PathBuf>,

    /// Path to SoVITS model file
    #[arg(long, required_unless_present = "http")]
    sovits_model: Option<PathBuf>,

    /// Reference audio path
    #[arg(long)]
    reference_audio: Option<PathBuf>,

    /// Reference audio text
    #[arg(long)]
    reference_text: Option<String>,

    /// Language of reference audio
    #[arg(long, default_value = "zh")]
    language: String,

    /// Output WAV file path
    #[arg(short, long)]
    output: PathBuf,

    /// Top-k sampling
    #[arg(long, default_value = "15")]
    top_k: usize,

    /// Top-p sampling
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Sampling temperature
    #[arg(long, default_value = "0.8")]
    temperature: f32,

    /// Speed multiplier
    #[arg(long, default_value = "1.0")]
    speed: f32,

    /// Enable half-precision (FP16)
    #[arg(long)]
    half: bool,

    /// Device to use
    #[arg(long, default_value = "cuda", value_parser = ["cuda", "cpu", "mps"])]
    device: String,

    /// Start HTTP server mode
    #[arg(long)]
    http: bool,

    /// HTTP server port
    #[arg(long, default_value = "9880")]
    port: u16,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    std::env::set_var("RUST_LOG", log_level);
    tracing_subscriber::fmt::init();

    info!("Starting GPT-SoVITS TTS Engine");

    // HTTP mode
    if args.http {
        #[cfg(feature = "http-api")]
        {
            run_http_server(args.port);
        }
        #[cfg(not(feature = "http-api"))]
        {
            error!("HTTP API feature is not enabled. Build with --features http-api");
        }
        return;
    }

    // Validate required arguments for CLI mode
    let gpt_model = args.gpt_model.expect("GPT model path required");
    let sovits_model = args.sovits_model.expect("SoVITS model path required");
    let reference_audio = args.reference_audio.expect("Reference audio required");
    let reference_text = args.reference_text.expect("Reference text required");

    info!("Loading models...");
    info!("  GPT model: {:?}", gpt_model);
    info!("  SoVITS model: {:?}", sovits_model);

    // Initialize configuration
    let config = Config::builder()
        .with_half_precision(args.half)
        .with_device(&args.device)
        .build();

    // Create pipeline
    let mut pipeline = match Pipeline::new(config) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to initialize pipeline: {}", e);
            std::process::exit(1);
        }
    };

    // Load models
    info!("Loading GPT model...");
    if let Err(e) = pipeline.load_gpt(&gpt_model) {
        error!("Failed to load GPT model: {}", e);
        std::process::exit(1);
    }

    info!("Loading SoVITS model...");
    if let Err(e) = pipeline.load_sovits(&sovits_model) {
        error!("Failed to load SoVITS model: {}", e);
        std::process::exit(1);
    }

    // Parse language
    let language = Language::from_str(&args.language).unwrap_or(Language::Chinese);

    // Create inference options
    let options = InferenceOptions::builder()
        .top_k(args.top_k)
        .top_p(args.top_p)
        .temperature(args.temperature)
        .speed(args.speed)
        .language(language)
        .build();

    // Run inference
    info!("Running inference...");
    info!("  Text: {}", args.text);
    info!("  Reference: {:?}", reference_audio);
    info!("  Language: {:?}", language);

    match pipeline.inference(
        &args.text,
        &reference_audio,
        &reference_text,
        &options,
    ) {
        Ok(audio) => {
            info!("Saving output to {:?}", args.output);
            if let Err(e) = audio.save(&args.output) {
                error!("Failed to save audio: {}", e);
                std::process::exit(1);
            }
            info!("Done! Output saved to {:?}", args.output);
        }
        Err(e) => {
            error!("Inference failed: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "http-api")]
fn run_http_server(port: u16) {
    use axum::{
        extract::State,
        http::StatusCode,
        response::Response,
        routing::post,
        Json, Router,
    };
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tower_http::trace::TraceLayer;

    #[derive(Clone)]
    struct AppState {
        pipeline: Arc<Mutex<Pipeline>>,
    }

    #[derive(Deserialize)]
    struct TtsRequest {
        text: String,
        text_language: String,
        refer_wav_path: Option<String>,
        prompt_text: Option<String>,
        prompt_language: Option<String>,
    }

    info!("Starting HTTP server on port {}", port);

    // HTTP server implementation would go here
    // This is a placeholder for the actual implementation
    println!("HTTP server started at http://localhost:{}", port);
    println!("Endpoints:");
    println!("  POST /tts - TTS inference");
    println!("  POST /change_refer - Change reference audio");
    println!("  POST /control - Server control");
}
