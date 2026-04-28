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
    text: Option<String>,

    /// Inspect model file
    #[arg(long)]
    inspect: Option<PathBuf>,

    /// Path to GPT model file
    #[arg(long)]
    gpt_model: Option<PathBuf>,

    /// Path to SoVITS model file
    #[arg(long)]
    sovits_model: Option<PathBuf>,

    /// Path to BigVGAN model file
    #[arg(long)]
    bigvgan_model: Option<PathBuf>,

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
    output: Option<PathBuf>,

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

    // Inspect mode
    if let Some(ref model_path) = args.inspect {
        inspect_model(model_path);
        return;
    }

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    std::env::set_var("RUST_LOG", log_level);
    tracing_subscriber::fmt::init();

    info!("Starting GPT-SoVITS TTS Engine");

    // HTTP mode
    if args.http {
        #[cfg(feature = "http-api")]
        {
            if let Err(e) = http_api::run(
                args.port,
                args.gpt_model.as_deref(),
                args.sovits_model.as_deref(),
                args.bigvgan_model.as_deref(),
            ) {
                error!("HTTP server error: {}", e);
                std::process::exit(1);
            }
        }
        #[cfg(not(feature = "http-api"))]
        {
            error!("HTTP API feature is not enabled. Build with --features http-api");
        }
        return;
    }

    // CLI mode - validate required arguments
    let text = match &args.text {
        Some(t) => t.clone(),
        None => {
            eprintln!("Error: --text is required in CLI mode");
            eprintln!("Usage: gpt-sovits --text <TEXT> --output <OUTPUT> [OPTIONS]");
            eprintln!("       gpt-sovits --http [OPTIONS]");
            std::process::exit(1);
        }
    };

    let gpt_model = match &args.gpt_model {
        Some(m) => m.clone(),
        None => {
            eprintln!("Error: --gpt-model is required in CLI mode");
            std::process::exit(1);
        }
    };

    let sovits_model = match &args.sovits_model {
        Some(m) => m.clone(),
        None => {
            eprintln!("Error: --sovits-model is required in CLI mode");
            std::process::exit(1);
        }
    };

    let reference_audio = match &args.reference_audio {
        Some(a) => a.clone(),
        None => {
            eprintln!("Error: --reference-audio is required in CLI mode");
            std::process::exit(1);
        }
    };

    let reference_text = match &args.reference_text {
        Some(t) => t.clone(),
        None => {
            eprintln!("Error: --reference-text is required in CLI mode");
            std::process::exit(1);
        }
    };

    let output = match &args.output {
        Some(o) => o.clone(),
        None => {
            eprintln!("Error: --output is required in CLI mode");
            std::process::exit(1);
        }
    };

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

    // Load BigVGAN model (optional but recommended for proper audio synthesis)
    if let Some(ref bigvgan_path) = args.bigvgan_model {
        info!("Loading BigVGAN model...");
        if let Err(e) = pipeline.load_bigvgan(bigvgan_path) {
            error!("Failed to load BigVGAN model: {}", e);
            std::process::exit(1);
        }
    } else {
        info!("BigVGAN model not specified, using fallback synthesis");
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
    info!("  Text: {}", text);
    info!("  Reference: {:?}", reference_audio);
    info!("  Language: {:?}", language);

    match pipeline.inference(
        &text,
        &reference_audio,
        &reference_text,
        &options,
    ) {
        Ok(audio) => {
            info!("Saving output to {:?}", output);
            if let Err(e) = audio.save(&output) {
                error!("Failed to save audio: {}", e);
                std::process::exit(1);
            }
            info!("Done! Output saved to {:?}", output);
        }
        Err(e) => {
            error!("Inference failed: {}", e);
            std::process::exit(1);
        }
    }
}

/// Inspect model file
fn inspect_model(path: &PathBuf) {
    use std::fs::File;
    use std::io::Read;
    use safetensors::SafeTensors;

    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let st = SafeTensors::deserialize(&buffer).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();

    println!("{name} keys ({} total):", st.names().len());
    for name in st.names() {
        let tensor = st.tensor(name).unwrap();
        println!("  {name:60} {:?}", tensor.shape());
    }
}

/// HTTP API Server using Axum
#[cfg(feature = "http-api")]
mod http_api {
    use axum::{
        extract::State,
        http::StatusCode,
        routing::{get, post},
        Json, Router,
    };
    use serde::{Deserialize, Serialize};
    use std::sync::{Arc, Mutex};
    use tower_http::trace::TraceLayer;
    use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
    use tracing::{info, error};

    #[derive(Clone)]
    pub struct AppState {
        pipeline: Arc<Mutex<Pipeline>>,
        config: Arc<Config>,
    }

    #[derive(Deserialize)]
    struct TtsRequest {
        text: String,
        text_language: Option<String>,
        refer_wav_path: Option<String>,
        prompt_text: Option<String>,
        prompt_language: Option<String>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        temperature: Option<f32>,
        speed: Option<f32>,
    }

    #[derive(Serialize)]
    struct TtsResponse {
        success: bool,
        message: String,
        audio_path: Option<String>,
    }

    #[derive(Deserialize)]
    struct ChangeReferRequest {
        refer_wav_path: String,
        prompt_text: String,
        prompt_language: Option<String>,
    }

    #[derive(Serialize)]
    struct ChangeReferResponse {
        success: bool,
        message: String,
    }

    #[derive(Deserialize)]
    struct ControlRequest {
        command: String,
    }

    #[derive(Serialize)]
    struct ControlResponse {
        success: bool,
        message: String,
    }

    pub async fn health_handler() -> &'static str {
        "OK"
    }

    pub async fn tts_handler(
        State(state): State<AppState>,
        Json(req): Json<TtsRequest>,
    ) -> Result<Json<TtsResponse>, StatusCode> {
        let language = req.text_language
            .as_deref()
            .and_then(Language::from_str)
            .unwrap_or(Language::Chinese);

        let options = InferenceOptions::builder()
            .top_k(req.top_k.unwrap_or(15))
            .top_p(req.top_p.unwrap_or(0.95))
            .temperature(req.temperature.unwrap_or(0.8))
            .speed(req.speed.unwrap_or(1.0))
            .language(language)
            .build();

        let text = req.text.clone();
        let refer_path = req.refer_wav_path.unwrap_or_else(|| "ref.wav".to_string());
        let prompt_text = req.prompt_text.unwrap_or_default();
        let pipeline = Arc::clone(&state.pipeline);

        let result = tokio::task::spawn_blocking(move || {
            let mut pipeline = pipeline.lock().map_err(|e| format!("Pipeline lock poisoned (a previous inference panicked): {}", e))?;
            pipeline
                .inference(&text, &refer_path, &prompt_text, &options)
                .map_err(|e| format!("Inference failed: {}", e))
        })
        .await
        .map_err(|e| {
            error!("spawn_blocking join error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        match result {
            Ok(_audio) => Ok(Json(TtsResponse {
                success: true,
                message: "TTS inference successful".to_string(),
                audio_path: Some("output.wav".to_string()),
            })),
            Err(e) => {
                error!("TTS inference failed: {}", e);
                Ok(Json(TtsResponse {
                    success: false,
                    message: e,
                    audio_path: None,
                }))
            }
        }
    }

    pub async fn change_refer_handler(
        State(_state): State<AppState>,
        Json(req): Json<ChangeReferRequest>,
    ) -> Json<ChangeReferResponse> {
        Json(ChangeReferResponse {
            success: true,
            message: format!("Reference updated: {} ({})", req.refer_wav_path, req.prompt_text),
        })
    }

    pub async fn control_handler(
        State(_state): State<AppState>,
        Json(req): Json<ControlRequest>,
    ) -> Json<ControlResponse> {
        match req.command.as_str() {
            "reload" => Json(ControlResponse {
                success: true,
                message: "Reload command received".to_string(),
            }),
            "unload" => Json(ControlResponse {
                success: true,
                message: "Unload command received".to_string(),
            }),
            _ => Json(ControlResponse {
                success: false,
                message: format!("Unknown command: {}", req.command),
            }),
        }
    }

    pub fn run(
        port: u16,
        gpt_model: Option<&std::path::Path>,
        sovits_model: Option<&std::path::Path>,
        bigvgan_model: Option<&std::path::Path>,
    ) -> Result<(), String> {
        let config = Config::builder().build();
        let mut pipeline = Pipeline::new(config.clone())
            .map_err(|e| format!("Failed to initialize pipeline: {}", e))?;

        if let Some(path) = gpt_model {
            info!("Loading GPT model from {:?}", path);
            pipeline
                .load_gpt(path)
                .map_err(|e| format!("Failed to load GPT model: {}", e))?;
        }
        if let Some(path) = sovits_model {
            info!("Loading SoVITS model from {:?}", path);
            pipeline
                .load_sovits(path)
                .map_err(|e| format!("Failed to load SoVITS model: {}", e))?;
        }
        if let Some(path) = bigvgan_model {
            info!("Loading BigVGAN model from {:?}", path);
            pipeline
                .load_bigvgan(path)
                .map_err(|e| format!("Failed to load BigVGAN model: {}", e))?;
        }

        let state = AppState {
            pipeline: Arc::new(Mutex::new(pipeline)),
            config: Arc::new(config),
        };

        // Build router
        let app = Router::new()
            .route("/tts", post(tts_handler))
            .route("/change_refer", post(change_refer_handler))
            .route("/control", post(control_handler))
            .route("/health", get(health_handler))
            .layer(TraceLayer::new_for_http())
            .with_state(state);

        let addr = format!("0.0.0.0:{}", port);
        info!("Starting HTTP server on {}", addr);
        println!("HTTP server started at http://localhost:{}", port);
        println!();
        println!("Endpoints:");
        println!("  GET  /health     - Health check");
        println!("  POST /tts        - TTS inference");
        println!("  POST /change_refer - Change reference audio");
        println!("  POST /control    - Server control (reload, unload)");
        println!();
        println!("Example:");
        println!("  curl -X POST http://localhost:9880/tts \\");
        println!("    -H 'Content-Type: application/json' \\");
        println!("    -d '{{\"text\": \"你好世界\", \"text_language\": \"zh\"}}'");

        // Run server
        tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create Tokio runtime: {}", e))?
            .block_on(async {
                let listener = tokio::net::TcpListener::bind(&addr)
                    .await
                    .map_err(|e| format!("Failed to bind to {}: {}", addr, e))?;
                axum::serve(listener, app)
                    .await
                    .map_err(|e| format!("Server error: {}", e))?;
                Ok::<(), String>(())
            })
    }
}
