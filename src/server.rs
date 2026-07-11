//! HTTP API server using Axum.

use axum::{
    routing::{get, post},
    Router,
};
use gpt_sovits_rs::Config;
use std::sync::Arc;
use tower_http::trace::TraceLayer;
use tracing::info;

mod audio;
mod handlers;
mod lifecycle;
mod pipeline_registry;
mod request;
mod response;
mod state;

use handlers::{openai_speech_handler, tts_batch_handler, tts_handler, tts_stream_handler};
use lifecycle::{health_handler, status_handler, voices_handler, warm_voice, warmup_handler};
use pipeline_registry::PipelineRegistry;
use state::AppState;

fn print_server_ready(port: u16, max_cached_pipelines: usize) {
    println!("HTTP server started at http://localhost:{port}");
    println!(
        "Model pipeline cache: {} entr{}",
        max_cached_pipelines.max(1),
        if max_cached_pipelines.max(1) == 1 {
            "y"
        } else {
            "ies"
        }
    );
    println!();
    println!("Endpoints:");
    println!("  GET  /health        - Health check");
    println!("  GET  /status        - Runtime and model-cache status");
    println!("  POST /warmup        - Load and warm one voice");
    println!("  GET  /voices        - List available voice profiles");
    println!("  POST /tts           - Single text -> audio/wav");
    println!("  POST /tts/stream    - Single text -> streaming audio/wav");
    println!("  POST /tts/batch     - Multiple texts -> NDJSON stream");
    println!("  POST /v1/audio/speech - OpenAI-compatible speech endpoint");
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    port: u16,
    device: &str,
    half_precision: bool,
    gpt_model: Option<&std::path::Path>,
    sovits_model: Option<&std::path::Path>,
    bigvgan_model: Option<&std::path::Path>,
    bert_model: Option<&std::path::Path>,
    hubert_model: Option<&std::path::Path>,
    max_cached_pipelines: usize,
    allow_external_reference_paths: bool,
    max_text_chars: usize,
    max_batch_items: usize,
    preload_voices: &[String],
    models_dir: &std::path::Path,
    voices_dir: &std::path::Path,
) -> Result<(), String> {
    let config = Config::builder()
        .with_device(device)
        .with_half_precision(half_precision)
        .build();
    let pipelines = PipelineRegistry::load(
        config,
        gpt_model,
        sovits_model,
        bigvgan_model,
        bert_model,
        hubert_model,
        max_cached_pipelines,
    )?;

    let state = AppState {
        pipelines,
        voices_dir: Arc::new(voices_dir.to_path_buf()),
        models_dir: Arc::new(models_dir.to_path_buf()),
        path_policy: request::RequestPathPolicy {
            allow_external_reference_paths,
        },
        max_text_chars,
        max_batch_items,
    };
    let startup_state = state.clone();

    let app = Router::new()
        .route("/tts", post(tts_handler))
        .route("/tts/stream", post(tts_stream_handler))
        .route("/tts/batch", post(tts_batch_handler))
        .route("/v1/audio/speech", post(openai_speech_handler))
        .route("/health", get(health_handler))
        .route("/status", get(status_handler))
        .route("/warmup", post(warmup_handler))
        .route("/voices", get(voices_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tokio::runtime::Runtime::new()
        .map_err(|e| format!("Failed to create Tokio runtime: {e}"))?
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(&addr)
                .await
                .map_err(|e| format!("Failed to bind to {addr}: {e}"))?;
            for voice in preload_voices.iter().map(|voice| voice.trim()) {
                if voice.is_empty() {
                    continue;
                }
                info!(voice, "Preloading voice");
                warm_voice(&startup_state, voice)
                    .await
                    .map_err(|e| format!("Failed to preload voice '{voice}': {e}"))?;
            }
            info!("Starting HTTP server on {addr}");
            print_server_ready(port, max_cached_pipelines);
            axum::serve(listener, app)
                .await
                .map_err(|e| format!("Server error: {e}"))?;
            Ok::<(), String>(())
        })
}
