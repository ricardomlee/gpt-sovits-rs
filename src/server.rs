//! HTTP API server using Axum.

use axum::{
    Router,
    routing::{get, post},
};
use gpt_sovits_rs::Config;
use std::{
    net::{IpAddr, SocketAddr, TcpListener},
    sync::Arc,
};
use tower_http::trace::TraceLayer;
use tracing::info;

mod audio;
mod encoding;
mod handlers;
mod lifecycle;
mod pipeline_registry;
#[cfg(all(test, unix))]
mod process_tests;
mod request;
mod response;
mod shutdown;
mod state;

use handlers::{openai_speech_handler, tts_batch_handler, tts_handler, tts_stream_handler};
use lifecycle::{health_handler, status_handler, voices_handler, warm_voice, warmup_handler};
use pipeline_registry::PipelineRegistry;
use shutdown::Shutdown;
use state::AppState;

fn print_server_ready(addr: SocketAddr, max_cached_pipelines: usize) {
    println!("HTTP server started at http://{addr}");
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

fn bind_listener(host: IpAddr, port: u16) -> Result<TcpListener, String> {
    let addr = SocketAddr::new(host, port);
    let listener = TcpListener::bind(addr).map_err(|e| format!("Failed to bind to {addr}: {e}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("Failed to configure listener at {addr}: {e}"))?;
    Ok(listener)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    host: IpAddr,
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
    queue_timeout_secs: usize,
    preload_voices: &[String],
    models_dir: &std::path::Path,
    voices_dir: &std::path::Path,
) -> Result<(), String> {
    // Reserve the address before loading models so a bind failure is cheap.
    let listener = bind_listener(host, port)?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Failed to read HTTP listener address: {e}"))?;
    let config = Config::builder()
        .with_device(device)
        .with_half_precision(half_precision)
        .build();
    let gpt_model = gpt_model.map(std::path::PathBuf::from);
    let sovits_model = sovits_model.map(std::path::PathBuf::from);
    let bigvgan_model = bigvgan_model.map(std::path::PathBuf::from);
    let bert_model = bert_model.map(std::path::PathBuf::from);
    let hubert_model = hubert_model.map(std::path::PathBuf::from);
    let shutdown = Shutdown::default();
    shutdown::run(shutdown.clone(), shutdown::GRACE_PERIOD, async {
        let loading_shutdown = shutdown.clone();
        let pipelines = tokio::task::spawn_blocking(move || {
            PipelineRegistry::load(
                config,
                gpt_model.as_deref(),
                sovits_model.as_deref(),
                bigvgan_model.as_deref(),
                bert_model.as_deref(),
                hubert_model.as_deref(),
                max_cached_pipelines,
                loading_shutdown,
            )
        })
        .await
        .map_err(|error| format!("Model loading task failed: {error}"))??;
        if shutdown.is_stopping() {
            return Ok(());
        }
        let state = AppState {
            pipelines: pipelines.clone(),
            voices_dir: Arc::new(voices_dir.to_path_buf()),
            models_dir: Arc::new(models_dir.to_path_buf()),
            path_policy: request::RequestPathPolicy {
                allow_external_reference_paths,
            },
            max_text_chars,
            max_batch_items,
            queue_timeout: std::time::Duration::from_secs(queue_timeout_secs as u64),
        };
        for voice in preload_voices.iter().map(|voice| voice.trim()) {
            if shutdown.is_stopping() {
                pipelines.wait_for_idle().await;
                return Ok(());
            }
            if voice.is_empty() {
                continue;
            }
            info!(voice, "Preloading voice");
            if let Err(error) = warm_voice(&state, voice).await {
                if shutdown.is_stopping() {
                    pipelines.wait_for_idle().await;
                    return Ok(());
                }
                return Err(format!("Failed to preload voice '{voice}': {error}"));
            }
        }
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

        let listener = tokio::net::TcpListener::from_std(listener)
            .map_err(|e| format!("Failed to register listener at {addr}: {e}"))?;
        info!("Starting HTTP server on {addr}");
        print_server_ready(addr, max_cached_pipelines);
        serve(listener, app, pipelines).await
    })
}

async fn serve(
    listener: tokio::net::TcpListener,
    app: Router,
    pipelines: PipelineRegistry,
) -> Result<(), String> {
    let shutdown = pipelines.shutdown.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await
        .map_err(|error| format!("Server error: {error}"))?;
    pipelines.wait_for_idle().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn check_health_on(host: IpAddr) {
        let listener = bind_listener(host, 0).unwrap();
        let addr = listener.local_addr().unwrap();
        assert_eq!(addr.ip(), host);
        assert_ne!(addr.port(), 0);
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        let task = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/health", get(health_handler))
                    .with_state(Shutdown::default()),
            )
            .await
            .unwrap();
        });
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
            stream
                .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                .await
                .unwrap();
            let mut response = String::new();
            stream.read_to_string(&mut response).await.unwrap();
            response
        })
        .await;
        task.abort();
        let response = result.expect("health request should complete");
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
        assert!(response.ends_with("OK"), "{response}");
    }

    #[tokio::test]
    async fn health_is_reachable_on_ipv4_loopback() {
        check_health_on(Ipv4Addr::LOCALHOST.into()).await;
    }

    #[tokio::test]
    async fn health_is_reachable_on_ipv6_loopback() {
        check_health_on(Ipv6Addr::LOCALHOST.into()).await;
    }

    #[test]
    fn occupied_port_fails_before_loading_models() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let missing = std::path::Path::new("missing-model.safetensors");
        let error = run(
            addr.ip(),
            addr.port(),
            "cpu",
            false,
            Some(missing),
            Some(missing),
            None,
            None,
            None,
            1,
            false,
            100,
            1,
            1,
            &[],
            missing,
            missing,
        )
        .unwrap_err();
        assert!(
            error.starts_with(&format!("Failed to bind to {addr}:")),
            "{error}"
        );
    }
}
