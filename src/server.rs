//! HTTP API server using Axum.

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use base64::Engine as _;
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use serde::Deserialize;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::{wrappers::ReceiverStream, StreamExt as _};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

#[derive(Clone)]
pub struct AppState {
    /// tokio async mutex: concurrent requests await the lock without blocking OS threads.
    /// The GPU is single-threaded; this enforces sequential inference with async queuing.
    pipeline: Arc<Mutex<Pipeline>>,
    #[allow(dead_code)]
    config: Arc<Config>,
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    text_language: Option<String>,
    refer_wav_path: Option<String>,
    prompt_text: Option<String>,
    #[allow(dead_code)]
    prompt_language: Option<String>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    speed: Option<f32>,
}

/// POST /tts/batch — synthesize multiple texts in one call.
/// Shared speaker features are computed once for all items.
/// Results stream back as NDJSON (one JSON line per item) as each completes.
#[derive(Deserialize)]
struct TtsBatchRequest {
    /// List of texts to synthesize (processed sequentially on GPU).
    texts: Vec<String>,
    text_language: Option<String>,
    refer_wav_path: Option<String>,
    prompt_text: Option<String>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    speed: Option<f32>,
}

#[derive(Serialize)]
struct BatchItemResult {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    wav_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    sample_rate: u32,
    duration_s: f32,
    inference_ms: u64,
}

async fn health_handler() -> &'static str {
    "OK"
}

/// Build a streaming WAV header for unknown-length audio (size=0xFFFFFFFF).
/// Compatible with ffplay, mpv, VLC, curl | aplay.
fn streaming_wav_header(sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size: u32 = 0xFFFF_FFFE; // streaming sentinel
    let riff_size: u32 = data_size; // also sentinel — no accurate total known

    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&riff_size.to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&byte_rate.to_le_bytes());
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&bits_per_sample.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_size.to_le_bytes());
    h
}

/// Encode f32 samples as i16 PCM bytes.
fn samples_to_pcm(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// POST /tts/stream — streams WAV audio sentence-by-sentence as chunked HTTP.
/// First byte arrives after the first sentence is synthesized (~1-2s for short sentences).
async fn tts_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response<Body>, StatusCode> {
    let language = req
        .text_language
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

    // Channel: inference thread sends PCM chunks; HTTP task streams them out
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>(8);

    tokio::task::spawn_blocking(move || {
        // tokio::sync::Mutex must be locked via block_on inside spawn_blocking
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(pipeline.lock());

        // Preload speaker features (cached — free on 2nd+ call)
        if let Err(e) = pipeline.preload_speaker(&refer_path, &prompt_text, options.language) {
            warn!("Failed to preload speaker: {}", e);
        }

        // Send WAV header first (sample_rate and channels are fixed)
        let header = streaming_wav_header(24000, 1);
        if tx.blocking_send(Ok(header)).is_err() {
            return;
        }

        // Stream each sentence
        for result in pipeline.inference_sentences(&text, &refer_path, &prompt_text, &options, 5) {
            match result {
                Ok(audio) => {
                    let pcm = samples_to_pcm(&audio.samples);
                    if tx.blocking_send(Ok(pcm)).is_err() {
                        break; // client disconnected
                    }
                }
                Err(e) => {
                    error!("Sentence inference failed: {}", e);
                    let _ = tx.blocking_send(Err(e.to_string()));
                    break;
                }
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|item| item.map_err(|e| std::io::Error::other(e)));

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(stream))
        .unwrap())
}

async fn tts_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response<Body>, StatusCode> {
    let language = req
        .text_language
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
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(pipeline.lock());
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
        Ok(audio) => {
            let wav_bytes = audio.to_wav_bytes().map_err(|e| {
                error!("Failed to encode WAV: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "audio/wav")
                .header(
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"tts_output.wav\"",
                )
                .body(Body::from(wav_bytes))
                .unwrap())
        }
        Err(e) => {
            error!("TTS inference failed: {}", e);
            let error_json = serde_json::json!({
                "success": false,
                "message": e,
            });
            Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(error_json.to_string()))
                .unwrap())
        }
    }
}

/// POST /tts/batch — batch synthesis with NDJSON streaming response.
///
/// Accepts multiple texts in a single request. Speaker features are preloaded once
/// and shared across all items. Results stream back as NDJSON as each item completes,
/// so the client can start decoding the first audio while later items are still running.
///
/// Response format (one JSON line per item, in order):
///   {"index":0,"wav_base64":"...","sample_rate":32000,"duration_s":1.5,"inference_ms":2100}
///   {"index":1,"wav_base64":"...","sample_rate":32000,"duration_s":2.2,"inference_ms":3100}
async fn tts_batch_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsBatchRequest>,
) -> Result<Response<Body>, StatusCode> {
    if req.texts.is_empty() {
        return Ok(Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"error":"texts array is empty"}"#))
            .unwrap());
    }

    let language = req
        .text_language
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

    let refer_path = req.refer_wav_path.unwrap_or_else(|| "ref.wav".to_string());
    let prompt_text = req.prompt_text.unwrap_or_default();
    let texts = req.texts;
    let pipeline = Arc::clone(&state.pipeline);

    // Channel: inference thread sends one NDJSON line per completed item.
    // Buffer=2 so inference stays one item ahead of the HTTP sender.
    let (tx, rx) = mpsc::channel::<Vec<u8>>(2);

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        // Acquire the async mutex from this blocking context.
        // Other concurrent requests will await the lock without consuming OS threads.
        let mut pipeline = rt.block_on(pipeline.lock());

        // Preload speaker once — free on subsequent calls (cache hit)
        if let Err(e) = pipeline.preload_speaker(&refer_path, &prompt_text, language) {
            warn!("Speaker preload failed: {}", e);
        }

        for (idx, text) in texts.iter().enumerate() {
            let t = std::time::Instant::now();
            let inference_result =
                pipeline.inference_cuda_graph(text, &refer_path, &prompt_text, &options);
            let inference_ms = t.elapsed().as_millis() as u64;

            let item = match inference_result {
                Ok(audio) => {
                    let duration_s = audio.duration();
                    let sample_rate = audio.sample_rate;
                    match audio.to_wav_bytes() {
                        Ok(wav_bytes) => {
                            let wav_b64 =
                                base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
                            BatchItemResult {
                                index: idx,
                                wav_base64: Some(wav_b64),
                                error: None,
                                sample_rate,
                                duration_s,
                                inference_ms,
                            }
                        }
                        Err(e) => BatchItemResult {
                            index: idx,
                            wav_base64: None,
                            error: Some(format!("WAV encode failed: {e}")),
                            sample_rate: 0,
                            duration_s: 0.0,
                            inference_ms,
                        },
                    }
                }
                Err(e) => BatchItemResult {
                    index: idx,
                    wav_base64: None,
                    error: Some(e.to_string()),
                    sample_rate: 0,
                    duration_s: 0.0,
                    inference_ms,
                },
            };

            // Serialize as one NDJSON line
            let mut line = serde_json::to_vec(&item).unwrap_or_default();
            line.push(b'\n');

            let dur = item.duration_s;
            if tx.blocking_send(line).is_err() {
                break; // client disconnected
            }

            info!(
                "batch[{}/{}] done: {:.0}ms, {:.2}s audio",
                idx + 1,
                texts.len(),
                inference_ms,
                dur
            );
        }
    });

    let stream =
        ReceiverStream::new(rx).map(|bytes| -> Result<axum::body::Bytes, std::io::Error> {
            Ok(axum::body::Bytes::from(bytes))
        });

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .header(header::TRANSFER_ENCODING, "chunked")
        .body(Body::from_stream(stream))
        .unwrap())
}

pub fn run(
    port: u16,
    device: &str,
    half_precision: bool,
    gpt_model: Option<&std::path::Path>,
    sovits_model: Option<&std::path::Path>,
    bigvgan_model: Option<&std::path::Path>,
    bert_model: Option<&std::path::Path>,
    hubert_model: Option<&std::path::Path>,
) -> Result<(), String> {
    let config = Config::builder()
        .with_device(device)
        .with_half_precision(half_precision)
        .build();
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
    if let Some(path) = bert_model {
        info!("Loading BERT model from {:?}", path);
        if let Err(e) = pipeline.load_bert(path) {
            error!("Failed to load BERT model (continuing without it): {}", e);
        }
    }
    if let Some(path) = hubert_model {
        info!("Loading Hubert model from {:?}", path);
        if let Err(e) = pipeline.load_hubert(path) {
            error!("Failed to load Hubert model (continuing without it): {}", e);
        }
        // Semantic tokenizer needed for speaker feature extraction from reference audio
        if let Some(sovits_path) = sovits_model {
            info!("Loading semantic tokenizer from SoVITS weights...");
            if let Err(e) = pipeline.load_semantic_tokenizer(sovits_path) {
                error!("Failed to load semantic tokenizer: {}", e);
            }
        }
    }

    let state = AppState {
        pipeline: Arc::new(tokio::sync::Mutex::new(pipeline)),
        config: Arc::new(config),
    };

    // Build router
    let app = Router::new()
        .route("/tts", post(tts_handler))
        .route("/tts/stream", post(tts_stream_handler))
        .route("/tts/batch", post(tts_batch_handler))
        .route("/health", get(health_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting HTTP server on {}", addr);
    println!("HTTP server started at http://localhost:{}", port);
    println!();
    println!("Endpoints:");
    println!("  GET  /health        - Health check");
    println!("  POST /tts           - Single text → audio/wav");
    println!("  POST /tts/stream    - Single text → streaming audio/wav (sentence by sentence)");
    println!("  POST /tts/batch     - Multiple texts → NDJSON stream (one result per line)");
    println!();
    println!("Example:");
    println!("  curl -X POST http://localhost:9880/tts \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"text\": \"你好世界\", \"text_language\": \"zh\", \"refer_wav_path\": \"ref.wav\", \"prompt_text\": \"参考文本\"}}' \\");
    println!("    --output tts_output.wav");

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
