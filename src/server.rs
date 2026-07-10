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
use gpt_sovits_rs::voice::list_voice_profiles;
use gpt_sovits_rs::{AudioBuffer, Config, InferenceOptions, Language, SplitMethod};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt as _};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

mod audio;
mod pipeline_registry;
mod request;
mod response;

use audio::{samples_to_pcm, streaming_wav_header};
use pipeline_registry::{PipelineLease, PipelineRegistry};
use request::{
    resolve_synthesis, validate_text, BatchItemResult, OpenAiSpeechRequest, ResolvedSynthesis,
    SynthesisOverrides, TtsBatchRequest, TtsRequest,
};
use response::{add_synthesis_headers, json_error, language_code, SpeechOutputFormat};

#[derive(Clone)]
pub struct AppState {
    pipelines: PipelineRegistry,
    voices_dir: Arc<PathBuf>,
    models_dir: Arc<PathBuf>,
    path_policy: request::RequestPathPolicy,
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn voices_handler(State(state): State<AppState>) -> Result<Response<Body>, StatusCode> {
    match list_voice_profiles(&state.voices_dir) {
        Ok(voices) => {
            let body = serde_json::json!({ "voices": voices });
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap())
        }
        Err(e) => Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

struct BufferedInferenceJob {
    text: String,
    mode: String,
    refer_path: String,
    prompt_text: String,
    options: InferenceOptions,
    split_sentences: bool,
    split_method: SplitMethod,
    min_sentence_chars: usize,
    sentence_gap_ms: u32,
    sentence_fade_ms: u32,
}

fn into_buffered_job(
    text: String,
    resolved: ResolvedSynthesis,
) -> (BufferedInferenceJob, Option<String>, Language) {
    let voice = resolved.voice;
    let language = resolved.language;
    let job = BufferedInferenceJob {
        text,
        mode: resolved.mode,
        refer_path: resolved.refer_path,
        prompt_text: resolved.prompt_text,
        options: resolved.options,
        split_sentences: resolved.split_sentences,
        split_method: resolved.split_method,
        min_sentence_chars: resolved.min_sentence_chars,
        sentence_gap_ms: resolved.sentence_gap_ms,
        sentence_fade_ms: resolved.sentence_fade_ms,
    };
    (job, voice, language)
}

async fn run_buffered_inference(
    lease: PipelineLease,
    job: BufferedInferenceJob,
) -> Result<Result<AudioBuffer, String>, StatusCode> {
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(lease.pipeline.lock());
        let result = if job.split_sentences {
            pipeline.inference_split_with_method(
                &job.text,
                &job.refer_path,
                &job.prompt_text,
                &job.options,
                &job.mode,
                job.min_sentence_chars,
                job.sentence_gap_ms,
                job.sentence_fade_ms,
                job.split_method,
            )
        } else {
            pipeline.inference_with_mode(
                &job.mode,
                &job.text,
                &job.refer_path,
                &job.prompt_text,
                &job.options,
            )
        };
        result.map_err(|e| format!("Inference failed: {}", e))
    })
    .await
    .map_err(|e| {
        error!("spawn_blocking join error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

async fn openai_speech_handler(
    State(state): State<AppState>,
    Json(req): Json<OpenAiSpeechRequest>,
) -> Result<Response<Body>, StatusCode> {
    if let Err(e) = validate_text(&req.input, "input") {
        return Ok(json_error(StatusCode::BAD_REQUEST, e));
    }
    if req.voice.trim().is_empty() {
        return Ok(json_error(
            StatusCode::BAD_REQUEST,
            "voice must not be empty",
        ));
    }

    let response_format = match SpeechOutputFormat::parse(req.response_format.as_deref()) {
        Ok(format) => format,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };

    let resolved = match resolve_synthesis(
        Some(&req.voice),
        req.text_language.as_deref(),
        None,
        None,
        SynthesisOverrides::from_openai(&req),
        &state.voices_dir,
        &state.models_dir,
        state.path_policy,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };
    let lease = match state.pipelines.acquire_pipeline(&resolved.models).await {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e)),
    };
    let text = req.input;
    let text_chars = text.chars().count();
    let (job, voice, language) = into_buffered_job(text, resolved);
    let result = run_buffered_inference(lease, job).await?;

    match result {
        Ok(audio) => {
            let audio_bytes = match response_format {
                SpeechOutputFormat::Wav => audio.to_wav_bytes().map_err(|e| {
                    error!("Failed to encode WAV: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?,
                SpeechOutputFormat::Pcm => samples_to_pcm(&audio.samples),
            };

            let builder =
                add_synthesis_headers(Response::builder(), voice.as_deref(), language, text_chars);

            Ok(builder
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, response_format.content_type())
                .header("x-tts-api", "openai-audio-speech")
                .header("x-tts-response-format", response_format.as_header_value())
                .header("x-tts-duration-s", format!("{:.3}", audio.duration()))
                .header("x-tts-sample-rate", audio.sample_rate.to_string())
                .header("x-tts-channels", audio.channels.to_string())
                .body(Body::from(audio_bytes))
                .unwrap())
        }
        Err(e) => {
            error!("TTS inference failed: {}", e);
            Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e))
        }
    }
}

/// POST /tts/stream — streams WAV audio sentence-by-sentence as chunked HTTP.
/// First byte arrives after the first sentence is synthesized (~1-2s for short sentences).
async fn tts_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response<Body>, StatusCode> {
    if let Err(e) = validate_text(&req.text, "text") {
        return Ok(json_error(StatusCode::BAD_REQUEST, e));
    }

    let overrides = SynthesisOverrides::from_tts(&req);
    let resolved = match resolve_synthesis(
        req.voice.as_deref(),
        req.text_language.as_deref(),
        req.refer_wav_path,
        req.prompt_text,
        overrides,
        &state.voices_dir,
        &state.models_dir,
        state.path_policy,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };
    let lease = match state.pipelines.acquire_pipeline(&resolved.models).await {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e)),
    };
    let text = req.text.clone();
    let text_chars = text.chars().count();
    let voice = resolved.voice;
    let mode = resolved.mode;
    let language = resolved.language;
    let refer_path = resolved.refer_path;
    let prompt_text = resolved.prompt_text;
    let options = resolved.options;
    let split_method = resolved.split_method;
    let min_sentence_chars = resolved.min_sentence_chars;
    let sentence_gap_ms = resolved.sentence_gap_ms;
    let sentence_fade_ms = resolved.sentence_fade_ms;

    // Channel: inference thread sends PCM chunks; HTTP task streams them out
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>(8);

    tokio::task::spawn_blocking(move || {
        // tokio::sync::Mutex must be locked via block_on inside spawn_blocking
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(lease.pipeline.lock());

        // Preload speaker features (cached — free on 2nd+ call)
        if let Err(e) = pipeline.preload_speaker_with_options(&refer_path, &prompt_text, &options) {
            warn!("Failed to preload speaker: {}", e);
        }

        // Stream each sentence
        let mut header_sent = false;
        for result in pipeline.inference_sentences_with_method(
            &text,
            &refer_path,
            &prompt_text,
            &options,
            &mode,
            min_sentence_chars,
            split_method,
        ) {
            match result {
                Ok(mut audio) => {
                    if sentence_fade_ms > 0 {
                        audio.fade_in(sentence_fade_ms);
                        audio.fade_out(sentence_fade_ms);
                    }
                    if !header_sent {
                        let header = streaming_wav_header(audio.sample_rate, audio.channels);
                        if tx.blocking_send(Ok(header)).is_err() {
                            break;
                        }
                        header_sent = true;
                    }
                    let mut samples = audio.samples;
                    let gap_samples = (sentence_gap_ms as f32 * audio.sample_rate as f32 / 1000.0)
                        as usize
                        * audio.channels as usize;
                    samples.resize(samples.len() + gap_samples, 0.0);
                    let pcm = samples_to_pcm(&samples);
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

    let stream = ReceiverStream::new(rx).map(|item| item.map_err(std::io::Error::other));

    let builder =
        add_synthesis_headers(Response::builder(), voice.as_deref(), language, text_chars);

    Ok(builder
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::TRANSFER_ENCODING, "chunked")
        .header("x-tts-streaming", "true")
        .body(Body::from_stream(stream))
        .unwrap())
}

async fn tts_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response<Body>, StatusCode> {
    if let Err(e) = validate_text(&req.text, "text") {
        return Ok(json_error(StatusCode::BAD_REQUEST, e));
    }

    let overrides = SynthesisOverrides::from_tts(&req);
    let resolved = match resolve_synthesis(
        req.voice.as_deref(),
        req.text_language.as_deref(),
        req.refer_wav_path,
        req.prompt_text,
        overrides,
        &state.voices_dir,
        &state.models_dir,
        state.path_policy,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };
    let lease = match state.pipelines.acquire_pipeline(&resolved.models).await {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e)),
    };
    let text = req.text.clone();
    let text_chars = text.chars().count();
    let (job, voice, language) = into_buffered_job(text, resolved);
    let result = run_buffered_inference(lease, job).await?;

    match result {
        Ok(audio) => {
            let wav_bytes = audio.to_wav_bytes().map_err(|e| {
                error!("Failed to encode WAV: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            let builder =
                add_synthesis_headers(Response::builder(), voice.as_deref(), language, text_chars);

            Ok(builder
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "audio/wav")
                .header(
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=\"tts_output.wav\"",
                )
                .header("x-tts-duration-s", format!("{:.3}", audio.duration()))
                .header("x-tts-sample-rate", audio.sample_rate.to_string())
                .header("x-tts-channels", audio.channels.to_string())
                .body(Body::from(wav_bytes))
                .unwrap())
        }
        Err(e) => {
            error!("TTS inference failed: {}", e);
            Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e))
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
        return Ok(json_error(StatusCode::BAD_REQUEST, "texts array is empty"));
    }
    if let Some((index, _)) = req
        .texts
        .iter()
        .enumerate()
        .find(|(_, text)| text.trim().is_empty())
    {
        return Ok(json_error(
            StatusCode::BAD_REQUEST,
            format!("texts[{index}] must not be empty"),
        ));
    }

    let overrides = SynthesisOverrides::from_batch(&req);
    let resolved = match resolve_synthesis(
        req.voice.as_deref(),
        req.text_language.as_deref(),
        req.refer_wav_path,
        req.prompt_text,
        overrides,
        &state.voices_dir,
        &state.models_dir,
        state.path_policy,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };
    let lease = match state.pipelines.acquire_pipeline(&resolved.models).await {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e)),
    };

    let refer_path = resolved.refer_path;
    let prompt_text = resolved.prompt_text;
    let language = resolved.language;
    let language_code = language_code(language);
    let voice = resolved.voice;
    let mode = resolved.mode;
    let options = resolved.options;
    let split_sentences = resolved.split_sentences;
    let split_method = resolved.split_method;
    let min_sentence_chars = resolved.min_sentence_chars;
    let sentence_gap_ms = resolved.sentence_gap_ms;
    let sentence_fade_ms = resolved.sentence_fade_ms;
    let texts = req.texts;

    // Channel: inference thread sends one NDJSON line per completed item.
    // Buffer=2 so inference stays one item ahead of the HTTP sender.
    let (tx, rx) = mpsc::channel::<Vec<u8>>(2);

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        // Acquire the async mutex from this blocking context.
        // Other concurrent requests will await the lock without consuming OS threads.
        let mut pipeline = rt.block_on(lease.pipeline.lock());

        // Preload speaker once — free on subsequent calls (cache hit)
        if let Err(e) = pipeline.preload_speaker_with_options(&refer_path, &prompt_text, &options) {
            warn!("Speaker preload failed: {}", e);
        }

        for (idx, text) in texts.iter().enumerate() {
            let t = std::time::Instant::now();
            let inference_result = if split_sentences {
                pipeline.inference_split_with_method(
                    text,
                    &refer_path,
                    &prompt_text,
                    &options,
                    &mode,
                    min_sentence_chars,
                    sentence_gap_ms,
                    sentence_fade_ms,
                    split_method,
                )
            } else {
                pipeline.inference_with_mode(&mode, text, &refer_path, &prompt_text, &options)
            };
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
                                voice: voice.clone(),
                                language: language_code,
                                text_chars: text.chars().count(),
                                wav_base64: Some(wav_b64),
                                error: None,
                                sample_rate,
                                duration_s,
                                inference_ms,
                            }
                        }
                        Err(e) => BatchItemResult {
                            index: idx,
                            voice: voice.clone(),
                            language: language_code,
                            text_chars: text.chars().count(),
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
                    voice: voice.clone(),
                    language: language_code,
                    text_chars: text.chars().count(),
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
    };

    // Build router
    let app = Router::new()
        .route("/tts", post(tts_handler))
        .route("/tts/stream", post(tts_stream_handler))
        .route("/tts/batch", post(tts_batch_handler))
        .route("/v1/audio/speech", post(openai_speech_handler))
        .route("/health", get(health_handler))
        .route("/voices", get(voices_handler))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting HTTP server on {}", addr);
    println!("HTTP server started at http://localhost:{}", port);
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
    println!("  GET  /voices        - List available voice profiles");
    println!("  POST /tts           - Single text → audio/wav");
    println!("  POST /tts/stream    - Single text → streaming audio/wav (sentence by sentence)");
    println!("  POST /tts/batch     - Multiple texts → NDJSON stream (one result per line)");
    println!("  POST /v1/audio/speech - OpenAI-compatible speech endpoint");
    println!();
    println!("Example:");
    println!("  curl -X POST http://localhost:{port}/tts \\");
    println!("    -H 'Content-Type: application/json' \\");
    println!("    -d '{{\"text\": \"你好世界\", \"text_language\": \"zh\", \"refer_wav_path\": \"voices/demo/ref.wav\", \"prompt_text\": \"参考文本\"}}' \\");
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
