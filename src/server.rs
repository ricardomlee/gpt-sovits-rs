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
use gpt_sovits_rs::voice::{
    list_voice_profiles, InferenceOptionOverrides, LoadedVoiceProfile, VoiceDefaults,
};
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline, SplitMethod};
use serde::Deserialize;
use serde::Serialize;
use std::path::{Path, PathBuf};
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
    voices_dir: Arc<PathBuf>,
    #[allow(dead_code)]
    config: Arc<Config>,
}

#[derive(Deserialize)]
struct TtsRequest {
    voice: Option<String>,
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

#[derive(Deserialize)]
struct OpenAiSpeechRequest {
    #[allow(dead_code)]
    model: Option<String>,
    #[serde(alias = "text")]
    input: String,
    #[serde(alias = "speakerVoice", alias = "speakerVoiceId", alias = "voiceId")]
    voice: String,
    #[serde(
        alias = "responseFormat",
        alias = "output_format",
        alias = "outputFormat"
    )]
    response_format: Option<String>,
    #[serde(alias = "languageCode", alias = "lang", alias = "language")]
    text_language: Option<String>,
    #[allow(dead_code)]
    instructions: Option<String>,
    speed: Option<f32>,
}

/// POST /tts/batch — synthesize multiple texts in one call.
/// Shared speaker features are computed once for all items.
/// Results stream back as NDJSON (one JSON line per item) as each completes.
#[derive(Deserialize)]
struct TtsBatchRequest {
    /// List of texts to synthesize (processed sequentially on GPU).
    texts: Vec<String>,
    voice: Option<String>,
    text_language: Option<String>,
    refer_wav_path: Option<String>,
    prompt_text: Option<String>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    speed: Option<f32>,
}

struct ResolvedSynthesis {
    voice: Option<String>,
    mode: String,
    language: Language,
    options: InferenceOptions,
    split_sentences: bool,
    split_method: SplitMethod,
    min_sentence_chars: usize,
    sentence_gap_ms: u32,
    sentence_fade_ms: u32,
    refer_path: String,
    prompt_text: String,
}

#[derive(Serialize)]
struct BatchItemResult {
    index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<String>,
    language: &'static str,
    text_chars: usize,
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

fn json_error(status: StatusCode, message: impl AsRef<str>) -> Response<Body> {
    let error_json = serde_json::json!({
        "success": false,
        "message": message.as_ref(),
    });
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(error_json.to_string()))
        .unwrap()
}

fn language_code(language: Language) -> &'static str {
    match language {
        Language::Chinese => "zh",
        Language::English => "en",
        Language::Japanese => "ja",
        Language::Korean => "ko",
        Language::Cantonese => "yue",
        Language::Auto => "auto",
    }
}

fn safe_header_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii() && !ch.is_ascii_control() {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn add_synthesis_headers(
    mut builder: axum::http::response::Builder,
    voice: Option<&str>,
    language: Language,
    text_chars: usize,
) -> axum::http::response::Builder {
    builder = builder
        .header("x-tts-language", language_code(language))
        .header("x-tts-text-chars", text_chars.to_string());

    if let Some(voice) = voice {
        builder = builder.header("x-tts-voice", safe_header_value(voice));
    }

    builder
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechOutputFormat {
    Wav,
    Pcm,
}

impl SpeechOutputFormat {
    fn parse(format: Option<&str>) -> Result<Self, String> {
        match format.unwrap_or("wav").to_ascii_lowercase().as_str() {
            "wav" => Ok(Self::Wav),
            "pcm" => Ok(Self::Pcm),
            other => Err(format!(
                "unsupported response_format: {other}; supported formats: wav, pcm"
            )),
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Pcm => "application/octet-stream",
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_synthesis(
    voice_name: Option<&str>,
    text_language: Option<&str>,
    refer_wav_path: Option<String>,
    prompt_text: Option<String>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    temperature: Option<f32>,
    speed: Option<f32>,
    voices_dir: &Path,
) -> Result<ResolvedSynthesis, String> {
    let voice = voice_name
        .map(|name| LoadedVoiceProfile::load(name, voices_dir))
        .transpose()?;
    let defaults = VoiceDefaults::from_profile(voice.as_ref().map(|v| &v.profile));

    let language_text = text_language.unwrap_or(&defaults.language);
    let language = Language::parse(language_text)
        .ok_or_else(|| format!("unsupported text_language: {language_text}"))?;

    let refer_path = refer_wav_path
        .or_else(|| {
            voice
                .as_ref()
                .and_then(|v| v.reference_audio_path())
                .map(|p| p.to_string_lossy().into_owned())
        })
        .ok_or_else(|| {
            "reference audio is required; select a configured voice or pass refer_wav_path"
                .to_string()
        })?;
    let prompt_text = prompt_text
        .or_else(|| {
            voice
                .as_ref()
                .and_then(|v| v.reference_text().map(str::to_string))
        })
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| {
            "reference text is required; select a configured voice or pass prompt_text".to_string()
        })?;

    let options = defaults.to_inference_options(
        language,
        InferenceOptionOverrides {
            top_k,
            top_p,
            temperature,
            speed,
            ..Default::default()
        },
    );

    Ok(ResolvedSynthesis {
        voice: voice.map(|v| v.name),
        mode: defaults.mode,
        language,
        options,
        split_sentences: defaults.split_sentences,
        split_method: defaults.split_method,
        min_sentence_chars: defaults.min_sentence_chars,
        sentence_gap_ms: defaults.sentence_gap_ms,
        sentence_fade_ms: defaults.sentence_fade_ms,
        refer_path,
        prompt_text,
    })
}

async fn openai_speech_handler(
    State(state): State<AppState>,
    Json(req): Json<OpenAiSpeechRequest>,
) -> Result<Response<Body>, StatusCode> {
    let response_format = match SpeechOutputFormat::parse(req.response_format.as_deref()) {
        Ok(format) => format,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };

    let resolved = match resolve_synthesis(
        Some(&req.voice),
        req.text_language.as_deref(),
        None,
        None,
        None,
        None,
        None,
        req.speed,
        &state.voices_dir,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };
    let text = req.input;
    let text_chars = text.chars().count();
    let voice = resolved.voice;
    let mode = resolved.mode;
    let language = resolved.language;
    let refer_path = resolved.refer_path;
    let prompt_text = resolved.prompt_text;
    let options = resolved.options;
    let split_sentences = resolved.split_sentences;
    let split_method = resolved.split_method;
    let min_sentence_chars = resolved.min_sentence_chars;
    let sentence_gap_ms = resolved.sentence_gap_ms;
    let sentence_fade_ms = resolved.sentence_fade_ms;
    let pipeline = Arc::clone(&state.pipeline);

    let result = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(pipeline.lock());
        let result = if split_sentences {
            pipeline.inference_split_with_method(
                &text,
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
            pipeline.inference_with_mode(&mode, &text, &refer_path, &prompt_text, &options)
        };
        result.map_err(|e| format!("Inference failed: {}", e))
    })
    .await
    .map_err(|e| {
        error!("spawn_blocking join error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

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
                .header(
                    "x-tts-response-format",
                    match response_format {
                        SpeechOutputFormat::Wav => "wav",
                        SpeechOutputFormat::Pcm => "pcm",
                    },
                )
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
    let resolved = match resolve_synthesis(
        req.voice.as_deref(),
        req.text_language.as_deref(),
        req.refer_wav_path,
        req.prompt_text,
        req.top_k,
        req.top_p,
        req.temperature,
        req.speed,
        &state.voices_dir,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
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
    let resolved = match resolve_synthesis(
        req.voice.as_deref(),
        req.text_language.as_deref(),
        req.refer_wav_path,
        req.prompt_text,
        req.top_k,
        req.top_p,
        req.temperature,
        req.speed,
        &state.voices_dir,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
    };
    let text = req.text.clone();
    let text_chars = text.chars().count();
    let voice = resolved.voice;
    let mode = resolved.mode;
    let language = resolved.language;
    let refer_path = resolved.refer_path;
    let prompt_text = resolved.prompt_text;
    let options = resolved.options;
    let split_sentences = resolved.split_sentences;
    let split_method = resolved.split_method;
    let min_sentence_chars = resolved.min_sentence_chars;
    let sentence_gap_ms = resolved.sentence_gap_ms;
    let sentence_fade_ms = resolved.sentence_fade_ms;
    let pipeline = Arc::clone(&state.pipeline);

    let result = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(pipeline.lock());
        let result = if split_sentences {
            pipeline.inference_split_with_method(
                &text,
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
            pipeline.inference_with_mode(&mode, &text, &refer_path, &prompt_text, &options)
        };
        result.map_err(|e| format!("Inference failed: {}", e))
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
        return Ok(Response::builder()
            .status(StatusCode::BAD_REQUEST)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(r#"{"error":"texts array is empty"}"#))
            .unwrap());
    }

    let resolved = match resolve_synthesis(
        req.voice.as_deref(),
        req.text_language.as_deref(),
        req.refer_wav_path,
        req.prompt_text,
        req.top_k,
        req.top_p,
        req.temperature,
        req.speed,
        &state.voices_dir,
    ) {
        Ok(resolved) => resolved,
        Err(e) => return Ok(json_error(StatusCode::BAD_REQUEST, e)),
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
    voices_dir: &std::path::Path,
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
    }

    let state = AppState {
        pipeline: Arc::new(tokio::sync::Mutex::new(pipeline)),
        voices_dir: Arc::new(voices_dir.to_path_buf()),
        config: Arc::new(config),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_legacy_reference_fields_without_voice() {
        let resolved = resolve_synthesis(
            None,
            Some("zh"),
            Some("ref.wav".to_string()),
            Some("prompt".to_string()),
            Some(20),
            Some(0.9),
            Some(0.7),
            Some(1.1),
            Path::new("voices"),
        )
        .unwrap();

        assert_eq!(resolved.language, Language::Chinese);
        assert_eq!(resolved.refer_path, "ref.wav");
        assert_eq!(resolved.prompt_text, "prompt");
        assert_eq!(resolved.options.top_k, 20);
        assert!((resolved.options.top_p - 0.9).abs() < 0.001);
        assert!((resolved.options.temperature - 0.7).abs() < 0.001);
        assert!((resolved.options.speed - 1.1).abs() < 0.001);
    }

    #[test]
    fn resolves_voice_profile_defaults() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("mao");
        std::fs::create_dir(&voice_dir).unwrap();
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"ref.wav",
                "reference_text":"会战兵力是八十万对六十万，优势在我",
                "language":"zh",
                "top_k":11,
                "top_p":0.8,
                "temperature":0.6,
                "speed":1.2,
                "max_tokens":123,
                "repetition_penalty":1.2
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("mao"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            temp.path(),
        )
        .unwrap();

        assert_eq!(resolved.voice.as_deref(), Some("mao"));
        assert_eq!(resolved.mode, "auto");
        assert_eq!(
            resolved.refer_path,
            voice_dir.join("ref.wav").to_string_lossy()
        );
        assert_eq!(resolved.prompt_text, "会战兵力是八十万对六十万，优势在我");
        assert_eq!(resolved.options.top_k, 11);
        assert!((resolved.options.top_p - 0.8).abs() < 0.001);
        assert!((resolved.options.temperature - 0.6).abs() < 0.001);
        assert!((resolved.options.speed - 1.2).abs() < 0.001);
        assert_eq!(resolved.options.max_tokens, 123);
        assert!((resolved.options.repetition_penalty - 1.2).abs() < 0.001);
    }

    #[test]
    fn request_values_override_voice_defaults() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("mao");
        std::fs::create_dir(&voice_dir).unwrap();
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"voice.wav",
                "reference_text":"voice prompt",
                "language":"zh",
                "top_k":11,
                "top_p":0.8,
                "temperature":0.6,
                "speed":1.2
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("mao"),
            Some("en"),
            Some("request.wav".to_string()),
            Some("request prompt".to_string()),
            Some(33),
            Some(0.7),
            Some(0.5),
            Some(0.9),
            temp.path(),
        )
        .unwrap();

        assert_eq!(resolved.language, Language::English);
        assert_eq!(resolved.refer_path, "request.wav");
        assert_eq!(resolved.prompt_text, "request prompt");
        assert_eq!(resolved.options.top_k, 33);
        assert!((resolved.options.top_p - 0.7).abs() < 0.001);
        assert!((resolved.options.temperature - 0.5).abs() < 0.001);
        assert!((resolved.options.speed - 0.9).abs() < 0.001);
    }

    #[test]
    fn resolves_voice_inference_mode() {
        let temp = tempfile::tempdir().unwrap();
        let voice_dir = temp.path().join("fast");
        std::fs::create_dir(&voice_dir).unwrap();
        std::fs::write(
            voice_dir.join("voice.json"),
            r#"{
                "reference_audio":"ref.wav",
                "reference_text":"prompt",
                "mode":"cuda-graph"
            }"#,
        )
        .unwrap();

        let resolved = resolve_synthesis(
            Some("fast"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            temp.path(),
        )
        .unwrap();

        assert_eq!(resolved.mode, "cuda-graph");
    }

    #[test]
    fn rejects_requests_without_reference_data() {
        let error = resolve_synthesis(
            None,
            Some("zh"),
            None,
            None,
            None,
            None,
            None,
            None,
            Path::new("voices"),
        )
        .err()
        .expect("missing reference data should fail");

        assert!(error.contains("reference audio is required"));
    }

    #[test]
    fn sanitizes_non_ascii_header_values() {
        assert_eq!(safe_header_value("mao"), "mao");
        assert_eq!(safe_header_value("角色 A"), "__ A");
    }

    #[test]
    fn accepts_only_lossless_speech_formats_for_openai_endpoint() {
        assert_eq!(
            SpeechOutputFormat::parse(None).unwrap(),
            SpeechOutputFormat::Wav
        );
        assert_eq!(
            SpeechOutputFormat::parse(Some("wav")).unwrap(),
            SpeechOutputFormat::Wav
        );
        assert_eq!(
            SpeechOutputFormat::parse(Some("PCM")).unwrap(),
            SpeechOutputFormat::Pcm
        );
        assert!(SpeechOutputFormat::parse(Some("mp3")).is_err());
        assert!(SpeechOutputFormat::parse(Some("opus")).is_err());
    }

    #[test]
    fn streaming_wav_header_encodes_the_generated_audio_format() {
        let header = streaming_wav_header(32_000, 2);

        assert_eq!(header.len(), 44);
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([header[22], header[23]]), 2);
        assert_eq!(
            u32::from_le_bytes([header[24], header[25], header[26], header[27]]),
            32_000
        );
        assert_eq!(
            u32::from_le_bytes([header[28], header[29], header[30], header[31]]),
            128_000
        );
        assert_eq!(u16::from_le_bytes([header[32], header[33]]), 4);
        assert_eq!(u16::from_le_bytes([header[34], header[35]]), 16);
    }
}
