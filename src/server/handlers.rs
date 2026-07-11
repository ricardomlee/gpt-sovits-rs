use super::audio::{samples_to_pcm, streaming_wav_header};
use super::pipeline_registry::PipelineLease;
use super::request::{
    resolve_synthesis, validate_text, BatchItemResult, OpenAiSpeechRequest, ResolvedSynthesis,
    SynthesisOverrides, TtsBatchRequest, TtsRequest,
};
use super::response::{add_synthesis_headers, json_error, language_code, SpeechOutputFormat};
use super::state::AppState;
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    Json,
};
use base64::Engine as _;
use gpt_sovits_rs::{AudioBuffer, InferenceOptions, Language, SplitMethod};
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt as _};
use tracing::{error, info, warn};

fn validate_request_text(text: &str, field: &str, max_chars: usize) -> Result<(), String> {
    validate_text(text, field)?;
    let chars = text.chars().count();
    if chars > max_chars {
        Err(format!(
            "{field} exceeds the {max_chars} character limit ({chars} received)"
        ))
    } else {
        Ok(())
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
        result.map_err(|e| format!("Inference failed: {e}"))
    })
    .await
    .map_err(|e| {
        error!("spawn_blocking join error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

pub(super) async fn openai_speech_handler(
    State(state): State<AppState>,
    Json(req): Json<OpenAiSpeechRequest>,
) -> Result<Response<Body>, StatusCode> {
    if let Err(e) = validate_request_text(&req.input, "input", state.max_text_chars) {
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
    let lease = match state
        .pipelines
        .acquire_pipeline(&resolved.models, state.queue_timeout)
        .await
    {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::SERVICE_UNAVAILABLE, e)),
    };
    let text = req.input;
    let text_chars = text.chars().count();
    let (job, voice, language) = into_buffered_job(text, resolved);
    let result = run_buffered_inference(lease, job).await?;

    match result {
        Ok(audio) => {
            let audio_bytes = match response_format {
                SpeechOutputFormat::Wav => audio.to_wav_bytes().map_err(|e| {
                    error!("Failed to encode WAV: {e}");
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
            error!("TTS inference failed: {e}");
            Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e))
        }
    }
}

/// Streams WAV audio sentence-by-sentence as chunked HTTP.
pub(super) async fn tts_stream_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response<Body>, StatusCode> {
    if let Err(e) = validate_request_text(&req.text, "text", state.max_text_chars) {
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
    let lease = match state
        .pipelines
        .acquire_pipeline(&resolved.models, state.queue_timeout)
        .await
    {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::SERVICE_UNAVAILABLE, e)),
    };
    let text = req.text;
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
    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>(8);

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(lease.pipeline.lock());
        if let Err(e) = pipeline.preload_speaker_with_options(&refer_path, &prompt_text, &options) {
            warn!("Failed to preload speaker: {e}");
        }

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
                    if tx.blocking_send(Ok(samples_to_pcm(&samples))).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    error!("Sentence inference failed: {e}");
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

pub(super) async fn tts_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsRequest>,
) -> Result<Response<Body>, StatusCode> {
    if let Err(e) = validate_request_text(&req.text, "text", state.max_text_chars) {
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
    let lease = match state
        .pipelines
        .acquire_pipeline(&resolved.models, state.queue_timeout)
        .await
    {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::SERVICE_UNAVAILABLE, e)),
    };
    let text = req.text;
    let text_chars = text.chars().count();
    let (job, voice, language) = into_buffered_job(text, resolved);
    let result = run_buffered_inference(lease, job).await?;

    match result {
        Ok(audio) => {
            let wav_bytes = audio.to_wav_bytes().map_err(|e| {
                error!("Failed to encode WAV: {e}");
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
            error!("TTS inference failed: {e}");
            Ok(json_error(StatusCode::INTERNAL_SERVER_ERROR, e))
        }
    }
}

/// Synthesizes multiple texts and streams one NDJSON result per item.
pub(super) async fn tts_batch_handler(
    State(state): State<AppState>,
    Json(req): Json<TtsBatchRequest>,
) -> Result<Response<Body>, StatusCode> {
    if req.texts.is_empty() {
        return Ok(json_error(StatusCode::BAD_REQUEST, "texts array is empty"));
    }
    if req.texts.len() > state.max_batch_items {
        return Ok(json_error(
            StatusCode::BAD_REQUEST,
            format!(
                "texts exceeds the {} item limit ({} received)",
                state.max_batch_items,
                req.texts.len()
            ),
        ));
    }
    if let Some((index, text)) =
        req.texts.iter().enumerate().find(|(_, text)| {
            validate_request_text(text, "batch item", state.max_text_chars).is_err()
        })
    {
        let error = validate_request_text(text, &format!("texts[{index}]"), state.max_text_chars)
            .expect_err("invalid batch item was selected");
        return Ok(json_error(StatusCode::BAD_REQUEST, error));
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
    let lease = match state
        .pipelines
        .acquire_pipeline(&resolved.models, state.queue_timeout)
        .await
    {
        Ok(lease) => lease,
        Err(e) => return Ok(json_error(StatusCode::SERVICE_UNAVAILABLE, e)),
    };

    let refer_path = resolved.refer_path;
    let prompt_text = resolved.prompt_text;
    let language_code = language_code(resolved.language);
    let voice = resolved.voice;
    let mode = resolved.mode;
    let options = resolved.options;
    let split_sentences = resolved.split_sentences;
    let split_method = resolved.split_method;
    let min_sentence_chars = resolved.min_sentence_chars;
    let sentence_gap_ms = resolved.sentence_gap_ms;
    let sentence_fade_ms = resolved.sentence_fade_ms;
    let texts = req.texts;
    let (tx, rx) = mpsc::channel::<Vec<u8>>(2);

    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(lease.pipeline.lock());
        if let Err(e) = pipeline.preload_speaker_with_options(&refer_path, &prompt_text, &options) {
            warn!("Speaker preload failed: {e}");
        }

        for (idx, text) in texts.iter().enumerate() {
            let started = std::time::Instant::now();
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
            let inference_ms = started.elapsed().as_millis() as u64;

            let item = match inference_result {
                Ok(audio) => {
                    let duration_s = audio.duration();
                    let sample_rate = audio.sample_rate;
                    match audio.to_wav_bytes() {
                        Ok(wav_bytes) => BatchItemResult {
                            index: idx,
                            voice: voice.clone(),
                            language: language_code,
                            text_chars: text.chars().count(),
                            wav_base64: Some(
                                base64::engine::general_purpose::STANDARD.encode(&wav_bytes),
                            ),
                            error: None,
                            sample_rate,
                            duration_s,
                            inference_ms,
                        },
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

            let mut line = serde_json::to_vec(&item).unwrap_or_default();
            line.push(b'\n');
            let duration_s = item.duration_s;
            if tx.blocking_send(line).is_err() {
                break;
            }
            info!(
                "batch[{}/{}] done: {:.0}ms, {:.2}s audio",
                idx + 1,
                texts.len(),
                inference_ms,
                duration_s
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

#[cfg(test)]
mod tests {
    use super::validate_request_text;

    #[test]
    fn text_limit_counts_unicode_characters() {
        assert!(validate_request_text("你好", "text", 2).is_ok());
        let error = validate_request_text("你好呀", "text", 2).unwrap_err();
        assert!(error.contains("2 character limit"));
        assert!(error.contains("3 received"));
    }

    #[test]
    fn text_limit_still_rejects_empty_input() {
        assert!(validate_request_text("  ", "text", 10).is_err());
    }
}
