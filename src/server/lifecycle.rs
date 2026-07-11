use super::request::{resolve_synthesis, SynthesisOverrides};
use super::response::json_error;
use super::state::AppState;
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    Json,
};
use gpt_sovits_rs::voice::list_voice_profiles;

pub(super) async fn health_handler() -> &'static str {
    "OK"
}

pub(super) async fn status_handler(
    State(state): State<AppState>,
) -> Result<Response<Body>, StatusCode> {
    let pipeline = state.pipelines.status().await;
    let body = serde_json::json!({
        "status": "ready",
        "cached_pipelines": pipeline.cached,
        "pipeline_cache_capacity": pipeline.capacity,
        "pipeline_cache_hits": pipeline.cache_hits,
        "pipeline_cache_misses": pipeline.cache_misses,
        "pipeline_evictions": pipeline.evictions,
        "queued_requests": pipeline.queued_requests,
        "busy": pipeline.busy,
        "gpu_inference_serialized": true,
        "max_text_chars": state.max_text_chars,
        "max_batch_items": state.max_batch_items,
    });
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap())
}

#[derive(serde::Deserialize)]
pub(super) struct WarmupRequest {
    voice: String,
}

pub(super) async fn warm_voice(state: &AppState, voice: &str) -> Result<(), String> {
    let voice = voice.trim();
    if voice.is_empty() {
        return Err("voice must not be empty".to_string());
    }

    let resolved = resolve_synthesis(
        Some(voice),
        None,
        None,
        None,
        SynthesisOverrides::default(),
        &state.voices_dir,
        &state.models_dir,
        state.path_policy,
    )?;
    let lease = state.pipelines.acquire_pipeline(&resolved.models).await?;
    tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let mut pipeline = rt.block_on(lease.pipeline.lock());
        pipeline
            .preload_speaker_with_options(
                &resolved.refer_path,
                &resolved.prompt_text,
                &resolved.options,
            )
            .map_err(|e| format!("Warmup failed: {e}"))
    })
    .await
    .map_err(|e| format!("Warmup task failed: {e}"))?
}

pub(super) async fn warmup_handler(
    State(state): State<AppState>,
    Json(req): Json<WarmupRequest>,
) -> Result<Response<Body>, StatusCode> {
    let voice = req.voice.trim();
    match warm_voice(&state, voice).await {
        Ok(()) => {
            let body = serde_json::json!({ "status": "ready", "voice": voice });
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap())
        }
        Err(e) => Ok(json_error(StatusCode::BAD_REQUEST, e)),
    }
}

pub(super) async fn voices_handler(
    State(state): State<AppState>,
) -> Result<Response<Body>, StatusCode> {
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
