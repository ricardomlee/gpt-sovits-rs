use super::request::{SynthesisOverrides, resolve_synthesis};
use super::response::json_error;
use super::shutdown::Shutdown;
use super::state::AppState;
use axum::{
    Json,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::Response,
};
use gpt_sovits_rs::voice::list_voice_profiles;

pub(super) async fn health_handler(State(shutdown): State<Shutdown>) -> (StatusCode, &'static str) {
    if shutdown.is_stopping() {
        (StatusCode::SERVICE_UNAVAILABLE, "STOPPING")
    } else {
        (StatusCode::OK, "OK")
    }
}

pub(super) async fn status_handler(
    State(state): State<AppState>,
) -> Result<Response<Body>, StatusCode> {
    let pipeline = state.pipelines.status().await;
    let body = serde_json::json!({
        "status": if pipeline.stopping { "stopping" } else { "ready" },
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
        "queue_timeout_seconds": state.queue_timeout.as_secs(),
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
    let lease = state
        .pipelines
        .acquire_pipeline(&resolved.models, state.queue_timeout)
        .await?;
    lease
        .spawn(move |pipeline| {
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
        Err(e) => Ok(json_error(
            if state.pipelines.shutdown.is_stopping() {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::BAD_REQUEST
            },
            e,
        )),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stopping_is_reflected_in_health_and_status() {
        let state = AppState::for_test();
        let shutdown = state.pipelines.shutdown.clone();
        assert_eq!(
            health_handler(State(shutdown.clone())).await,
            (StatusCode::OK, "OK")
        );
        shutdown.begin(std::time::Duration::from_secs(1));
        assert_eq!(
            health_handler(State(shutdown)).await,
            (StatusCode::SERVICE_UNAVAILABLE, "STOPPING")
        );
        let response = status_handler(State(state)).await.unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "stopping");
    }
}
