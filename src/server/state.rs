use super::pipeline_registry::PipelineRegistry;
use super::request::RequestPathPolicy;
use super::shutdown::Shutdown;
use axum::extract::FromRef;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone)]
pub(super) struct AppState {
    pub(super) pipelines: PipelineRegistry,
    pub(super) voices_dir: Arc<PathBuf>,
    pub(super) models_dir: Arc<PathBuf>,
    pub(super) path_policy: RequestPathPolicy,
    pub(super) max_text_chars: usize,
    pub(super) max_batch_items: usize,
    pub(super) queue_timeout: Duration,
}

impl FromRef<AppState> for Shutdown {
    fn from_ref(state: &AppState) -> Self {
        state.pipelines.shutdown.clone()
    }
}

#[cfg(test)]
impl AppState {
    pub(super) fn for_test() -> Self {
        Self {
            pipelines: PipelineRegistry::for_test(),
            voices_dir: Arc::new(PathBuf::from(".")),
            models_dir: Arc::new(PathBuf::from(".")),
            path_policy: RequestPathPolicy {
                allow_external_reference_paths: false,
            },
            max_text_chars: 100,
            max_batch_items: 4,
            queue_timeout: Duration::from_secs(60),
        }
    }
}
