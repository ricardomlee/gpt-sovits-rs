use super::pipeline_registry::PipelineRegistry;
use super::request::RequestPathPolicy;
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
