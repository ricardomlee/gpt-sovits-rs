//! Bounded model-pipeline cache and global inference serialization.

use gpt_sovits_rs::voice::VoiceModelPaths;
use gpt_sovits_rs::{Config, Pipeline, SharedPipelineResources};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{error, info};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    gpt: Option<PathBuf>,
    sovits: Option<PathBuf>,
    bigvgan: Option<PathBuf>,
}

struct PipelineCache {
    entries: HashMap<PipelineKey, Arc<Mutex<Pipeline>>>,
    lru: VecDeque<PipelineKey>,
    capacity: usize,
}

impl PipelineCache {
    fn new(capacity: usize, key: PipelineKey, pipeline: Pipeline) -> Self {
        let mut entries = HashMap::new();
        entries.insert(key.clone(), Arc::new(Mutex::new(pipeline)));
        Self {
            entries,
            lru: VecDeque::from([key]),
            capacity: capacity.max(1),
        }
    }

    fn get(&mut self, key: &PipelineKey) -> Option<Arc<Mutex<Pipeline>>> {
        let pipeline = self.entries.get(key).cloned()?;
        self.touch(key);
        Some(pipeline)
    }

    fn evict_before_insert(
        &mut self,
        key: &PipelineKey,
    ) -> Option<(PipelineKey, Arc<Mutex<Pipeline>>)> {
        if self.entries.contains_key(key) || self.entries.len() < self.capacity {
            return None;
        }
        while let Some(oldest) = self.lru.pop_front() {
            if let Some(pipeline) = self.entries.remove(&oldest) {
                return Some((oldest, pipeline));
            }
        }
        None
    }

    fn insert(&mut self, key: PipelineKey, pipeline: Arc<Mutex<Pipeline>>) {
        self.entries.insert(key.clone(), pipeline);
        self.touch(&key);
    }

    fn touch(&mut self, key: &PipelineKey) {
        self.lru.retain(|candidate| candidate != key);
        self.lru.push_back(key.clone());
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[derive(Clone)]
pub(super) struct PipelineRegistry {
    config: Arc<Config>,
    defaults: Arc<PipelineKey>,
    shared_resources: Arc<SharedPipelineResources>,
    cache: Arc<Mutex<PipelineCache>>,
    operation: Arc<Mutex<()>>,
    metrics: Arc<PipelineMetrics>,
}

pub(super) struct PipelineLease {
    pub(super) pipeline: Arc<Mutex<Pipeline>>,
    _operation: OwnedMutexGuard<()>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PipelineStatus {
    pub(super) cached: usize,
    pub(super) capacity: usize,
    pub(super) cache_hits: u64,
    pub(super) cache_misses: u64,
    pub(super) evictions: u64,
    pub(super) queued_requests: usize,
    pub(super) busy: bool,
}

#[derive(Default)]
struct PipelineMetrics {
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    evictions: AtomicU64,
    queued_requests: AtomicUsize,
}

struct QueueGuard(Arc<PipelineMetrics>);

impl QueueGuard {
    fn new(metrics: Arc<PipelineMetrics>) -> Self {
        metrics.queued_requests.fetch_add(1, Ordering::Relaxed);
        Self(metrics)
    }
}

impl Drop for QueueGuard {
    fn drop(&mut self) {
        self.0.queued_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

impl PipelineRegistry {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        config: Config,
        gpt_model: Option<&Path>,
        sovits_model: Option<&Path>,
        bigvgan_model: Option<&Path>,
        bert_model: Option<&Path>,
        hubert_model: Option<&Path>,
        cache_capacity: usize,
    ) -> Result<Self, String> {
        let defaults = PipelineKey {
            gpt: gpt_model.map(PathBuf::from),
            sovits: sovits_model.map(PathBuf::from),
            bigvgan: bigvgan_model.map(PathBuf::from),
        }
        .normalized();
        let mut default_pipeline = load_pipeline(config.clone(), &defaults, None)?;
        if let Some(path) = bert_model {
            info!("Loading shared BERT model from {:?}", path);
            if let Err(e) = default_pipeline.load_bert(path) {
                error!("Failed to load BERT model (continuing without it): {}", e);
            }
        }
        if let Some(path) = hubert_model {
            info!("Loading shared HuBERT model from {:?}", path);
            if let Err(e) = default_pipeline.load_hubert(path) {
                error!("Failed to load HuBERT model (continuing without it): {}", e);
            }
        }
        Ok(Self::new(
            config,
            defaults,
            default_pipeline,
            cache_capacity,
        ))
    }

    fn new(
        config: Config,
        defaults: PipelineKey,
        default_pipeline: Pipeline,
        cache_capacity: usize,
    ) -> Self {
        let shared_resources = default_pipeline.shared_resources();
        Self {
            config: Arc::new(config),
            defaults: Arc::new(defaults.clone()),
            shared_resources: Arc::new(shared_resources),
            cache: Arc::new(Mutex::new(PipelineCache::new(
                cache_capacity,
                defaults,
                default_pipeline,
            ))),
            operation: Arc::new(Mutex::new(())),
            metrics: Arc::new(PipelineMetrics::default()),
        }
    }

    fn key_for(&self, models: &VoiceModelPaths) -> PipelineKey {
        PipelineKey {
            gpt: models.gpt.clone().or_else(|| self.defaults.gpt.clone()),
            sovits: models
                .sovits
                .clone()
                .or_else(|| self.defaults.sovits.clone()),
            bigvgan: models
                .bigvgan
                .clone()
                .or_else(|| self.defaults.bigvgan.clone()),
        }
        .normalized()
    }

    pub(super) async fn acquire_pipeline(
        &self,
        models: &VoiceModelPaths,
        queue_timeout: Duration,
    ) -> Result<PipelineLease, String> {
        // Cover cold loading and inference with one lease so waiting requests cannot
        // retain extra pipelines or start duplicate loads.
        let queued = QueueGuard::new(Arc::clone(&self.metrics));
        let operation =
            tokio::time::timeout(queue_timeout, Arc::clone(&self.operation).lock_owned())
                .await
                .map_err(|_| {
                    format!(
                        "Inference queue timed out after {} second(s)",
                        queue_timeout.as_secs()
                    )
                })?;
        drop(queued);
        let key = self.key_for(models);
        if let Some(pipeline) = self.cache.lock().await.get(&key) {
            self.metrics.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(PipelineLease {
                pipeline,
                _operation: operation,
            });
        }
        self.metrics.cache_misses.fetch_add(1, Ordering::Relaxed);

        let evicted = self.cache.lock().await.evict_before_insert(&key);
        if let Some((evicted_key, pipeline)) = evicted {
            self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
            info!(?evicted_key, "Evicting least recently used model pipeline");
            drop(pipeline);
        }

        let config = (*self.config).clone();
        let shared_resources = Arc::clone(&self.shared_resources);
        let load_key = key.clone();
        let loaded = tokio::task::spawn_blocking(move || {
            load_pipeline(config, &load_key, Some(&shared_resources))
        })
        .await
        .map_err(|e| format!("Pipeline loader task failed: {e}"))??;
        let loaded = Arc::new(Mutex::new(loaded));

        self.cache.lock().await.insert(key, Arc::clone(&loaded));
        Ok(PipelineLease {
            pipeline: loaded,
            _operation: operation,
        })
    }

    pub(super) async fn status(&self) -> PipelineStatus {
        let cache = self.cache.lock().await;
        PipelineStatus {
            cached: cache.len(),
            capacity: cache.capacity,
            cache_hits: self.metrics.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.metrics.cache_misses.load(Ordering::Relaxed),
            evictions: self.metrics.evictions.load(Ordering::Relaxed),
            queued_requests: self.metrics.queued_requests.load(Ordering::Relaxed),
            busy: self.operation.try_lock().is_err(),
        }
    }
}

impl PipelineKey {
    fn normalized(mut self) -> Self {
        self.gpt = normalize_model_path(self.gpt);
        self.sovits = normalize_model_path(self.sovits);
        self.bigvgan = normalize_model_path(self.bigvgan);
        self
    }
}

fn normalize_model_path(path: Option<PathBuf>) -> Option<PathBuf> {
    path.map(|path| path.canonicalize().unwrap_or(path))
}

fn load_pipeline(
    config: Config,
    key: &PipelineKey,
    shared_resources: Option<&SharedPipelineResources>,
) -> Result<Pipeline, String> {
    let mut pipeline = match shared_resources {
        Some(resources) => Pipeline::new_with_shared_resources(config, resources),
        None => Pipeline::new(config),
    }
    .map_err(|e| format!("Failed to initialize pipeline: {e}"))?;

    if let Some(path) = key.gpt.as_deref() {
        info!("Loading GPT model from {:?}", path);
        pipeline
            .load_gpt(path)
            .map_err(|e| format!("Failed to load GPT model: {e}"))?;
    }
    if let Some(path) = key.sovits.as_deref() {
        info!("Loading SoVITS model from {:?}", path);
        pipeline
            .load_sovits(path)
            .map_err(|e| format!("Failed to load SoVITS model: {e}"))?;
    }
    if let Some(path) = key.bigvgan.as_deref() {
        info!("Loading BigVGAN model from {:?}", path);
        pipeline
            .load_bigvgan(path)
            .map_err(|e| format!("Failed to load BigVGAN model: {e}"))?;
    }
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_pipeline() -> Pipeline {
        Pipeline::new(Config::builder().with_device("cpu").build()).unwrap()
    }

    fn key(name: &str) -> PipelineKey {
        PipelineKey {
            gpt: Some(PathBuf::from(format!("{name}-gpt.safetensors"))),
            sovits: Some(PathBuf::from(format!("{name}-sovits.safetensors"))),
            bigvgan: None,
        }
    }

    #[test]
    fn cache_evicts_the_least_recently_used_entry() {
        let first = key("first");
        let second = key("second");
        let third = key("third");
        let mut cache = PipelineCache::new(2, first.clone(), empty_pipeline());
        cache.insert(second.clone(), Arc::new(Mutex::new(empty_pipeline())));
        assert!(cache.get(&first).is_some());

        let (evicted, pipeline) = cache
            .evict_before_insert(&third)
            .expect("full cache should evict one entry");
        drop(pipeline);
        cache.insert(third.clone(), Arc::new(Mutex::new(empty_pipeline())));

        assert_eq!(evicted, second);
        assert!(cache.entries.contains_key(&first));
        assert!(cache.entries.contains_key(&third));
        assert_eq!(cache.entries.len(), 2);
    }

    #[tokio::test]
    async fn leases_serialize_requests_globally() {
        let config = Config::builder().with_device("cpu").build();
        let registry = PipelineRegistry::new(
            config,
            PipelineKey {
                gpt: None,
                sovits: None,
                bigvgan: None,
            },
            empty_pipeline(),
            2,
        );
        let first = registry
            .acquire_pipeline(&VoiceModelPaths::default(), Duration::from_secs(1))
            .await
            .unwrap();
        let active_status = registry.status().await;
        assert!(active_status.busy);
        assert_eq!(active_status.cache_hits, 1);
        let waiting_registry = registry.clone();
        let waiting = tokio::spawn(async move {
            let lease = waiting_registry
                .acquire_pipeline(&VoiceModelPaths::default(), Duration::from_secs(1))
                .await
                .unwrap();
            drop(lease);
        });

        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        assert_eq!(registry.status().await.queued_requests, 1);
        drop(first);
        waiting.await.unwrap();
        let complete_status = registry.status().await;
        assert!(!complete_status.busy);
        assert_eq!(complete_status.cache_hits, 2);
        assert_eq!(complete_status.queued_requests, 0);
    }

    #[tokio::test]
    async fn queue_timeout_does_not_leave_a_waiter_count() {
        let registry = PipelineRegistry::new(
            Config::builder().with_device("cpu").build(),
            PipelineKey {
                gpt: None,
                sovits: None,
                bigvgan: None,
            },
            empty_pipeline(),
            1,
        );
        let active = registry
            .acquire_pipeline(&VoiceModelPaths::default(), Duration::from_secs(1))
            .await
            .unwrap();
        let error = registry
            .acquire_pipeline(&VoiceModelPaths::default(), Duration::from_millis(1))
            .await
            .err()
            .expect("second request should time out");

        assert!(error.contains("Inference queue timed out"));
        assert_eq!(registry.status().await.queued_requests, 0);
        drop(active);
    }
}
