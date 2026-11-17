use std::future::Future;
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::info;

pub(super) const GRACE_PERIOD: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub(super) struct Shutdown(watch::Sender<Option<Instant>>);

impl Default for Shutdown {
    fn default() -> Self {
        Self(watch::channel(None).0)
    }
}

impl Shutdown {
    pub(super) fn begin(&self, grace: Duration) {
        self.0.send_if_modified(|deadline| {
            if deadline.is_some() {
                return false;
            }
            *deadline = Some(Instant::now() + grace);
            true
        });
    }

    pub(super) fn is_stopping(&self) -> bool {
        self.0.borrow().is_some()
    }

    pub(super) async fn cancelled(&self) {
        let mut receiver = self.0.subscribe();
        let _ = receiver.wait_for(Option::is_some).await;
    }

    fn remaining(&self) -> Option<Duration> {
        self.0
            .borrow()
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }
}

#[cfg(unix)]
fn shutdown_signal() -> io::Result<impl Future<Output = io::Result<()>>> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    Ok(async move {
        tokio::select! {
            _ = terminate.recv() => {},
            _ = interrupt.recv() => {},
        }
        Ok(())
    })
}

#[cfg(not(unix))]
fn shutdown_signal() -> io::Result<impl Future<Output = io::Result<()>>> {
    Ok(tokio::signal::ctrl_c())
}

pub(super) fn run(
    shutdown: Shutdown,
    grace: Duration,
    work: impl Future<Output = Result<(), String>>,
) -> Result<(), String> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|error| format!("Failed to create Tokio runtime: {error}"))?;
    let result = runtime.block_on(async {
        // Install handlers before polling startup, including blocking model loading.
        let signal = shutdown_signal()
            .map_err(|error| format!("Failed to install shutdown signals: {error}"))?;
        tokio::pin!(work);
        tokio::select! {
            biased;
            signal = signal => {
                signal.map_err(|error| format!("Shutdown signal failed: {error}"))?;
                shutdown.begin(grace);
                info!(grace_seconds = grace.as_secs_f64(), "Stopping HTTP service");
                tokio::time::timeout(grace, work)
                    .await
                    .map_err(|_| format!("Shutdown timed out after {:.3} seconds", grace.as_secs_f64()))?
            }
            result = &mut work => result,
        }
    });
    // Runtime drop otherwise waits indefinitely for detached blocking work.
    runtime.shutdown_timeout(shutdown.remaining().unwrap_or(grace));
    if result.is_ok()
        && shutdown
            .remaining()
            .is_some_and(|remaining| remaining.is_zero())
    {
        return Err(format!(
            "Shutdown timed out after {:.3} seconds",
            grace.as_secs_f64()
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_deadline_leaves_time_before_compose_force_kill() {
        assert!(GRACE_PERIOD < Duration::from_secs(45));
    }

    #[tokio::test]
    async fn shutdown_is_observed_by_existing_and_late_subscribers() {
        let shutdown = Shutdown::default();
        assert!(!shutdown.is_stopping());
        let waiting = shutdown.clone();
        let task = tokio::spawn(async move { waiting.cancelled().await });
        shutdown.begin(Duration::from_secs(1));
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), shutdown.cancelled())
            .await
            .unwrap();
        let deadline = *shutdown.0.borrow();
        shutdown.begin(Duration::from_secs(30));
        assert_eq!(*shutdown.0.borrow(), deadline);
        assert!(shutdown.is_stopping());
    }
}
