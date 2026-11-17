use super::{shutdown, state::AppState};
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::Response,
    routing::{get, post},
    Router,
};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

const CHILD_ENV: &str = "GPT_SOVITS_TEST_SERVER_CHILD";
const GATE_ENV: &str = "GPT_SOVITS_TEST_SERVER_GATE";

fn wait_for_release() {
    let path = std::path::PathBuf::from(std::env::var_os(GATE_ENV).unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(Instant::now() < deadline, "test worker was not released");
        std::thread::sleep(Duration::from_millis(5));
    }
}

async fn test_work(State(state): State<AppState>) -> Result<Response<Body>, StatusCode> {
    let lease = state
        .pipelines
        .acquire_pipeline(&Default::default(), state.queue_timeout)
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    let mode = std::env::var(CHILD_ENV).unwrap();
    if mode == "stream" || mode == "panic" {
        let receiver = lease.spawn_stream(1, move |_, tx| {
            tx.blocking_send(Ok(b"first\n".to_vec())).unwrap();
            wait_for_release();
            assert_ne!(mode, "panic", "worker failure probe");
            tx.blocking_send(Ok(b"last\n".to_vec())).unwrap();
        });
        let stream = ReceiverStream::new(receiver).map(|item| item.map_err(std::io::Error::other));
        Ok(Response::new(Body::from_stream(stream)))
    } else {
        lease
            .spawn(|_| wait_for_release())
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Ok(Response::new(Body::from("work complete")))
    }
}

#[test]
fn child_server() {
    let Ok(mode) = std::env::var(CHILD_ENV) else {
        return;
    };
    if mode != "idle" {
        let state = AppState::for_test();
        let pipelines = state.pipelines.clone();
        let shutdown = pipelines.shutdown.clone();
        let listener = super::bind_listener(Ipv4Addr::LOCALHOST.into(), 0).unwrap();
        let addr = listener.local_addr().unwrap();
        let grace = if matches!(mode.as_str(), "deadline" | "startup" | "background") {
            Duration::from_millis(200)
        } else {
            Duration::from_secs(3)
        };
        let result = shutdown::run(shutdown, grace, async move {
            super::print_server_ready(addr, 1);
            if mode == "startup" {
                tokio::task::spawn_blocking(wait_for_release).await.unwrap();
                return Ok(());
            }
            if mode == "background" {
                tokio::task::spawn_blocking(wait_for_release);
            }
            let app = Router::new()
                .route("/health", get(super::health_handler))
                .route("/status", get(super::status_handler))
                .route("/work", post(test_work))
                .with_state(state);
            super::serve(
                tokio::net::TcpListener::from_std(listener).unwrap(),
                app,
                pipelines,
            )
            .await
        });
        if let Err(error) = result {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    super::run(
        Ipv4Addr::LOCALHOST.into(),
        0,
        "cpu",
        false,
        None,
        None,
        None,
        None,
        None,
        1,
        false,
        100,
        4,
        1,
        &[],
        std::path::Path::new("."),
        std::path::Path::new("."),
    )
    .unwrap();
}

struct ServerProcess {
    child: Child,
    addr: SocketAddr,
    gate: tempfile::TempDir,
}

impl ServerProcess {
    fn start(mode: &str) -> Self {
        let gate = tempfile::tempdir().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "server::process_tests::child_server",
                "--nocapture",
            ])
            .env(CHILD_ENV, mode)
            .env(GATE_ENV, gate.path().join("release"))
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Some(addr) = line.strip_prefix("HTTP server started at http://") {
                    let _ = tx.send(addr.parse::<SocketAddr>().unwrap());
                }
            }
        });
        let addr = match rx.recv_timeout(Duration::from_secs(20)) {
            Ok(addr) => addr,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("server did not become ready: {error}");
            }
        };
        Self { child, addr, gate }
    }

    fn request(&self, request: &[u8]) -> TcpStream {
        let mut stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(5)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream.write_all(request).unwrap();
        stream
    }

    fn health(&self) {
        let mut stream =
            self.request(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    }

    fn work(&self) -> TcpStream {
        self.request(b"POST /work HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    }

    fn wait_for_work(&self, queued: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let mut stream = self
                .request(b"GET /status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
            let mut response = String::new();
            stream.read_to_string(&mut response).unwrap();
            let body: serde_json::Value =
                serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
            if body["busy"] == true && body["queued_requests"] == queued {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "work did not enter the expected state: {body}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    fn release(&self) {
        std::fs::write(self.gate.path().join("release"), b"ready").unwrap();
    }

    fn signal(&self, signal: &str) {
        assert!(Command::new("kill")
            .args([signal, &self.child.id().to_string()])
            .status()
            .unwrap()
            .success());
    }

    fn wait(&mut self) -> ExitStatus {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(
                Instant::now() < deadline,
                "server did not exit within five seconds"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn normal_signals_stop_server_cleanly() {
    for signal in ["-TERM", "-INT"] {
        let mut server = ServerProcess::start("idle");
        server.health();
        server.signal(signal);
        let status = server.wait();
        assert!(status.success(), "{signal}: {status}");
    }
}

fn read_first_chunk(stream: &mut TcpStream) -> String {
    let mut response = Vec::new();
    let mut byte = [0];
    while !response.ends_with(b"first\n") {
        stream.read_exact(&mut byte).unwrap();
        response.push(byte[0]);
    }
    String::from_utf8(response).unwrap()
}

#[test]
fn shutdown_drains_buffered_streaming_and_disconnected_work() {
    for mode in ["buffered", "stream", "disconnected"] {
        let mut server = ServerProcess::start(mode);
        let mut request = server.work();
        let prefix = if mode == "stream" {
            read_first_chunk(&mut request)
        } else {
            String::new()
        };
        server.wait_for_work(0);
        let mut request = if mode == "disconnected" {
            drop(request);
            None
        } else {
            Some(request)
        };
        server.signal("-TERM");
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            server.child.try_wait().unwrap().is_none(),
            "{mode}: active work was abandoned"
        );
        server.release();
        if let Some(request) = request.as_mut() {
            let mut response = prefix;
            request.read_to_string(&mut response).unwrap();
            assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
            if mode == "stream" {
                assert!(
                    response.contains("last\n") && response.ends_with("0\r\n\r\n"),
                    "{response}"
                );
            } else {
                assert!(response.ends_with("work complete"), "{response}");
            }
        }
        assert!(server.wait().success(), "{mode}");
    }
}

#[test]
fn shutdown_rejects_queued_http_work() {
    let mut server = ServerProcess::start("queued");
    let mut active = server.work();
    server.wait_for_work(0);
    let mut queued = server.work();
    server.wait_for_work(1);
    server.signal("-TERM");
    let mut response = String::new();
    queued.read_to_string(&mut response).unwrap();
    assert!(
        response.starts_with("HTTP/1.1 503 Service Unavailable"),
        "{response}"
    );
    server.release();
    response.clear();
    active.read_to_string(&mut response).unwrap();
    assert!(response.ends_with("work complete"), "{response}");
    assert!(server.wait().success());
}

#[test]
fn shutdown_deadline_bounds_startup_and_detached_blocking_work() {
    for mode in ["startup", "deadline", "background"] {
        let mut server = ServerProcess::start(mode);
        if mode == "background" {
            server.health();
        }
        let request = if mode == "deadline" {
            let request = server.work();
            server.wait_for_work(0);
            Some(request)
        } else {
            None
        };
        drop(request);
        server.signal("-TERM");
        assert_eq!(server.wait().code(), Some(1), "{mode}");
    }
}

#[test]
fn streaming_worker_panic_is_not_a_successful_http_eof() {
    let mut server = ServerProcess::start("panic");
    let mut request = server.work();
    let mut response = read_first_chunk(&mut request);
    server.release();
    let _ = request.read_to_string(&mut response);
    assert!(
        !response.ends_with("0\r\n\r\n"),
        "panic was reported as successful EOF: {response}"
    );
    server.health();
    server.signal("-TERM");
    assert!(server.wait().success());
}
