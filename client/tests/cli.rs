use std::io::{Cursor, Write};
use std::process::{Command, Output, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};
use tiny_http::{Header, Response, Server, StatusCode};

fn cli() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gpt-sovits-client"));
    command.stdin(Stdio::null());
    command
}

fn diagnostics(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn wav(sample_rate: u32, samples: usize) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..samples {
            writer.write_sample(123_i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    bytes.into_inner()
}

fn serve(
    status: u16,
    mime: &str,
    body: Vec<u8>,
    delay: Duration,
) -> (String, JoinHandle<(String, Value)>) {
    let server = Server::http("127.0.0.1:0").unwrap();
    let url = format!("http://{}", server.server_addr());
    let mime = Header::from_bytes("Content-Type", mime).unwrap();
    let thread = thread::spawn(move || {
        let mut request = server
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .expect("client did not send a request");
        assert_eq!(request.method().as_str(), "POST");
        let path = request.url().to_owned();
        let payload = serde_json::from_reader(request.as_reader()).unwrap();
        thread::sleep(delay);
        let response = Response::from_data(body)
            .with_status_code(StatusCode(status))
            .with_header(mime)
            .with_header(Header::from_bytes("Location", "/redirected").unwrap());
        // A timeout intentionally closes the client before this response is sent.
        let _ = request.respond(response);
        (path, payload)
    });
    (url, thread)
}

#[test]
fn help_version_and_missing_values() {
    for arg in ["--help", "--version"] {
        let output = cli().arg(arg).output().unwrap();
        assert!(output.status.success(), "{}", diagnostics(&output));
    }
    for flag in ["--url", "--voice", "--text", "--output", "--timeout"] {
        let output = cli().arg(flag).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{flag}");
        assert!(!diagnostics(&output).contains("panicked"));
    }
}

#[test]
fn rejects_invalid_arguments_and_empty_text() {
    let cases = [
        vec!["--text", ""],
        vec!["--text", " \n\t"],
        vec!["--timeout", "0"],
        vec!["--timeout", "oops"],
        vec!["--url", "file:///tmp/audio"],
        vec!["--url", "http://localhost/?token=secret"],
        vec!["--url", "http://localhost/#fragment"],
        vec!["--url", "http://user:pass@localhost"],
        vec!["--unknown"],
        vec![],
    ];
    for args in cases {
        let output = cli()
            .args(["--voice", "demo"])
            .args(&args)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{args:?}");
        assert!(output.stdout.is_empty(), "{args:?}");
        assert!(!diagnostics(&output).contains("panicked"));
    }
    let output = cli().args(["--text", "hello"]).output().unwrap();
    assert_eq!(output.status.code(), Some(2), "voice must be explicit");
    let output = cli()
        .args(["--voice", " ", "--text", "hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(diagnostics(&output).contains("--voice must not be empty"));
}

#[test]
fn sends_only_text_and_voice_and_writes_clean_stdout() {
    let audio = wav(32_000, 100);
    let (url, server) = serve(200, "audio/wav", audio.clone(), Duration::ZERO);
    let text = "  \u{91cd}[chong2]\u{65b0}\n\"quoted\" $HOME $(false) `false` ";
    let output = cli()
        .args([
            "--url",
            &format!("{url}/prefix/"),
            "--voice",
            "demo",
            "--text",
            text,
        ])
        .output()
        .unwrap();
    let (path, payload) = server.join().unwrap();
    assert!(output.status.success(), "{}", diagnostics(&output));
    assert!(output.stderr.is_empty(), "{}", diagnostics(&output));
    assert_eq!(output.stdout, audio);
    assert_eq!(path, "/prefix/tts");
    assert_eq!(payload, json!({"voice": "demo", "text": text.trim()}));
}

#[test]
fn reads_stdin_and_replaces_output_without_models() {
    let dir = tempfile::tempdir().unwrap();
    let audio = wav(32_000, 100);
    let (url, server) = serve(
        200,
        "audio/wav; charset=binary",
        audio.clone(),
        Duration::ZERO,
    );
    let path = dir.path().join("reply with spaces.wav");
    std::fs::write(&path, b"old audio").unwrap();
    let mut child = cli()
        .current_dir(dir.path())
        .args(["--url", &url, "--voice", "demo", "-o"])
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hello\nworld")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let (_, payload) = server.join().unwrap();
    assert!(output.status.success(), "{}", diagnostics(&output));
    assert!(output.stdout.is_empty());
    assert_eq!(std::fs::read(path).unwrap(), audio);
    assert_eq!(payload["text"], "hello\nworld");
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[test]
fn http_errors_preserve_existing_output() {
    for status in [400, 404, 503] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reply.wav");
        std::fs::write(&path, b"keep me").unwrap();
        let (url, server) = serve(
            status,
            "application/json",
            br#"{"error":"test failure"}"#.to_vec(),
            Duration::ZERO,
        );
        let output = cli()
            .args(["--url", &url, "--voice", "demo", "--text", "hello", "-o"])
            .arg(&path)
            .output()
            .unwrap();
        server.join().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(diagnostics(&output).contains(&format!("HTTP {status}")));
        assert!(diagnostics(&output).contains("test failure"));
        assert_eq!(std::fs::read(path).unwrap(), b"keep me");
    }
}

#[test]
fn invalid_responses_leave_no_output_file() {
    let mut truncated = wav(32_000, 100);
    truncated.truncate(truncated.len() - 2);
    for (mime, body) in [
        ("text/html", b"<html>proxy error</html>".to_vec()),
        ("audio/wav", b"not a WAV".to_vec()),
        ("audio/wav", Vec::new()),
        ("audio/wav", wav(32_000, 0)),
        ("audio/wav", wav(16_000, 100)),
        ("audio/wav", truncated),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reply.wav");
        let (url, server) = serve(200, mime, body, Duration::ZERO);
        let output = cli()
            .args(["--url", &url, "--voice", "demo", "--text", "hello", "-o"])
            .arg(&path)
            .output()
            .unwrap();
        server.join().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}

#[test]
fn redirects_are_reported_without_forwarding_text() {
    let (url, server) = serve(307, "text/plain", Vec::new(), Duration::ZERO);
    let output = cli()
        .args(["--url", &url, "--voice", "demo", "--text", "hello"])
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(!output.status.success());
    assert!(diagnostics(&output).contains("HTTP 307"));
    assert!(output.stdout.is_empty());
}

#[test]
fn timeout_does_not_create_audio() {
    let (url, server) = serve(200, "audio/wav", wav(32_000, 100), Duration::from_secs(2));
    let output = cli()
        .args([
            "--url",
            &url,
            "--voice",
            "demo",
            "--text",
            "hello",
            "--timeout",
            "1",
        ])
        .output()
        .unwrap();
    server.join().unwrap();
    assert!(!output.status.success());
    assert!(diagnostics(&output).contains("TTS request failed"));
    assert!(output.stdout.is_empty());
}

#[cfg(unix)]
#[test]
fn command_provider_can_redirect_input_and_output_paths_with_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("input text.txt");
    let output_path = dir.path().join("output audio.wav");
    let text = "quotes \" ' $HOME $(false)\nsecond line";
    std::fs::write(&input, text).unwrap();
    let audio = wav(32_000, 100);
    let (url, server) = serve(200, "audio/wav", audio.clone(), Duration::ZERO);
    let output = Command::new("sh")
        .args([
            "-c",
            "\"$1\" --url \"$2\" --voice demo -o \"$3\" < \"$4\"",
            "tts",
        ])
        .arg(env!("CARGO_BIN_EXE_gpt-sovits-client"))
        .arg(&url)
        .arg(&output_path)
        .arg(&input)
        .output()
        .unwrap();
    let (_, payload) = server.join().unwrap();
    assert!(output.status.success(), "{}", diagnostics(&output));
    assert!(output.stdout.is_empty());
    assert_eq!(payload["text"], text);
    assert_eq!(std::fs::read(output_path).unwrap(), audio);
}
