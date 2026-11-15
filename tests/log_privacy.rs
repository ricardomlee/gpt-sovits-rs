use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::process::Command;

const PRIVATE_TEXT: &str = "PRIVATE_LOG_MARKER hello world.";

#[test]
fn streaming_text_logging_requires_explicit_opt_in() {
    for enabled in [false, true] {
        let mut child = Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "streaming_log_probe", "--nocapture"])
            .env("GPT_SOVITS_LOG_PROBE", "1")
            .env_remove("GPT_SOVITS_LOG_CHUNK_TEXT");
        if enabled {
            child.env("GPT_SOVITS_LOG_CHUNK_TEXT", "1");
        }
        let output = child.output().unwrap();
        let logs = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{logs}");
        assert_eq!(logs.contains(PRIVATE_TEXT), enabled, "{logs}");
        assert_eq!(logs.contains("PRIVATE_LOG_MARKER"), enabled, "{logs}");
    }
}

#[test]
fn streaming_log_probe() {
    if std::env::var("GPT_SOVITS_LOG_PROBE").as_deref() != Ok("1") {
        return;
    }
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_ansi(false)
        .without_time()
        .init();
    let mut pipeline = Pipeline::new(Config::builder().with_device("cpu").build()).unwrap();
    let options = InferenceOptions {
        language: Language::English,
        ..Default::default()
    };
    let mut stream =
        pipeline.inference_sentences(PRIVATE_TEXT, "missing.wav", "", &options, "kv", 1);
    assert!(stream.next().unwrap().is_err());
}
