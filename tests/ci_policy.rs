#![cfg(unix)]

use std::io::Write;
use std::process::{Command, Stdio};

fn container_needed(paths: &[&str]) -> bool {
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/container-needed.sh");
    let mut child = Command::new("bash")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let input: Vec<u8> = paths
        .iter()
        .flat_map(|path| path.bytes().chain([0]))
        .collect();
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    match output.stdout.as_slice() {
        b"true\n" => true,
        b"false\n" => false,
        other => panic!("unexpected policy result: {other:?}"),
    }
}

#[test]
fn source_docs_and_client_changes_use_packaged_binary_acceptance() {
    assert!(!container_needed(&[]));
    assert!(!container_needed(&[
        "src/inference/options.rs",
        "src/models/gpt.rs",
        "src/server/handlers.rs",
        "src/voice.rs",
        "README.md",
        "docs/API.md",
        "client/Cargo.toml",
        "tests/integration_tests.rs",
        "src/a name\nwith newline.rs",
    ]));
}

#[test]
fn build_runtime_and_startup_changes_keep_full_container_acceptance() {
    for path in [
        "Dockerfile",
        "Dockerfile.cuda",
        ".dockerignore",
        "compose.cpu.yml",
        "compose.cuda.yml",
        ".env.example",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "src/cli.rs",
        "src/main.rs",
        "src/mkl_compat.rs",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
        ".github/container-needed.sh",
        "examples/first-run/prepare.sh",
        "examples/first-run/downloads.txt",
        "examples/verify-http-formats.sh",
        "tests/deployment_smoke.rs",
        "tests/first_run.rs",
        "LICENSE",
        "THIRD_PARTY.md",
        "licenses/CMUDICT.txt",
    ] {
        assert!(container_needed(&["src/models/gpt.rs", path]), "{path}");
    }
}
