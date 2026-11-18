#![cfg(unix)]

use std::io::Write;
use std::process::{Command, Stdio};

fn check_scope(paths: &[&str], quality: bool, client: bool, container: bool) {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/ci-scope.sh");
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
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("quality={quality}\nclient={client}\ncontainer={container}\n"),
        "{paths:?}"
    );
}

#[test]
fn site_and_client_changes_do_not_run_native_jobs() {
    check_scope(&[], false, false, false);
    check_scope(
        &[
            "site/check.cjs",
            "assets/pages-voice-console.webp",
            "examples/samples/README.md",
            "experiments/wasm-sovits/web/index.html",
            ".github/workflows/pages.yml",
        ],
        false,
        false,
        false,
    );
    check_scope(&["site/check.cjs", "client/Cargo.toml"], false, true, false);
}

#[test]
fn native_and_mixed_changes_keep_linux_quality_checks() {
    for path in [
        "src/models/gpt.rs",
        "docs/API.md",
        "tests/first_run.rs",
        "src/a name\nwith newline.rs",
    ] {
        check_scope(&["site/index.html", path], true, false, false);
    }
    check_scope(&["client/Cargo.toml", "src/server.rs"], true, true, false);
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
        ".github/workflows/release.yml",
        "examples/first-run/prepare.sh",
        "examples/first-run/downloads.txt",
        "examples/verify-http-formats.sh",
        "tests/deployment_smoke.rs",
        "LICENSE",
        "THIRD_PARTY.md",
        "licenses/CMUDICT.txt",
    ] {
        check_scope(&["site/check.cjs", path], true, false, true);
    }
}

#[test]
fn ci_routing_changes_do_not_rebuild_the_runtime_image() {
    check_scope(
        &[
            "site/index.html",
            ".github/workflows/ci.yml",
            ".github/ci-scope.sh",
        ],
        true,
        true,
        false,
    );
}
