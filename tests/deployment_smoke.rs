//! Model-free Compose/image compatibility check; real inference is tested separately.
//!
//! GPT_SOVITS_SMOKE_IMAGE=<cpu-image> cargo test --test deployment_smoke -- --ignored
//! For a CUDA image, also set GPT_SOVITS_SMOKE_GPUS=all on a GPU host.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

#[derive(Deserialize)]
struct Compose {
    services: HashMap<String, Service>,
}

#[derive(Deserialize)]
struct Service {
    command: Vec<String>,
    environment: HashMap<String, String>,
    ports: Vec<Port>,
    stop_grace_period: String,
}

#[derive(Deserialize)]
struct Port {
    host_ip: Option<String>,
}

fn compose_defaults(file: &str, env_file: &Path) -> Service {
    let output = Command::new("docker")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env_remove("GPT_SOVITS_PORT")
        .args(["compose", "--env-file"])
        .arg(env_file)
        .args(["-f", file, "config", "--format", "json"])
        .output()
        .expect("Docker Compose must be installed");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut config: Compose = serde_json::from_slice(&output.stdout).unwrap();
    let service = config.services.remove("gpt-sovits").unwrap();
    assert!(
        matches!(service.ports.as_slice(), [p] if p.host_ip.as_deref() == Some("127.0.0.1")),
        "{file} with {} should publish only on loopback",
        env_file.display()
    );
    service
}

#[test]
#[ignore = "requires Docker Compose and an explicitly selected local image"]
fn compose_commands_are_accepted_by_container_image() {
    let image = std::env::var("GPT_SOVITS_SMOKE_IMAGE")
        .expect("set GPT_SOVITS_SMOKE_IMAGE to the image under test");
    let env_file = tempfile::NamedTempFile::new().unwrap();
    for file in ["compose.cpu.yml", "compose.cuda.yml"] {
        compose_defaults(file, Path::new(".env.example"));
        let service = compose_defaults(file, env_file.path());
        assert_eq!(service.environment["GPT_SOVITS_HOST"], "0.0.0.0");
        assert_eq!(service.stop_grace_period, "45s");

        let mut command = Command::new("docker");
        command.args(["run", "--rm", "--pull", "never", "--network", "none"]);
        if let Ok(gpus) = std::env::var("GPT_SOVITS_SMOKE_GPUS") {
            command.args(["--gpus", &gpus]);
        }
        for (key, value) in &service.environment {
            command.args(["--env", &format!("{key}={value}")]);
        }
        // No volume mounts: startup must reach the missing-model error, not fail
        // on an unsupported argument or a missing runtime shared library.
        let output = command.arg(&image).args(&service.command).output().unwrap();
        let diagnostics = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.status.code(), Some(1), "{file}: {diagnostics}");
        assert!(
            diagnostics.contains("GPT model file does not exist:"),
            "{file}: {diagnostics}"
        );
        println!("{file}: {image} accepted the Compose startup arguments");
    }
}
