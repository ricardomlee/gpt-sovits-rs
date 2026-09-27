use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::process::{Command, Output};

fn save_header(path: &Path, name: &str) {
    let weights = HashMap::from([(name, Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap())]);
    candle_core::safetensors::save(&weights, path).unwrap();
}

fn setup(pro: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    save_header(&dir.path().join("gpt-model.safetensors"), "weight");
    save_header(
        &dir.path().join("sovits-model.safetensors"),
        if pro { "sv_emb.weight" } else { "weight" },
    );
    dir
}

fn doctor(dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gpt-sovits"));
    command
        .env_remove("GPT_SOVITS_SV_MODEL")
        .env_remove("GPT_SOVITS_HOST")
        .args(["--doctor", "--device", "cpu", "--models-dir"])
        .arg(dir)
        .arg("--voices-dir")
        .arg(dir.join("voices"));
    command
}

fn diagnostics(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn doctor_requires_sv_only_for_v2pro() {
    for pro in [false, true] {
        let dir = setup(pro);
        let output = doctor(dir.path()).output().unwrap();
        let text = diagnostics(&output);
        assert_eq!(text.contains("v2Pro needs an SV encoder"), pro, "{text}");
        if pro {
            assert!(!output.status.success());
            assert!(text.contains("gpt-sovits-convert sv-model"));
        }
    }
}

#[test]
fn empty_compose_sv_environment_keeps_automatic_discovery() {
    let dir = setup(true);
    save_header(&dir.path().join("sv.safetensors"), "weight");
    let output = doctor(dir.path())
        .env("GPT_SOVITS_SV_MODEL", "")
        .output()
        .unwrap();
    let text = diagnostics(&output);
    assert!(text.contains("[ok] SV encoder"), "{text}");
    assert!(!text.contains("v2Pro needs an SV encoder"), "{text}");
}

#[test]
fn explicit_embedding_does_not_require_encoder_weights() {
    let dir = setup(true);
    let embedding = dir.path().join("ref_sv.safetensors");
    save_header(&embedding, "sv_embedding");
    let output = doctor(dir.path())
        .arg("--sv-embedding")
        .arg(embedding)
        .output()
        .unwrap();
    let text = diagnostics(&output);
    assert!(text.contains("[ok] Selected SV embedding"), "{text}");
    assert!(!text.contains("v2Pro needs an SV encoder"), "{text}");
}
