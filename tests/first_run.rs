//! First-run fixtures and preparation safety; no network or Docker in normal tests.

use std::path::Path;

fn check_wav(path: &Path) {
    let mut reader = hound::WavReader::open(path).expect("a complete WAV, not an HTTP error body");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate, 32_000);
    assert_eq!(spec.bits_per_sample, 16);
    let samples: Vec<f64> = reader
        .samples::<i16>()
        .map(|s| f64::from(s.unwrap()) / 32768.0)
        .collect();
    let seconds = samples.len() as f64 / f64::from(spec.sample_rate);
    assert!((2.0..30.0).contains(&seconds), "duration: {seconds}");
    let rms = (samples.iter().map(|v| v * v).sum::<f64>() / samples.len() as f64).sqrt();
    assert!(rms > 0.001, "silent output: RMS {rms}");
    let clipped = samples.iter().filter(|v| v.abs() >= 0.999).count();
    assert!(clipped as f64 / (samples.len() as f64) < 0.01);
    println!("{}: {seconds:.2}s, RMS {rms:.4}", path.display());
}

#[test]
fn public_sample_is_real_non_silent_audio() {
    for name in [
        "local-assistant-en.wav",
        "local-assistant-zh.wav",
        "sun-greeting.wav",
        "sun-zh.wav",
    ] {
        check_wav(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("examples/samples")
                .join(name),
        );
    }
}

#[test]
#[ignore = "requires a completed HTTP request; set FIRST_RUN_WAV to the response path"]
fn generated_http_audio_is_valid() {
    check_wav(Path::new(&std::env::var("FIRST_RUN_WAV").unwrap()));
}

#[test]
fn demo_fixtures_match_the_voice_loader() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("demo")).unwrap();
    std::fs::write(
        temp.path().join("demo/voice.json"),
        include_str!("../examples/first-run/voice.json"),
    )
    .unwrap();
    let voice = gpt_sovits_rs::voice::LoadedVoiceProfile::load("demo", temp.path()).unwrap();
    assert_eq!(voice.profile.language.as_deref(), Some("en"));
    assert_eq!(voice.profile.top_k, Some(1));
    assert_eq!(voice.profile.reference_audio.as_deref(), Some("ref.wav"));
    let request: serde_json::Value =
        serde_json::from_str(include_str!("../examples/first-run/request.json")).unwrap();
    assert_eq!(request["voice"], "demo");
    assert!(!request["text"].as_str().unwrap().is_empty());
    let rows: Vec<_> = include_str!("../examples/first-run/downloads.txt")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect();
    assert_eq!(rows.len(), 6);
    for row in rows {
        assert_eq!(row.len(), 3);
        assert_eq!(row[0].len(), 64);
        assert!(row[0].bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(Path::new(row[1])
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_))));
        assert!(row[2].starts_with("https://"));
        assert!(!row[2].contains("/main/"));
    }
}

#[cfg(unix)]
mod preparation {
    use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, process::Command};

    struct Fixture {
        root: tempfile::TempDir,
        destination: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            let bin = root.path().join("bin");
            fs::create_dir(&bin).unwrap();
            let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/first-run");
            for file in ["prepare.sh", "voice.json", "compose.yml", "request.json"] {
                fs::copy(source.join(file), root.path().join(file)).unwrap();
            }
            let rows = fs::read_to_string(source.join("downloads.txt")).unwrap();
            let manifest = rows
                .lines()
                .filter(|s| !s.starts_with('#') && !s.is_empty())
                .map(|s| {
                    let (_, rest) = s.split_once(' ').unwrap();
                    format!(
                        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad {rest}\n"
                    )
                })
                .collect::<String>();
            fs::write(root.path().join("downloads.txt"), manifest).unwrap();
            let docker = r#"#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$TEST_LOG"
if [[ "$1" == info ]]; then printf '%s\n' "${TEST_PLATFORM:-linux/x86_64}"; exit 0; fi
[[ "$1" != run ]] && exit 0
models=''
previous=''
for arg in "$@"; do
  if [[ "$previous" == --volume && "$arg" == *:/models ]]; then models=${arg%:/models}; fi
  previous=$arg
done
if [[ -n "$models" ]]; then
  [[ ${FAIL_CONVERSION:-0} == 1 ]] && exit 9
  output=${!#}
  printf converted > "$models/${output#/models/}"
fi
exit 0
"#;
            let curl = r#"#!/usr/bin/env bash
set -eu
printf 'download\n' >> "$TEST_LOG"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --output ]]; then printf '%s' "${DOWNLOAD_CONTENT:-abc}" > "$2"; exit 0; fi
  shift
done
exit 2
"#;
            for (name, content) in [("docker", docker), ("curl", curl)] {
                let path = bin.join(name);
                fs::write(&path, content).unwrap();
                fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
            }
            let destination = root.path().join("demo with spaces");
            Self { root, destination }
        }

        fn command(&self) -> Command {
            let mut cmd = Command::new("bash");
            cmd.arg(self.root.path().join("prepare.sh"))
                .arg(&self.destination)
                .env("TEST_LOG", self.root.path().join("commands.log"))
                .env_remove("FAIL_CONVERSION")
                .env_remove("DOWNLOAD_CONTENT")
                .env_remove("TEST_PLATFORM")
                .env(
                    "PATH",
                    format!(
                        "{}:{}",
                        self.root.path().join("bin").display(),
                        std::env::var("PATH").unwrap()
                    ),
                );
            cmd
        }

        fn log(&self) -> String {
            fs::read_to_string(self.root.path().join("commands.log")).unwrap()
        }
    }

    #[test]
    fn prepares_all_models_and_reuses_verified_downloads() {
        let f = Fixture::new();
        for _ in 0..2 {
            let result = f.command().output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(!f.destination.join(".prepare-lock").exists());
        }
        assert_eq!(f.log().lines().filter(|s| *s == "download").count(), 6);
        assert_eq!(
            f.log().matches("--entrypoint gpt-sovits-convert").count(),
            8
        );
        assert_eq!(f.log().matches("--doctor").count(), 2);
        for path in [
            "models/gpt-model.safetensors",
            "models/sovits-model.safetensors",
            "models/bert/bert.safetensors",
            "models/bert/tokenizer.json",
            "models/hubert/hubert.safetensors",
            "voices/demo/ref.wav",
            "voices/demo/voice.json",
            "compose.yml",
            "request.json",
        ] {
            assert!(f.destination.join(path).is_file(), "{path}");
        }
    }

    #[test]
    fn refuses_an_existing_deployment() {
        let f = Fixture::new();
        fs::create_dir(&f.destination).unwrap();
        fs::write(f.destination.join("voice.json"), "private").unwrap();
        assert!(!f.command().output().unwrap().status.success());
        assert!(!f.log().contains("download"));
        assert_eq!(
            fs::read_to_string(f.destination.join("voice.json")).unwrap(),
            "private"
        );
    }

    #[test]
    fn checksum_failure_never_reaches_conversion() {
        let f = Fixture::new();
        assert!(!f
            .command()
            .env("DOWNLOAD_CONTENT", "corrupt")
            .output()
            .unwrap()
            .status
            .success());
        assert!(!f.log().contains("--entrypoint"));
        assert!(!f.destination.join(".prepare-lock").exists());
        assert!(f.command().output().unwrap().status.success());
    }

    #[test]
    fn conversion_failure_can_be_retried() {
        let f = Fixture::new();
        assert!(!f
            .command()
            .env("FAIL_CONVERSION", "1")
            .output()
            .unwrap()
            .status
            .success());
        assert!(!f.destination.join("models/gpt-model.safetensors").exists());
        assert!(!f.destination.join(".prepare-lock").exists());
        assert!(f.command().output().unwrap().status.success());
    }

    #[test]
    fn corrupted_cached_file_is_not_reused_or_silently_replaced() {
        let f = Fixture::new();
        assert!(f.command().output().unwrap().status.success());
        let reference = f.destination.join("source/reference.wav");
        fs::write(&reference, "changed").unwrap();
        let before = f.log().matches("--entrypoint").count();
        assert!(!f.command().output().unwrap().status.success());
        assert_eq!(f.log().matches("--entrypoint").count(), before);
        assert_eq!(fs::read_to_string(reference).unwrap(), "changed");
    }

    #[test]
    fn refuses_a_second_preparation_and_preserves_its_lock() {
        let f = Fixture::new();
        fs::create_dir_all(f.destination.join(".prepare-lock")).unwrap();
        fs::write(
            f.destination.join(".gpt-sovits-demo"),
            "gpt-sovits-public-v2-demo\n",
        )
        .unwrap();
        assert!(!f.command().output().unwrap().status.success());
        assert!(f.destination.join(".prepare-lock").is_dir());
        assert!(!f.log().contains("download"));
    }

    #[test]
    fn refuses_unknown_marker() {
        let f = Fixture::new();
        fs::create_dir_all(&f.destination).unwrap();
        fs::write(f.destination.join(".gpt-sovits-demo"), "other project").unwrap();
        assert!(!f.command().output().unwrap().status.success());
        assert!(!f.log().contains("download"));
    }

    #[test]
    fn unsupported_platform_fails_before_downloads() {
        let f = Fixture::new();
        assert!(!f
            .command()
            .env("TEST_PLATFORM", "linux/aarch64")
            .output()
            .unwrap()
            .status
            .success());
        assert!(!f.destination.exists());
        assert!(!f.log().contains("download"));
    }
}

#[cfg(unix)]
#[test]
fn site_build_contains_only_public_files_and_refuses_leftovers() {
    let temp = tempfile::tempdir().unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out = temp.path().join("public site");
    let build = || {
        std::process::Command::new("bash")
            .arg(root.join("site/build.sh"))
            .arg(&out)
            .output()
            .unwrap()
    };
    assert!(build().status.success());
    let mut names = std::fs::read_dir(&out)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        [".nojekyll", "assets", "audio", "index.html", "styles.css"]
    );
    assert_eq!(std::fs::read_dir(out.join("assets")).unwrap().count(), 1);
    assert_eq!(std::fs::read_dir(out.join("audio")).unwrap().count(), 2);
    assert!(out.join("audio/sun-greeting.wav").is_file());
    assert!(out.join("audio/sun-zh.wav").is_file());
    assert!(!out.join("audio/local-assistant-en.wav").exists());
    std::fs::write(out.join("private.txt"), "do not publish").unwrap();
    assert!(!build().status.success());
    assert_eq!(
        std::fs::read_to_string(out.join("private.txt")).unwrap(),
        "do not publish"
    );
}
