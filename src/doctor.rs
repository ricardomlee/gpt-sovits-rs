//! Setup diagnostics for the CLI `--doctor` command.

use crate::cli::Args;
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::voice::{list_voice_profiles, LoadedVoiceProfile};
use gpt_sovits_rs::Config;
use std::path::{Path, PathBuf};

pub(crate) fn run_doctor(args: &Args) -> bool {
    let mut report = DoctorReport::default();
    println!("GPT-SoVITS-RS doctor\n");

    check_device(&mut report, &args.device);
    check_models(&mut report, args);
    check_voices(&mut report, args);

    println!();
    if report.errors == 0 {
        println!(
            "Doctor finished: {} check(s), {} warning(s), no blocking errors.",
            report.checks, report.warnings
        );
        true
    } else {
        println!(
            "Doctor found {} blocking error(s) and {} warning(s) across {} check(s).",
            report.errors, report.warnings, report.checks
        );
        println!("Fix the errors above, then run `gpt-sovits --doctor` again.");
        false
    }
}

#[derive(Default)]
struct DoctorReport {
    checks: usize,
    warnings: usize,
    errors: usize,
}

impl DoctorReport {
    fn ok(&mut self, message: impl AsRef<str>) {
        self.checks += 1;
        println!("[ok] {}", message.as_ref());
    }

    fn warn(&mut self, message: impl AsRef<str>) {
        self.checks += 1;
        self.warnings += 1;
        println!("[warn] {}", message.as_ref());
    }

    fn error(&mut self, message: impl AsRef<str>) {
        self.checks += 1;
        self.errors += 1;
        println!("[error] {}", message.as_ref());
    }
}

fn check_device(report: &mut DoctorReport, requested: &str) {
    let config = Config::builder().with_device(requested).build();
    match requested {
        "cuda" => {
            #[cfg(feature = "cuda")]
            match candle_core::Device::new_cuda(0) {
                Ok(_) => report.ok("CUDA device is available"),
                Err(e) => report.error(format!(
                    "CUDA was requested, but Candle could not open CUDA device 0: {e}"
                )),
            }
            #[cfg(not(feature = "cuda"))]
            report.error("CUDA was requested, but this binary was not built with --features cuda");
        }
        "mps" => match candle_core::Device::new_metal(0) {
            Ok(_) => report.ok("Metal/MPS device is available"),
            Err(e) => report.error(format!(
                "MPS was requested, but Candle could not open Metal device 0: {e}"
            )),
        },
        "cpu" => report.ok("CPU device selected"),
        "auto" => report.ok(format!("auto device resolved to {}", config.device)),
        other => report.error(format!(
            "Unsupported device '{other}'; expected auto, cuda, cpu, or mps"
        )),
    }
}

fn check_models(report: &mut DoctorReport, args: &Args) {
    if args.models_dir.is_dir() {
        report.ok(format!(
            "models directory exists: {}",
            args.models_dir.display()
        ));
    } else {
        report.warn(format!(
            "models directory does not exist yet: {}",
            args.models_dir.display()
        ));
    }

    match ModelPaths::discover(
        &args.models_dir,
        ModelPathOverrides {
            gpt: args.gpt_model.clone(),
            sovits: args.sovits_model.clone(),
            bert: args.bert_model.clone(),
            hubert: args.hubert_model.clone(),
        },
    ) {
        Ok(paths) => {
            check_safetensors(report, "GPT", &paths.gpt, true);
            check_safetensors(report, "SoVITS", &paths.sovits, true);
            match paths.bert.as_ref() {
                Some(path) => {
                    check_safetensors(report, "BERT", path, false);
                    check_bert_tokenizer(report, path);
                }
                None => report.warn("BERT model not found; speech quality will be reduced"),
            }
            match paths.hubert.as_ref() {
                Some(path) => check_safetensors(report, "HuBERT", path, false),
                None => report.warn("HuBERT model not found; voice similarity will be reduced"),
            }
        }
        Err(e) => report.error(e),
    }
}

fn check_safetensors(report: &mut DoctorReport, label: &str, path: &Path, required: bool) {
    if !path.is_file() {
        let message = format!("{label} model file does not exist: {}", path.display());
        if required {
            report.error(message);
        } else {
            report.warn(message);
        }
        return;
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("safetensors") => {}
        Some("ckpt" | "pth") => {
            report.error(format!(
                "{label} points to a raw PyTorch checkpoint, not a runtime model: {}. Convert it to safetensors first.",
                path.display()
            ));
            return;
        }
        Some(other) => {
            report.warn(format!(
                "{label} model extension is .{other}; expected .safetensors: {}",
                path.display()
            ));
        }
        None => report.warn(format!(
            "{label} model has no file extension; expected .safetensors: {}",
            path.display()
        )),
    }

    match read_safetensors_header_count(path) {
        Ok(count) => report.ok(format!(
            "{label} safetensors header is readable: {} ({count} tensor entries)",
            path.display()
        )),
        Err(e) => report.error(format!(
            "{label} safetensors header is not readable: {} ({e})",
            path.display()
        )),
    }
}

fn read_safetensors_header_count(path: &Path) -> Result<usize, String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(format!("invalid safetensors header length: {header_len}"));
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "safetensors header is not a JSON object".to_string())?;
    Ok(object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .count())
}

fn check_bert_tokenizer(report: &mut DoctorReport, bert_path: &Path) {
    let sibling = bert_path.with_file_name("tokenizer.json");
    let fallback = PathBuf::from("models/bert/tokenizer.json");
    if sibling.is_file() {
        report.ok(format!("BERT tokenizer found: {}", sibling.display()));
    } else if fallback.is_file() {
        report.ok(format!(
            "BERT tokenizer found through fallback: {}",
            fallback.display()
        ));
    } else {
        report.error(format!(
            "BERT tokenizer not found. Put tokenizer.json next to {} or at {}",
            bert_path.display(),
            fallback.display()
        ));
    }
}

fn check_voices(report: &mut DoctorReport, args: &Args) {
    match list_voice_profiles(&args.voices_dir) {
        Ok(voices) if voices.is_empty() => report.warn(format!(
            "no voice profiles found in {}; create voices/<name>/voice.json or pass --reference-audio and --reference-text",
            args.voices_dir.display()
        )),
        Ok(voices) => report.ok(format!(
            "found {} voice profile(s) in {}: {}",
            voices.len(),
            args.voices_dir.display(),
            voices.join(", ")
        )),
        Err(e) => report.error(e),
    }

    if let Some(name) = args.voice.as_deref() {
        match LoadedVoiceProfile::load(name, &args.voices_dir) {
            Ok(profile) => check_loaded_voice(report, &profile),
            Err(e) => report.error(e),
        }
    } else {
        if let Some(audio) = args.reference_audio.as_ref() {
            check_reference_audio(report, audio);
        }
        if let Some(text) = args.reference_text.as_deref() {
            check_reference_text(report, text);
        }
        if args.reference_audio.is_none() && args.reference_text.is_none() {
            report.warn("no --voice or inline reference audio/text selected for a test run");
        }
    }
}

fn check_loaded_voice(report: &mut DoctorReport, voice: &LoadedVoiceProfile) {
    report.ok(format!(
        "voice profile '{}' parsed: {}",
        voice.name,
        voice.dir.join("voice.json").display()
    ));

    match voice.reference_audio_path() {
        Some(path) => check_reference_audio(report, &path),
        None => report.error(format!(
            "voice '{}' does not set reference_audio",
            voice.name
        )),
    }
    match voice.reference_text() {
        Some(text) => check_reference_text(report, text),
        None => report.error(format!(
            "voice '{}' does not set reference_text",
            voice.name
        )),
    }
}

fn check_reference_audio(report: &mut DoctorReport, path: &Path) {
    if !path.is_file() {
        report.error(format!(
            "reference audio does not exist: {}",
            path.display()
        ));
        return;
    }
    if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
        report.warn(format!(
            "reference audio is not a .wav file; current CLI path expects WAV: {}",
            path.display()
        ));
    }
    match hound::WavReader::open(path) {
        Ok(reader) => {
            let spec = reader.spec();
            let duration = reader.duration() as f32 / spec.sample_rate as f32;
            let level = if (3.0..=10.0).contains(&duration) {
                "ok"
            } else {
                "warn"
            };
            let message = format!(
                "reference audio: {} ({:.2}s, {} Hz, {} channel(s))",
                path.display(),
                duration,
                spec.sample_rate,
                spec.channels
            );
            if level == "ok" {
                report.ok(message);
            } else {
                report.warn(format!("{message}; recommended reference length is 3-10s"));
            }
        }
        Err(e) => report.error(format!(
            "reference audio is not readable as WAV: {} ({e})",
            path.display()
        )),
    }
}

fn check_reference_text(report: &mut DoctorReport, text: &str) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        report.error("reference_text is empty");
    } else {
        report.ok(format!(
            "reference_text is present ({} chars)",
            trimmed.chars().count()
        ));
    }
}
