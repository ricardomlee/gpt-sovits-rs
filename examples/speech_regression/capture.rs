use super::{Case, Endpoint};
use anyhow::{Context, Result, ensure};
use gpt_sovits_rs::{
    AudioBuffer,
    audio_checks::{AudioQualityMetrics, AudioQualityThresholds, validate_audio_quality},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Cursor, Write},
    path::Path,
    process::Command,
};

#[derive(Deserialize, Serialize)]
pub(super) struct Capture {
    pub schema_version: u32,
    pub label: String,
    pub voice: String,
    pub repeats: u32,
    #[serde(default)]
    pub seeds: Vec<u64>,
    pub cases: Vec<Case>,
    pub samples: Vec<Sample>,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Sample {
    pub id: String,
    pub request: Value,
    pub duration_s: Option<f32>,
    pub rms: Option<f32>,
    pub issues: Vec<String>,
}

fn request(case: &Case, voice: &str, seed: Option<u64>) -> Value {
    // Keep the voice's sampling settings. Auto detects reference and target languages separately.
    let mut request = json!({"voice": voice, "text_language": "auto"});
    if let Some(seed) = seed {
        request["seed"] = json!(seed);
    }
    if case.endpoint == Endpoint::Speech {
        request["input"] = json!(case.text);
        request["model"] = json!("gpt-sovits");
        request["response_format"] = json!("wav");
    } else {
        request["text"] = json!(case.text);
    }
    request
}

fn audio(bytes: &[u8], streaming: bool) -> Result<(AudioBuffer, Vec<u8>)> {
    let mut bytes = bytes.to_vec();
    if streaming {
        // Only normalize our documented unknown-length PCM header, retaining the raw response.
        ensure!(
            bytes.len() >= 44
                && &bytes[..4] == b"RIFF"
                && &bytes[8..16] == b"WAVEfmt "
                && bytes[16..20] == 16u32.to_le_bytes()
                && &bytes[36..40] == b"data"
                && bytes[4..8] == 0xffff_fffeu32.to_le_bytes()
                && bytes[40..44] == 0xffff_fffeu32.to_le_bytes(),
            "unexpected streaming WAV header"
        );
        ensure!((bytes.len() - 44).is_multiple_of(2), "partial PCM sample");
        let length = u32::try_from(bytes.len()).context("WAV too large")?;
        bytes[4..8].copy_from_slice(&(length - 8).to_le_bytes());
        bytes[40..44].copy_from_slice(&(length - 44).to_le_bytes());
    }
    let mut reader = hound::WavReader::new(Cursor::new(&bytes))?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.sample_rate == 32_000
            && spec.bits_per_sample == 16
            && spec.sample_format == hound::SampleFormat::Int,
        "expected 32 kHz mono PCM16"
    );
    let samples = reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) / 32768.0))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((AudioBuffer::new(samples, 32_000, 1), bytes))
}

fn one(url: &str, directory: &Path, case: &Case, sample: &mut Sample) -> Result<()> {
    let stem = directory.join(&sample.id);
    let request_path = stem.with_extension("request.json");
    let raw = stem.with_extension("response.wav");
    fs::write(&request_path, serde_json::to_vec_pretty(&sample.request)?)?;
    let result = Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--connect-timeout",
            "10",
            "--max-time",
            "300",
            "--header",
            "Content-Type: application/json",
            "--data-binary",
        ])
        .arg(format!("@{}", request_path.display()))
        .arg("--output")
        .arg(&raw)
        .arg("--dump-header")
        .arg(stem.with_extension("headers.txt"))
        .args(["--write-out", "%{json}", "--url"])
        .arg(format!(
            "{}{}",
            url.trim_end_matches('/'),
            case.endpoint.path()
        ))
        .output()
        .context("curl is required")?;
    fs::write(stem.with_extension("curl.json"), &result.stdout)?;
    fs::write(stem.with_extension("stderr.txt"), &result.stderr)?;
    ensure!(result.status.success(), "curl failed: {}", result.status);
    let transfer: Value = serde_json::from_slice(&result.stdout)?;
    ensure!(transfer["http_code"] == 200, "expected HTTP 200");
    if let Some(seed) = sample.request["seed"].as_u64() {
        let headers = fs::read_to_string(stem.with_extension("headers.txt"))?;
        ensure!(
            headers.lines().any(|line| line
                .split_once(':')
                .is_some_and(|(key, value)| key.eq_ignore_ascii_case("x-tts-seed")
                    && value.trim() == seed.to_string())),
            "server did not acknowledge seed {seed}; it may not support seeded synthesis"
        );
    }
    let (wav, normalized) = audio(&fs::read(raw)?, case.endpoint == Endpoint::Stream)?;
    fs::write(stem.with_extension("wav"), normalized)?;
    let metrics = AudioQualityMetrics::from_audio(&wav);
    sample.duration_s = Some(metrics.duration_s);
    sample.rms = Some(metrics.rms);
    sample.issues = validate_audio_quality(&metrics, &AudioQualityThresholds::default());
    fs::write(
        stem.with_extension("metrics.json"),
        serde_json::to_vec_pretty(&json!({
            "duration_s": metrics.duration_s, "peak": metrics.peak, "rms": metrics.rms,
            "clipping_ratio": metrics.clipping_ratio, "silence_ratio": metrics.silence_ratio,
            "dc_offset": metrics.dc_offset, "has_non_finite": metrics.has_non_finite,
            "issues": sample.issues
        }))?,
    )?;
    Ok(())
}

pub(super) fn run(
    url: &str,
    voice: &str,
    label: &str,
    cases: &[Case],
    repeats: u32,
    seeds: &[u64],
    directory: &Path,
) -> Result<()> {
    let mut capture = Capture {
        schema_version: 1,
        label: label.into(),
        voice: voice.into(),
        repeats,
        seeds: seeds.to_vec(),
        cases: cases.to_vec(),
        samples: Vec::new(),
    };
    let mut manifest = fs::File::create(directory.join("asr.jsonl"))?;
    for case in cases {
        for repeat in 1..=repeats {
            let mut sample = Sample {
                id: format!("{}-r{repeat:02}", case.id),
                request: request(case, voice, seeds.get((repeat - 1) as usize).copied()),
                duration_s: None,
                rms: None,
                issues: Vec::new(),
            };
            println!("Capturing {} ({})", sample.id, case.endpoint.path());
            match one(url, directory, case, &mut sample) {
                Ok(()) => writeln!(
                    manifest,
                    "{}",
                    json!({"id": sample.id,
                    "audio": format!("{}.wav", sample.id), "reference": case.reference,
                    "language": case.language, "force_language": case.language})
                )?,
                Err(error) => sample.issues.push(format!("{error:#}")),
            }
            capture.samples.push(sample);
            fs::write(
                directory.join("capture.json"),
                serde_json::to_vec_pretty(&capture)?,
            )?;
        }
    }
    ensure!(
        capture.samples.iter().all(|s| s.issues.is_empty()),
        "capture has failures; all attempts retained in capture.json"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav() -> Vec<u8> {
        AudioBuffer::new(vec![0.1; 3200], 32000, 1)
            .to_wav_bytes()
            .unwrap()
    }

    #[test]
    fn reads_pcm_and_rejects_truncated_or_error_responses() {
        let bytes = wav();
        assert_eq!(audio(&bytes, false).unwrap().0.samples.len(), 3200);
        assert!(audio(&bytes[..bytes.len() - 1], false).is_err());
        assert!(audio(b"{\"error\":\"failed\"}", false).is_err());
    }

    #[test]
    fn normalizes_only_known_streaming_headers() {
        let mut bytes = wav();
        // hound writes a standard PCM16 WAV with the same fixed-format header.
        assert_eq!(&bytes[36..40], b"data");
        assert!(audio(&bytes, true).is_err());
        bytes[4..8].copy_from_slice(&0xffff_fffeu32.to_le_bytes());
        bytes[40..44].copy_from_slice(&0xffff_fffeu32.to_le_bytes());
        let (decoded, normalized) = audio(&bytes, true).unwrap();
        assert_eq!(decoded.samples.len(), 3200);
        assert_eq!(normalized[44..], bytes[44..]);
        assert!(audio(&bytes[..bytes.len() - 1], true).is_err());
    }

    #[test]
    fn requests_do_not_override_voice_sampling() {
        let cases: Vec<Case> = serde_json::from_str(include_str!("corpus.json")).unwrap();
        for case in cases {
            let request = request(&case, "sun", None);
            assert!(request.get("seed").is_none());
            for key in ["top_k", "temperature", "top_p", "repetition_penalty"] {
                assert!(request.get(key).is_none());
            }
            let field = if case.endpoint == Endpoint::Speech {
                "input"
            } else {
                "text"
            };
            assert_eq!(request[field], case.text);
            assert_eq!(super::request(&case, "sun", Some(0))["seed"], 0);
        }
    }
}
