//! Offline release tooling, not a runtime ASR dependency.
mod capture;
mod compare;

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Capture every attempt and an Audire-compatible ASR manifest using curl.
    Capture {
        #[arg(long)]
        url: String,
        #[arg(long)]
        voice: String,
        /// Include the tested image digest/version; this is recorded, not auto-detected.
        #[arg(long)]
        label: String,
        #[arg(long, default_value = "examples/speech_regression/corpus.json")]
        corpus: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=10))]
        repeats: u32,
    },
    /// Compare capture.json plus externally produced asr.json in two run directories.
    Compare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long, default_value_t = 0.05)]
        max_mean_increase: f64,
        #[arg(long, default_value_t = 0.20)]
        max_case_increase: f64,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Endpoint {
    Tts,
    Stream,
    Speech,
}

impl Endpoint {
    fn path(&self) -> &'static str {
        match self {
            Self::Tts => "/tts",
            Self::Stream => "/tts/stream",
            Self::Speech => "/v1/audio/speech",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    endpoint: Endpoint,
    text: String,
    reference: String,
    language: String,
}

fn validate_cases(cases: &[Case]) -> Result<()> {
    ensure!(!cases.is_empty(), "empty corpus");
    let mut ids = HashSet::new();
    for case in cases {
        ensure!(
            !case.id.is_empty()
                && case
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "unsafe case ID: {}",
            case.id
        );
        ensure!(ids.insert(&case.id), "duplicate case ID: {}", case.id);
        ensure!(
            !case.text.trim().is_empty() && !case.reference.trim().is_empty(),
            "empty text/reference"
        );
        ensure!(
            matches!(case.language.as_str(), "zh" | "en"),
            "unsupported ASR language"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Capture {
            url,
            voice,
            label,
            corpus,
            output_dir,
            repeats,
        } => {
            let cases = serde_json::from_slice::<Vec<Case>>(&fs::read(corpus)?)?;
            validate_cases(&cases)?;
            ensure!(
                !voice.trim().is_empty() && !label.trim().is_empty(),
                "voice and label are required"
            );
            ensure!(
                url.starts_with("http://") || url.starts_with("https://"),
                "expected an HTTP base URL"
            );
            // Never overwrite prior runs, even if empty or only partially complete.
            fs::create_dir(&output_dir).context("output directory must not already exist")?;
            capture::run(&url, &voice, &label, &cases, repeats, &output_dir)
        }
        Command::Compare {
            baseline,
            candidate,
            max_mean_increase,
            max_case_increase,
            output,
        } => {
            let report = compare::run(&baseline, &candidate, max_mean_increase, max_case_increase)?;
            let file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)?;
            serde_json::to_writer_pretty(file, &report)?;
            ensure!(
                report.passed,
                "speech regression: inspect the comparison report and audio"
            );
            println!("No measured regression within the configured tolerances; listening is still required.");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_valid_and_covers_all_http_audio_endpoints() {
        let cases: Vec<Case> = serde_json::from_str(include_str!("corpus.json")).unwrap();
        validate_cases(&cases).unwrap();
        for endpoint in [Endpoint::Tts, Endpoint::Stream, Endpoint::Speech] {
            assert!(cases.iter().any(|c| c.endpoint == endpoint));
        }
    }

    #[test]
    fn rejects_empty_duplicate_and_path_traversal_ids() {
        assert!(validate_cases(&[]).is_err());
        let mut cases: Vec<Case> = serde_json::from_str(include_str!("corpus.json")).unwrap();
        cases.push(cases[0].clone());
        assert!(validate_cases(&cases).is_err());
        cases.pop();
        cases[0].id = "../private".into();
        assert!(validate_cases(&cases).is_err());
    }
}
