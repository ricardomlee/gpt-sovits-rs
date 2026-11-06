use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fmt::Write as _, fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Case {
    pub id: String,
    pub text: String,
    pub reference: String,
    pub language: String,
}

#[derive(Debug, Serialize)]
pub struct Attempt {
    pub elapsed_ms: f64,
    pub audio_seconds: Option<f64>,
    pub output: Option<String>,
    pub metrics: Option<Value>,
    pub issues: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CaseReport {
    pub input: Case,
    pub warmups: Vec<Attempt>,
    pub runs: Vec<Attempt>,
}

#[derive(Debug, Serialize)]
pub struct Statistics {
    pub min: f64,
    pub median: f64,
    pub max: f64,
}

fn statistics(mut values: Vec<f64>) -> Option<Statistics> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values[mid - 1] / 2.0 + values[mid] / 2.0
    } else {
        values[mid]
    };
    Some(Statistics {
        min: values[0],
        median,
        max: values[values.len() - 1],
    })
}

#[derive(Debug, Serialize)]
pub struct Summary {
    pub latency_ms: Statistics,
    pub audio_seconds: Statistics,
    pub rtf: Statistics,
}

impl CaseReport {
    pub fn summary(&self) -> Option<Summary> {
        if self
            .warmups
            .iter()
            .chain(&self.runs)
            .any(|run| !run.issues.is_empty())
        {
            return None;
        }
        let durations = self
            .runs
            .iter()
            .map(|run| run.audio_seconds)
            .collect::<Option<Vec<_>>>()?;
        let latency_ms = statistics(self.runs.iter().map(|run| run.elapsed_ms).collect())?;
        let audio_seconds = statistics(durations.clone())?;
        let rtf = statistics(
            self.runs
                .iter()
                .zip(durations)
                .map(|(run, duration)| run.elapsed_ms / 1000.0 / duration)
                .collect(),
        )?;
        Some(Summary {
            latency_ms,
            audio_seconds,
            rtf,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub metadata: Value,
    pub warmup_count: u32,
    pub repeat_count: u32,
    pub completed: bool,
    pub fatal_error: Option<String>,
    pub model_load_ms: Option<f64>,
    pub reference_preload_ms: Option<f64>,
    pub cases: Vec<CaseReport>,
}

impl Report {
    fn summary(&self, case: &CaseReport) -> Option<Summary> {
        if case.warmups.len() != self.warmup_count as usize
            || case.runs.len() != self.repeat_count as usize
        {
            return None;
        }
        case.summary()
    }

    pub fn passed(&self) -> bool {
        self.completed
            && self.fatal_error.is_none()
            && !self.cases.is_empty()
            && self.cases.iter().all(|case| {
                case.warmups.len() == self.warmup_count as usize
                    && case.runs.len() == self.repeat_count as usize
                    && case.summary().is_some()
            })
    }

    pub fn save(&self, directory: &Path) -> Result<()> {
        let mut json = serde_json::to_value(self)?;
        json["signal_checks_passed"] = self.passed().into();
        json["summaries"] = serde_json::to_value(
            self.cases
                .iter()
                .map(|case| (&case.input.id, self.summary(case)))
                .collect::<std::collections::BTreeMap<_, _>>(),
        )?;
        fs::write(
            directory.join("report.json"),
            serde_json::to_vec_pretty(&json)?,
        )?;
        fs::write(directory.join("report.md"), self.markdown())?;
        Ok(())
    }

    pub fn markdown(&self) -> String {
        let mut output = String::from("# Local GPU Benchmark\n\n");
        writeln!(
            output,
            "Completed: **{}**. Signal checks passed: **{}**.\n",
            self.completed,
            self.passed()
        )
        .unwrap();
        output.push_str("This is warm in-process synthesis, not HTTP latency, audio TTFA, or a Python comparison.\n");
        output.push_str("RTF = synthesis seconds / generated audio seconds. Lower is faster. Shorter or incorrect speech is not an improvement.\n");
        output.push_str("Signal checks cannot prove spoken completeness; inspect WAVs and run the separate ASR manifest before release.\n\n");
        writeln!(
            output,
            "Warmups per case: {}. Measured repeats: {}. No retries or sampling overrides.\n",
            self.warmup_count, self.repeat_count
        )
        .unwrap();
        let milliseconds = |value: Option<f64>| {
            value.map_or_else(|| "unavailable".into(), |value| format!("{value:.2} ms"))
        };
        writeln!(
            output,
            "Model load: {}. Reference preload: {} (excluded from warm measurements).\n",
            milliseconds(self.model_load_ms),
            milliseconds(self.reference_preload_ms)
        )
        .unwrap();
        output.push_str("| Case | Runs | Median ms | Min-max ms | Median audio s | Median RTF |\n| --- | ---: | ---: | ---: | ---: | ---: |\n");
        for case in &self.cases {
            if let Some(summary) = self.summary(case) {
                writeln!(
                    output,
                    "| {} | {} | {:.2} | {:.2}-{:.2} | {:.2} | {:.4} |",
                    case.input.id,
                    case.runs.len(),
                    summary.latency_ms.median,
                    summary.latency_ms.min,
                    summary.latency_ms.max,
                    summary.audio_seconds.median,
                    summary.rtf.median
                )
                .unwrap();
            } else {
                writeln!(
                    output,
                    "| {} | {} | unavailable | failed/incomplete | | |",
                    case.input.id,
                    case.runs.len()
                )
                .unwrap();
            }
        }
        output.push_str("\n## Provenance\n\n```json\n");
        output.push_str(&serde_json::to_string_pretty(&self.metadata).unwrap());
        output.push_str("\n```\n\nAll warmups and measured attempts, including failures, are retained in `report.json`.\n");
        if let Some(error) = &self.fatal_error {
            writeln!(output, "\nFatal error: {error}").unwrap();
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempt(ms: f64, seconds: f64) -> Attempt {
        Attempt {
            elapsed_ms: ms,
            audio_seconds: Some(seconds),
            output: None,
            metrics: None,
            issues: vec![],
        }
    }

    #[test]
    fn median_and_rtf_use_each_measured_attempt_not_warmup() {
        let case = CaseReport {
            input: Case {
                id: "test".into(),
                text: "text".into(),
                reference: "text".into(),
                language: "en".into(),
            },
            warmups: vec![attempt(9000.0, 1.0)],
            runs: vec![
                attempt(100.0, 1.0),
                attempt(300.0, 2.0),
                attempt(200.0, 4.0),
            ],
        };
        let summary = case.summary().unwrap();
        assert_eq!(summary.latency_ms.median, 200.0);
        assert_eq!(summary.rtf.median, 0.1);
        assert_eq!(statistics(vec![4.0, 2.0]).unwrap().median, 3.0);
    }

    #[test]
    fn invalid_measurements_are_not_summarized() {
        for values in [
            vec![],
            vec![f64::NAN],
            vec![f64::INFINITY],
            vec![0.0],
            vec![-1.0],
        ] {
            assert!(statistics(values).is_none());
        }
    }

    #[test]
    fn incomplete_and_failed_attempts_do_not_pass() {
        let mut report = Report {
            schema_version: 1,
            metadata: serde_json::json!({}),
            warmup_count: 1,
            repeat_count: 1,
            completed: true,
            fatal_error: None,
            model_load_ms: None,
            reference_preload_ms: None,
            cases: vec![CaseReport {
                input: Case {
                    id: "test".into(),
                    text: "text".into(),
                    reference: "text".into(),
                    language: "en".into(),
                },
                warmups: vec![attempt(200.0, 1.0)],
                runs: vec![],
            }],
        };
        assert!(!report.passed());
        report.cases[0].runs.push(attempt(100.0, 1.0));
        assert!(report.passed());
        report.cases[0].runs[0].issues.push("silent".into());
        assert!(!report.passed());
        assert!(report.markdown().contains("failed/incomplete"));
        let directory = tempfile::tempdir().unwrap();
        report.save(directory.path()).unwrap();
        let saved: Value =
            serde_json::from_slice(&fs::read(directory.path().join("report.json")).unwrap())
                .unwrap();
        assert_eq!(saved["signal_checks_passed"], false);
        assert_eq!(saved["cases"][0]["runs"][0]["issues"][0], "silent");
    }
}
