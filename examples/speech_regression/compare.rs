use super::{capture::Capture, validate_cases};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Deserialize)]
struct AsrReport {
    schema_version: u32,
    backend: String,
    samples: Vec<AsrSample>,
}

#[derive(Deserialize)]
struct AsrSample {
    id: String,
    reference: String,
    hypothesis: String,
    metric: String,
    edits: usize,
    reference_units: usize,
}

#[derive(Serialize)]
pub(super) struct Comparison {
    schema_version: u32,
    baseline: String,
    candidate: String,
    backend: String,
    max_mean_increase: f64,
    max_case_increase: f64,
    mean_increase: f64,
    pub passed: bool,
    cases: Vec<CaseComparison>,
}

#[derive(Serialize)]
struct CaseComparison {
    id: String,
    metric: String,
    baseline_error_rate: f64,
    candidate_error_rate: f64,
    increase: f64,
    baseline_hypotheses: Vec<String>,
    candidate_hypotheses: Vec<String>,
}

fn validate<'a>(
    capture: &Capture,
    report: &'a AsrReport,
) -> Result<BTreeMap<String, &'a AsrSample>> {
    ensure!(
        capture.schema_version == 1 && report.schema_version == 1,
        "unsupported report schema"
    );
    validate_cases(&capture.cases)?;
    ensure!((1..=10).contains(&capture.repeats), "invalid repeat count");
    let expected = capture.cases.len() * capture.repeats as usize;
    ensure!(
        capture.samples.len() == expected && report.samples.len() == expected,
        "incomplete capture or ASR report"
    );
    let mut audio = BTreeMap::new();
    for sample in &capture.samples {
        ensure!(
            sample.issues.is_empty(),
            "{} has capture/quality failures",
            sample.id
        );
        ensure!(
            audio.insert(&sample.id, sample).is_none(),
            "duplicate capture ID"
        );
    }
    let mut asr = BTreeMap::new();
    for sample in &report.samples {
        ensure!(
            matches!(
                sample.metric.as_str(),
                "character_error_rate" | "word_error_rate" | "mixed_error_rate"
            ),
            "unsupported ASR metric: {}",
            sample.metric
        );
        ensure!(sample.reference_units > 0, "empty ASR reference");
        ensure!(
            asr.insert(sample.id.clone(), sample).is_none(),
            "duplicate ASR ID"
        );
    }
    for case in &capture.cases {
        for repeat in 1..=capture.repeats {
            let id = format!("{}-r{repeat:02}", case.id);
            ensure!(audio.contains_key(&id), "missing capture {id}");
            let row = asr
                .get(&id)
                .with_context(|| format!("missing ASR result {id}"))?;
            ensure!(
                row.reference == case.reference,
                "ASR reference changed for {id}"
            );
        }
    }
    Ok(asr)
}

fn compare(
    baseline: Capture,
    candidate: Capture,
    baseline_asr: AsrReport,
    candidate_asr: AsrReport,
    max_mean: f64,
    max_case: f64,
) -> Result<Comparison> {
    ensure!(
        max_mean.is_finite() && max_mean >= 0.0 && max_case.is_finite() && max_case >= 0.0,
        "tolerances must be finite and nonnegative"
    );
    ensure!(
        baseline.voice == candidate.voice
            && baseline.cases == candidate.cases
            && baseline.repeats == candidate.repeats,
        "voice, corpus, or repeat count differs"
    );
    ensure!(
        baseline_asr.backend == candidate_asr.backend,
        "ASR backends differ"
    );
    let old = validate(&baseline, &baseline_asr)?;
    let new = validate(&candidate, &candidate_asr)?;
    let requests: BTreeMap<_, _> = baseline
        .samples
        .iter()
        .map(|s| (&s.id, &s.request))
        .collect();
    for sample in &candidate.samples {
        ensure!(
            requests.get(&sample.id) == Some(&&sample.request),
            "request differs for {}",
            sample.id
        );
    }
    let mut cases = Vec::new();
    for case in &baseline.cases {
        let mut row = CaseComparison {
            id: case.id.clone(),
            metric: old[&format!("{}-r01", case.id)].metric.clone(),
            baseline_error_rate: 0.0,
            candidate_error_rate: 0.0,
            increase: 0.0,
            baseline_hypotheses: Vec::new(),
            candidate_hypotheses: Vec::new(),
        };
        for repeat in 1..=baseline.repeats {
            let id = format!("{}-r{repeat:02}", case.id);
            let a = old[&id];
            let b = new[&id];
            ensure!(
                a.metric == row.metric
                    && a.metric == b.metric
                    && a.reference_units == b.reference_units,
                "ASR normalization differs for {id}"
            );
            row.baseline_error_rate += a.edits as f64 / a.reference_units as f64;
            row.candidate_error_rate += b.edits as f64 / b.reference_units as f64;
            row.baseline_hypotheses.push(a.hypothesis.clone());
            row.candidate_hypotheses.push(b.hypothesis.clone());
        }
        row.baseline_error_rate /= f64::from(baseline.repeats);
        row.candidate_error_rate /= f64::from(baseline.repeats);
        row.increase = row.candidate_error_rate - row.baseline_error_rate;
        cases.push(row);
    }
    let mean_increase = cases.iter().map(|c| c.increase).sum::<f64>() / cases.len() as f64;
    let passed = mean_increase <= max_mean && cases.iter().all(|c| c.increase <= max_case);
    Ok(Comparison {
        schema_version: 1,
        baseline: baseline.label,
        candidate: candidate.label,
        backend: baseline_asr.backend,
        max_mean_increase: max_mean,
        max_case_increase: max_case,
        mean_increase,
        passed,
        cases,
    })
}

pub(super) fn run(
    baseline: &Path,
    candidate: &Path,
    max_mean: f64,
    max_case: f64,
) -> Result<Comparison> {
    compare(
        serde_json::from_slice(&fs::read(baseline.join("capture.json"))?)?,
        serde_json::from_slice(&fs::read(candidate.join("capture.json"))?)?,
        serde_json::from_slice(&fs::read(baseline.join("asr.json"))?)?,
        serde_json::from_slice(&fs::read(candidate.join("asr.json"))?)?,
        max_mean,
        max_case,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn capture() -> Capture {
        serde_json::from_value(json!({"schema_version":1,"label":"test","voice":"demo","repeats":1,
            "cases":[{"id":"short","endpoint":"tts","text":"hello","reference":"hello","language":"en"}],
            "samples":[{"id":"short-r01","request":{"text":"hello"},"duration_s":2.0,"rms":0.1,"issues":[]}]})).unwrap()
    }

    fn asr(edits: usize) -> AsrReport {
        serde_json::from_value(json!({"schema_version":1,"backend":"fixture","samples":[
            {"id":"short-r01","reference":"hello","hypothesis":"hello","metric":"word_error_rate",
             "edits":edits,"reference_units":10}]}))
        .unwrap()
    }

    #[test]
    fn compares_all_attempts_and_flags_a_regression() {
        assert!(
            compare(capture(), capture(), asr(1), asr(1), 0.05, 0.20)
                .unwrap()
                .passed
        );
        assert!(
            !compare(capture(), capture(), asr(1), asr(5), 0.05, 0.20)
                .unwrap()
                .passed
        );
    }

    #[test]
    fn accepts_character_word_and_mixed_metrics_without_hiding_metric_changes() {
        for metric in [
            "character_error_rate",
            "word_error_rate",
            "mixed_error_rate",
        ] {
            let mut baseline = asr(1);
            baseline.samples[0].metric = metric.into();
            let mut candidate = asr(1);
            candidate.samples[0].metric = metric.into();
            let report = compare(capture(), capture(), baseline, candidate, 0.05, 0.2).unwrap();
            assert!(report.passed);
            assert_eq!(report.cases[0].metric, metric);
        }
        let mut different = asr(0);
        different.samples[0].metric = "mixed_error_rate".into();
        assert!(compare(capture(), capture(), asr(0), different, 0.05, 0.2).is_err());
        let mut unsupported = asr(0);
        unsupported.samples[0].metric = "accuracy".into();
        assert!(validate(&capture(), &unsupported).is_err());
    }

    #[test]
    fn rejects_missing_duplicate_and_incompatible_evidence() {
        let mut incomplete = asr(0);
        incomplete.samples.clear();
        assert!(compare(capture(), capture(), asr(0), incomplete, 0.05, 0.2).is_err());
        let mut different = asr(0);
        different.samples[0].reference = "other".into();
        assert!(compare(capture(), capture(), asr(0), different, 0.05, 0.2).is_err());
        let mut duplicate = asr(0);
        duplicate.samples.push(asr(0).samples.remove(0));
        assert!(validate(&capture(), &duplicate).is_err());
        let mut failed = capture();
        failed.samples[0].issues.push("silent".into());
        assert!(compare(capture(), failed, asr(0), asr(0), 0.05, 0.2).is_err());
    }

    #[test]
    fn rejects_changed_sampling_and_invalid_tolerances() {
        let mut different = capture();
        different.samples[0].request["top_k"] = json!(1);
        assert!(compare(capture(), different, asr(0), asr(0), 0.05, 0.2).is_err());
        assert!(compare(capture(), capture(), asr(0), asr(0), f64::NAN, 0.2).is_err());
    }
}
