//! Objective audio sanity checks for generated samples.

use crate::utils::AudioBuffer;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioQualityMetrics {
    pub duration_s: f32,
    pub peak: f32,
    pub rms: f32,
    pub clipping_ratio: f32,
    pub silence_ratio: f32,
    pub dc_offset: f32,
    pub has_non_finite: bool,
}

impl AudioQualityMetrics {
    pub fn from_audio(audio: &AudioBuffer) -> Self {
        if audio.samples.is_empty() {
            return Self {
                duration_s: 0.0,
                peak: 0.0,
                rms: 0.0,
                clipping_ratio: 0.0,
                silence_ratio: 1.0,
                dc_offset: 0.0,
                has_non_finite: false,
            };
        }

        let mut peak = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum = 0.0f32;
        let mut clipped = 0usize;
        let mut silent = 0usize;
        let mut has_non_finite = false;

        for &sample in &audio.samples {
            if !sample.is_finite() {
                has_non_finite = true;
                continue;
            }
            let abs = sample.abs();
            peak = peak.max(abs);
            sum_sq += sample * sample;
            sum += sample;
            if abs >= 0.999 {
                clipped += 1;
            }
            if abs < 1e-4 {
                silent += 1;
            }
        }

        let n = audio.samples.len() as f32;
        Self {
            duration_s: audio.duration(),
            peak,
            rms: (sum_sq / n).sqrt(),
            clipping_ratio: clipped as f32 / n,
            silence_ratio: silent as f32 / n,
            dc_offset: sum / n,
            has_non_finite,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioQualityThresholds {
    pub min_duration_s: f32,
    pub max_duration_s: Option<f32>,
    pub min_rms: f32,
    pub max_peak: f32,
    pub max_clipping_ratio: f32,
    pub max_silence_ratio: f32,
    pub max_abs_dc_offset: f32,
}

impl Default for AudioQualityThresholds {
    fn default() -> Self {
        Self {
            min_duration_s: 0.1,
            max_duration_s: None,
            min_rms: 1e-4,
            max_peak: 1.2,
            max_clipping_ratio: 0.01,
            max_silence_ratio: 0.98,
            max_abs_dc_offset: 0.2,
        }
    }
}

pub fn validate_audio_quality(
    metrics: &AudioQualityMetrics,
    thresholds: &AudioQualityThresholds,
) -> Vec<String> {
    let mut issues = Vec::new();
    if metrics.has_non_finite {
        issues.push("audio contains NaN or infinite samples".to_string());
    }
    if metrics.duration_s < thresholds.min_duration_s {
        issues.push(format!(
            "duration {:.3}s is below minimum {:.3}s",
            metrics.duration_s, thresholds.min_duration_s
        ));
    }
    if let Some(max_duration_s) = thresholds.max_duration_s {
        if metrics.duration_s > max_duration_s {
            issues.push(format!(
                "duration {:.3}s exceeds maximum {:.3}s",
                metrics.duration_s, max_duration_s
            ));
        }
    }
    if metrics.rms < thresholds.min_rms {
        issues.push(format!(
            "rms {:.6} is below minimum {:.6}",
            metrics.rms, thresholds.min_rms
        ));
    }
    if metrics.peak > thresholds.max_peak {
        issues.push(format!(
            "peak {:.3} exceeds maximum {:.3}",
            metrics.peak, thresholds.max_peak
        ));
    }
    if metrics.clipping_ratio > thresholds.max_clipping_ratio {
        issues.push(format!(
            "clipping ratio {:.4} exceeds maximum {:.4}",
            metrics.clipping_ratio, thresholds.max_clipping_ratio
        ));
    }
    if metrics.silence_ratio > thresholds.max_silence_ratio {
        issues.push(format!(
            "silence ratio {:.4} exceeds maximum {:.4}",
            metrics.silence_ratio, thresholds.max_silence_ratio
        ));
    }
    if metrics.dc_offset.abs() > thresholds.max_abs_dc_offset {
        issues.push(format!(
            "dc offset {:.4} exceeds maximum absolute {:.4}",
            metrics.dc_offset, thresholds.max_abs_dc_offset
        ));
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_normal_audio() {
        let samples = (0..24_000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.2)
            .collect();
        let audio = AudioBuffer::new(samples, 24_000, 1);
        let metrics = AudioQualityMetrics::from_audio(&audio);
        let issues = validate_audio_quality(&metrics, &AudioQualityThresholds::default());
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn flags_silence() {
        let audio = AudioBuffer::new(vec![0.0; 24_000], 24_000, 1);
        let metrics = AudioQualityMetrics::from_audio(&audio);
        let issues = validate_audio_quality(&metrics, &AudioQualityThresholds::default());
        assert!(issues.iter().any(|issue| issue.contains("rms")));
        assert!(issues.iter().any(|issue| issue.contains("silence")));
    }

    #[test]
    fn flags_clipping_and_non_finite_samples() {
        let mut samples = vec![1.0; 24_000];
        samples[0] = f32::NAN;
        let audio = AudioBuffer::new(samples, 24_000, 1);
        let metrics = AudioQualityMetrics::from_audio(&audio);
        let issues = validate_audio_quality(&metrics, &AudioQualityThresholds::default());
        assert!(issues.iter().any(|issue| issue.contains("NaN")));
        assert!(issues.iter().any(|issue| issue.contains("clipping")));
    }
}
