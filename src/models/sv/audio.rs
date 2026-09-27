//! WAV decoding and the upstream two-stage sinc resampling used for SV.

use crate::{Error, Result};
use std::path::Path;

pub(super) fn load_reference(path: &Path, sovits_rate: u32) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path).map_err(|e| Error::AudioError(e.to_string()))?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 || sovits_rate == 0 {
        return Err(Error::AudioError(
            "Invalid SV reference audio format".into(),
        ));
    }
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>(),
        hound::SampleFormat::Int => {
            let scale = 2.0f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|v| v.map(|v| v as f32 / scale))
                .collect()
        }
    }
    .map_err(|e| Error::AudioError(format!("Invalid SV reference WAV: {e}")))?;
    if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
        return Err(Error::AudioError(
            "SV reference audio is empty or non-finite".into(),
        ));
    }
    let channels = usize::from(spec.channels);
    if samples.len() % channels != 0 {
        return Err(Error::AudioError(
            "Incomplete SV reference audio frame".into(),
        ));
    }
    let mono: Vec<f32> = samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    let mut audio = resample(&mono, spec.sample_rate, sovits_rate)?;
    let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        audio.iter_mut().for_each(|s| *s /= peak.min(2.0));
    }
    resample(&audio, sovits_rate, super::features::SAMPLE_RATE)
}

// Matches torchaudio.transforms.Resample's default Hann sinc filter (width=6,
// rolloff=0.99), including ceil output length and zero padding at either end.
fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 || to == 0 || from > 384_000 || to > 384_000 {
        return Err(Error::AudioError(
            "Unsupported SV reference sample rate".into(),
        ));
    }
    if from == to {
        return Ok(samples.to_vec());
    }
    let (mut a, mut b) = (from, to);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let orig = (from / a) as usize;
    let new = (to / a) as usize;
    let base = orig.min(new) as f64 * 0.99;
    let width = (6.0 * orig as f64 / base).ceil() as isize;
    let length = samples
        .len()
        .checked_mul(new)
        .ok_or_else(|| Error::AudioError("SV audio is too long".into()))?
        .div_ceil(orig);
    let mut output = vec![0.0; length];
    for phase in 0..new.min(length) {
        // Upstream's phase offsets are f32 before addition to the f64 indices.
        let offset = -(phase as f32) / new as f32;
        let center = -(offset as f64) * orig as f64;
        let start = (center - width as f64).floor() as isize;
        let end = (center + width as f64).ceil() as isize;
        let kernel: Vec<(isize, f32)> = (start..=end)
            .filter_map(|i| {
                let t = (offset as f64 + i as f64 / orig as f64) * base;
                if t.abs() >= 6.0 {
                    return None;
                }
                let window = (t * std::f64::consts::PI / 12.0).cos().powi(2);
                let angle = t * std::f64::consts::PI;
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                Some((i, (sinc * window * base / orig as f64) as f32))
            })
            .collect();
        for index in (phase..length).step_by(new) {
            let base_index = (index / new * orig) as isize;
            output[index] = kernel
                .iter()
                .filter_map(|(offset, weight)| {
                    let pos = base_index + offset;
                    (pos >= 0)
                        .then(|| samples.get(pos as usize).map(|v| v * weight))
                        .flatten()
                })
                .sum();
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_has_ceil_length_and_preserves_dc_away_from_edges() -> Result<()> {
        for rate in [8_000, 16_000, 22_050, 32_000, 44_100, 48_000] {
            let out = resample(&vec![0.25; rate as usize + 1], rate, 16_000)?;
            assert_eq!(
                out.len(),
                ((rate as usize + 1) * 16_000).div_ceil(rate as usize)
            );
            assert!(out[100..out.len() - 100]
                .iter()
                .all(|v| (v - 0.25).abs() < 0.001));
        }
        assert!(resample(&[0.0], 0, 16_000).is_err());
        Ok(())
    }

    #[test]
    fn stereo_pcm_is_averaged_and_normalized_without_peak_boost() -> Result<()> {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(temp.path(), spec).unwrap();
        for _ in 0..400 {
            writer.write_sample(8192i16).unwrap();
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
        let samples = load_reference(temp.path(), 16_000)?;
        assert_eq!(samples, vec![0.125; 400]);
        Ok(())
    }
}
