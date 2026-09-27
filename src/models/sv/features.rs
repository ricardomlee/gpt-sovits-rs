//! The fixed Kaldi fbank configuration used by GPT-SoVITS SV (not a TTS mel).

use crate::{Error, Result};
use candle_core::{Device, Tensor};
use rustfft::{num_complex::Complex32, FftPlanner};

pub const SAMPLE_RATE: u32 = 16_000;
pub const MEL_BINS: usize = 80;
const WINDOW: usize = 400;
const SHIFT: usize = 160;
const FFT: usize = 512;

/// 25 ms Povey windows, 10 ms shift, snip_edges, no dither or CMVN.
pub fn fbank(samples: &[f32], device: &Device) -> Result<Tensor> {
    if samples.len() < WINDOW || samples.iter().any(|v| !v.is_finite()) {
        return Err(Error::AudioError(
            "SV needs at least 25 ms of finite 16 kHz mono audio".into(),
        ));
    }
    let frames = 1 + (samples.len() - WINDOW) / SHIFT;
    let angle_step = (2.0 * std::f64::consts::PI / (WINDOW - 1) as f64) as f32;
    let window: Vec<f32> = (0..WINDOW)
        .map(|i| (0.5 - 0.5 * (angle_step * i as f32).cos()).powf(0.85))
        .collect();
    let mel = |hz: f32| 1127.0 * (1.0 + hz / 700.0).ln();
    // Kaldi computes scalar endpoints in double precision, then tensor math in f32.
    let low = (1127.0f64 * (1.0 + 20.0 / 700.0f64).ln()) as f32;
    let delta = ((1127.0f64 * (1.0 + 8000.0 / 700.0f64).ln()
        - 1127.0f64 * (1.0 + 20.0 / 700.0f64).ln())
        / (MEL_BINS + 1) as f64) as f32;
    let banks: Vec<Vec<f32>> = (0..MEL_BINS)
        .map(|bin| {
            let left = low + bin as f32 * delta;
            let center = low + (bin + 1) as f32 * delta;
            let right = low + (bin + 2) as f32 * delta;
            (0..FFT / 2)
                .map(|k| {
                    let frequency = mel(k as f32 * SAMPLE_RATE as f32 / FFT as f32);
                    ((frequency - left) / (center - left))
                        .min((right - frequency) / (right - center))
                        .max(0.0)
                })
                .collect()
        })
        .collect();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT);
    let mut scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
    let mut spectrum = vec![Complex32::default(); FFT];
    let mut features = Vec::with_capacity(frames * MEL_BINS);
    for frame in samples.windows(WINDOW).step_by(SHIFT) {
        let mean = frame.iter().sum::<f32>() / WINDOW as f32;
        spectrum.fill(Complex32::default());
        for i in 0..WINDOW {
            let previous = frame[i.saturating_sub(1)] - mean;
            spectrum[i].re = (frame[i] - mean - 0.97 * previous) * window[i];
        }
        fft.process_with_scratch(&mut spectrum, &mut scratch);
        for bank in &banks {
            let energy: f32 = bank
                .iter()
                .zip(&spectrum)
                .map(|(weight, value)| weight * value.norm_sqr())
                .sum();
            features.push(energy.max(f32::EPSILON).ln());
        }
    }
    Ok(Tensor::from_vec(features, (1, frames, MEL_BINS), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_uses_kaldi_log_floor_and_snips_edges() -> Result<()> {
        let features = fbank(&vec![0.0; 16_000], &Device::Cpu)?;
        assert_eq!(features.dims(), &[1, 98, 80]);
        for value in features.flatten_all()?.to_vec1::<f32>()? {
            assert!((value - f32::EPSILON.ln()).abs() < 1e-6);
        }
        assert_eq!(fbank(&[1.0; 559], &Device::Cpu)?.dims(), &[1, 1, 80]);
        assert_eq!(fbank(&[1.0; 560], &Device::Cpu)?.dims(), &[1, 2, 80]);
        Ok(())
    }

    #[test]
    fn rejects_short_and_non_finite_audio() {
        assert!(fbank(&[0.0; 399], &Device::Cpu).is_err());
        assert!(fbank(&[f32::NAN; 400], &Device::Cpu).is_err());
        assert!(fbank(&[f32::INFINITY; 400], &Device::Cpu).is_err());
    }
}
