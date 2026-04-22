//! Audio feature extraction utilities
//!
//! Provides spectrogram extraction from raw audio samples,
//! used for processing reference audio in voice cloning.
//!
//! STFT extraction matches the Python GPT-SoVITS implementation:
//! - center=False with reflect padding: pad((n_fft-hop)/2, (n_fft-hop)/2)
//! - Hann window
//! - Magnitude: sqrt(re^2 + im^2 + 1e-8)

use candle_core::{Device, Tensor};
use rustfft::{FftPlanner, num_complex::Complex};
use crate::{Error, Result};

/// Spectrogram extractor matching Python's spectrogram_torch
pub struct SpectrogramExtractor {
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    n_mels: usize,
    sample_rate: u32,
    mel_basis: Vec<f32>,
    window: Vec<f32>,
}

impl SpectrogramExtractor {
    /// Create with GPT-SoVITS default parameters
    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> Self {
        let mel_basis = create_mel_basis(sample_rate as f32, n_fft, n_mels);
        let window = create_hann_window(n_fft);

        Self {
            n_fft,
            hop_length,
            win_length: n_fft,
            n_mels,
            sample_rate,
            mel_basis,
            window,
        }
    }

    /// Extract mel spectrogram: STFT → magnitude → mel filterbank → log compression
    /// Returns [n_mels, n_frames] tensor
    pub fn extract(&self, audio: &[f32], device: &Device) -> Result<Tensor> {
        let spec = self.extract_spectrogram(audio, device)?;

        // Apply mel filterbank: [n_mels, n_bins] @ [n_bins, n_frames] → [n_mels, n_frames]
        let n_bins = self.n_fft / 2 + 1;
        let n_frames = spec.dims()[1];
        let spec_flat: Vec<f32> = spec.flatten_all()?.to_vec1()?;

        let mut mel_spec = vec![0.0f32; self.n_mels * n_frames];
        for mel_idx in 0..self.n_mels {
            for frame_idx in 0..n_frames {
                let mut sum = 0.0f32;
                for bin in 0..n_bins {
                    sum += self.mel_basis[mel_idx * n_bins + bin] * spec_flat[frame_idx * n_bins + bin];
                }
                mel_spec[mel_idx * n_frames + frame_idx] = sum;
            }
        }

        // Log compression: log(max(mel, 1e-5))
        for val in &mut mel_spec {
            *val = (*val).max(1e-5).ln();
        }

        Tensor::from_vec(mel_spec, (self.n_mels, n_frames), device)
            .map_err(|e| Error::AudioError(format!("Failed to create mel spectrogram tensor: {}", e)))
    }

    /// Extract mel spectrogram as [1, n_mels, time] for model input
    pub fn extract_batched(&self, audio: &[f32], device: &Device) -> Result<Tensor> {
        let mel = self.extract(audio, device)?;
        mel.unsqueeze(0).map_err(|e| Error::AudioError(format!("Failed to batch mel spectrogram: {}", e)))
    }

    /// Extract raw STFT magnitude spectrum [n_bins, time]
    /// Matches Python's spectrogram_torch exactly:
    /// - Reflect padding: pad((n_fft-hop)/2, (n_fft-hop)/2)
    /// - Hann window, center=False
    /// - Magnitude: sqrt(re^2 + im^2 + 1e-8)
    pub fn extract_spectrogram(&self, audio: &[f32], device: &Device) -> Result<Tensor> {
        let n_fft = self.n_fft;
        let hop = self.hop_length;
        let win_len = self.win_length;

        // Reflect padding: pad((n_fft-hop)/2, (n_fft-hop)/2)
        let pad = (n_fft - hop) / 2;
        let padded = reflect_pad(audio, pad);

        let n_frames = (padded.len() - n_fft) / hop + 1;
        let n_bins = n_fft / 2 + 1;
        let mut magnitudes = vec![0.0f32; n_frames * n_bins];

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut fft_buf = vec![Complex::new(0.0f32, 0.0); n_fft];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;

            // Apply Hann window and copy to FFT buffer
            for i in 0..n_fft {
                if i < win_len && start + i < padded.len() {
                    fft_buf[i] = Complex::new(padded[start + i] * self.window[i], 0.0);
                } else {
                    fft_buf[i] = Complex::new(0.0, 0.0);
                }
            }

            // FFT
            fft.process(&mut fft_buf);

            // Compute magnitude: sqrt(re^2 + im^2 + 1e-8)  (matches Python exactly)
            for bin in 0..n_bins {
                let re = fft_buf[bin].re;
                let im = fft_buf[bin].im;
                magnitudes[frame_idx * n_bins + bin] = (re * re + im * im + 1e-8).sqrt();
            }
        }

        // Create tensor: [n_bins, n_frames]
        Tensor::from_vec(magnitudes, (n_bins, n_frames), device)
            .map_err(|e| Error::AudioError(format!("Failed to create spectrogram tensor: {}", e)))
    }

    /// Extract STFT magnitude as [1, n_bins, time] for model input
    pub fn extract_spectrogram_batched(&self, audio: &[f32], device: &Device) -> Result<Tensor> {
        let spec = self.extract_spectrogram(audio, device)?;
        spec.unsqueeze(0).map_err(|e| Error::AudioError(format!("Failed to batch spectrogram: {}", e)))
    }
}

/// Reflect padding: mirrors the signal at boundaries
/// Python: F.pad(y.unsqueeze(1), (pad, pad), mode="reflect")
fn reflect_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    let len = audio.len();
    if len <= pad {
        // Fallback to zero padding if audio is too short
        let mut buf = vec![0.0f32; pad + len + pad];
        buf[pad..pad + len].copy_from_slice(audio);
        return buf;
    }

    let mut buf = Vec::with_capacity(pad + len + pad);

    // Left reflect: mirror from [pad..0]
    for i in (0..pad).rev() {
        buf.push(audio[i + 1]);
    }

    // Original audio
    buf.extend_from_slice(audio);

    // Right reflect: mirror from [len-2..len-1-pad]
    for i in (0..pad).rev() {
        buf.push(audio[len - 2 - i]);
    }

    buf
}

/// Backwards compatibility alias
#[deprecated(since = "0.2.0", note = "use SpectrogramExtractor instead")]
pub type MelExtractor = SpectrogramExtractor;

/// Create Hann window
fn create_hann_window(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (length - 1) as f64).cos() as f32)
        })
        .collect()
}

/// Create mel filterbank matrix
fn create_mel_basis(sample_rate: f32, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_min = 0.0f32;
    let f_max = sample_rate / 2.0;

    // Convert Hz to mel and back
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel center points (evenly spaced in mel domain)
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| (hz * n_fft as f32 / sample_rate).floor())
        .collect();

    // Create filterbank
    let mut mel_basis = vec![0.0f32; n_mels * n_bins];

    for m in 1..=n_mels {
        let prev = bin_points[m - 1] as usize;
        let center = bin_points[m] as usize;
        let next = bin_points[m + 1] as usize;

        // Rising edge
        for bin in prev..center.min(n_bins) {
            let weight = if center > prev {
                (bin as f32 - prev as f32) / (center as f32 - prev as f32)
            } else {
                0.0
            };
            mel_basis[(m - 1) * n_bins + bin] = weight;
        }

        // Falling edge
        for bin in center..next.min(n_bins) {
            let weight = if next > center {
                (next as f32 - bin as f32) / (next as f32 - center as f32)
            } else {
                0.0
            };
            mel_basis[(m - 1) * n_bins + bin] = weight;
        }
    }

    mel_basis
}
