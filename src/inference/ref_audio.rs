//! Reference-audio feature extraction.

use crate::utils::SpectrogramExtractor;
use crate::{Error, Result};
use candle_core::{Device, Tensor};
use std::path::Path;

pub(super) fn extract_ref_mel(
    ref_audio: &Path,
    device: &Device,
    sovits_sr: u32,
    sovits_n_mels: usize,
) -> Result<Option<Tensor>> {
    use hound::WavReader;

    let mut reader = WavReader::open(ref_audio)
        .map_err(|e| Error::AudioError(format!("Failed to open reference audio: {}", e)))?;

    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = match spec.bits_per_sample {
                8 => i8::MAX as f32,
                16 => i16::MAX as f32,
                24 => (1 << 23) as f32,
                _ => i16::MAX as f32,
            };
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    let samples = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|c| c.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples
    };

    let samples = if spec.sample_rate != sovits_sr {
        use soxr::{format::Mono, Soxr};
        let ratio = sovits_sr as f64 / spec.sample_rate as f64;
        let out_cap = (samples.len() as f64 * ratio).ceil() as usize + 64;
        let mut output = vec![0.0f32; out_cap];
        let mut resampler = Soxr::<Mono<f32>>::new(spec.sample_rate as f64, sovits_sr as f64)
            .map_err(|e| Error::AudioError(format!("soxr init: {}", e)))?;
        let proc = resampler
            .process(&samples, &mut output)
            .map_err(|e| Error::AudioError(format!("soxr process: {}", e)))?;
        let mut tail = vec![0.0f32; out_cap];
        let tail_n = resampler
            .drain(&mut tail)
            .map_err(|e| Error::AudioError(format!("soxr drain: {}", e)))?;
        output.truncate(proc.output_frames);
        output.extend_from_slice(&tail[..tail_n]);
        output
    } else {
        samples
    };

    let n_fft = 2048;
    let hop_length = 640;
    let extractor = SpectrogramExtractor::new(sovits_sr, n_fft, hop_length, sovits_n_mels);
    let stft_mag = extractor.extract_spectrogram_batched(&samples, device)?;
    let stft_mag = stft_mag.narrow(1, 0, 704)?;

    tracing::info!("Extracted reference STFT magnitude: {:?}", stft_mag.dims());
    Ok(Some(stft_mag))
}
