//! Reference-audio feature extraction.

use crate::utils::SpectrogramExtractor;
use crate::{AudioBuffer, Error, Result};
use candle_core::{Device, Tensor};
use std::path::Path;

pub(super) fn extract_ref_mel(
    ref_audio: &Path,
    device: &Device,
    sovits_sr: u32,
    sovits_n_mels: usize,
) -> Result<Option<Tensor>> {
    let audio = AudioBuffer::load(ref_audio)?;
    if audio.is_empty() {
        return Err(Error::AudioError(
            "Reference audio must contain samples".into(),
        ));
    }
    let samples = if audio.channels > 1 {
        audio
            .samples
            .chunks(audio.channels as usize)
            .map(|c| c.iter().sum::<f32>() / audio.channels as f32)
            .collect()
    } else {
        audio.samples
    };

    let samples = if audio.sample_rate != sovits_sr {
        use soxr::{format::Mono, Soxr};
        let ratio = sovits_sr as f64 / audio.sample_rate as f64;
        let out_cap = (samples.len() as f64 * ratio).ceil() as usize + 64;
        let mut output = vec![0.0f32; out_cap];
        let mut resampler = Soxr::<Mono<f32>>::new(audio.sample_rate as f64, sovits_sr as f64)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm32_and_float_reference_spectra_match() {
        let pcm = tempfile::NamedTempFile::new().unwrap();
        let float = tempfile::NamedTempFile::new().unwrap();
        for (path, format) in [
            (pcm.path(), hound::SampleFormat::Int),
            (float.path(), hound::SampleFormat::Float),
        ] {
            let mut writer = hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 32000,
                    bits_per_sample: 32,
                    sample_format: format,
                },
            )
            .unwrap();
            for sample in [1i32 << 29, -(1 << 29), 0, 0]
                .into_iter()
                .cycle()
                .take(4096)
            {
                match format {
                    hound::SampleFormat::Int => writer.write_sample(sample).unwrap(),
                    hound::SampleFormat::Float => writer
                        .write_sample(sample as f32 / i32::MAX as f32)
                        .unwrap(),
                }
            }
            writer.finalize().unwrap();
        }
        let extract = |path: &Path| {
            extract_ref_mel(path, &Device::Cpu, 32000, 704)
                .unwrap()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(extract(pcm.path()), extract(float.path()));
    }
}
