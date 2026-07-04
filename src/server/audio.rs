//! Audio helpers for HTTP responses.

/// Build a streaming WAV header for unknown-length audio (size=0xFFFFFFFF).
/// Compatible with ffplay, mpv, VLC, curl | aplay.
pub(super) fn streaming_wav_header(sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * bits_per_sample as u32 / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_size: u32 = 0xFFFF_FFFE;
    let riff_size: u32 = data_size;

    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&riff_size.to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&byte_rate.to_le_bytes());
    h.extend_from_slice(&block_align.to_le_bytes());
    h.extend_from_slice(&bits_per_sample.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_size.to_le_bytes());
    h
}

/// Encode f32 samples as i16 PCM bytes.
pub(super) fn samples_to_pcm(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_wav_header_encodes_the_generated_audio_format() {
        let header = streaming_wav_header(32_000, 2);

        assert_eq!(header.len(), 44);
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([header[22], header[23]]), 2);
        assert_eq!(
            u32::from_le_bytes([header[24], header[25], header[26], header[27]]),
            32_000
        );
        assert_eq!(
            u32::from_le_bytes([header[28], header[29], header[30], header[31]]),
            128_000
        );
        assert_eq!(u16::from_le_bytes([header[32], header[33]]), 4);
        assert_eq!(u16::from_le_bytes([header[34], header[35]]), 16);
    }

    #[test]
    fn samples_to_pcm_clamps_and_encodes_little_endian_i16() {
        let pcm = samples_to_pcm(&[-2.0, -0.5, 0.0, 0.5, 2.0]);

        let samples: Vec<i16> = pcm
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        assert_eq!(samples, vec![-32767, -16383, 0, 16383, 32767]);
    }
}
