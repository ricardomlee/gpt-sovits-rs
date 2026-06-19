//! Audio Buffer utilities

use hound::{WavSpec, WavWriter};
use std::path::Path;
use crate::{Error, Result};

/// Audio buffer representing PCM audio data
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Audio samples (normalized to [-1.0, 1.0])
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Create an empty buffer
    pub fn empty(sample_rate: u32, channels: u16) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
            channels,
        }
    }

    /// Get duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Resample to a new sample rate
    pub fn resample(&self, target_rate: u32) -> Self {
        if self.sample_rate == target_rate {
            return self.clone();
        }

        let ratio = target_rate as f32 / self.sample_rate as f32;
        let new_len = (self.samples.len() as f32 * ratio) as usize;
        let mut new_samples = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = (i as f32 / ratio) as usize;
            if src_idx < self.samples.len() {
                new_samples.push(self.samples[src_idx]);
            }
        }

        Self {
            samples: new_samples,
            sample_rate: target_rate,
            channels: self.channels,
        }
    }

    /// Normalize audio to [-1, 1] range
    pub fn normalize(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let max_amp = self
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        if max_amp > 0.0 {
            let scale = 1.0 / max_amp;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
    }

    /// Apply volume gain
    pub fn apply_gain(&mut self, gain: f32) {
        for sample in &mut self.samples {
            *sample *= gain;
        }
    }

    /// Save to WAV file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let spec = WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path.as_ref(), spec)
            .map_err(|e| Error::AudioError(e.to_string()))?;

        for &sample in &self.samples {
            // Convert f32 [-1, 1] to i16
            let amplitude = i16::MAX as f32;
            let sample_i16 = (sample * amplitude).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| Error::AudioError(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| Error::AudioError(e.to_string()))?;

        Ok(())
    }

    /// Encode as WAV bytes (in-memory, no file I/O)
    pub fn to_wav_bytes(&self) -> Result<Vec<u8>> {
        let spec = WavSpec {
            channels: self.channels,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| Error::AudioError(e.to_string()))?;

        for &sample in &self.samples {
            let amplitude = i16::MAX as f32;
            let sample_i16 = (sample * amplitude).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| Error::AudioError(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| Error::AudioError(e.to_string()))?;

        Ok(cursor.into_inner())
    }

    /// Load from WAV file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = hound::WavReader::open(path.as_ref())
            .map_err(|e| Error::AudioError(e.to_string()))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels;

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                if bits <= 16 {
                    reader
                        .samples::<i16>()
                        .filter_map(|s| s.ok())
                        .map(|s: i16| s as f32 / i16::MAX as f32)
                        .collect()
                } else {
                    reader
                        .samples::<i32>()
                        .filter_map(|s| s.ok())
                        .map(|s: i32| s as f32 / i32::MAX as f32)
                        .collect()
                }
            }
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .filter_map(|s| s.ok())
                .map(|s: f32| s)
                .collect(),
        };

        Ok(Self::new(samples, sample_rate, channels))
    }

    /// Concatenate two audio buffers
    pub fn concat(&mut self, other: &Self) -> Result<()> {
        if self.sample_rate != other.sample_rate {
            return Err(Error::AudioError(
                "Sample rate mismatch".to_string(),
            ));
        }
        if self.channels != other.channels {
            return Err(Error::AudioError(
                "Channel count mismatch".to_string(),
            ));
        }

        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    /// Fade in/out
    pub fn fade_in(&mut self, duration_ms: u32) {
        let fade_samples = (duration_ms as f32 * self.sample_rate as f32 / 1000.0) as usize;
        let fade_samples = fade_samples.min(self.samples.len());

        for i in 0..fade_samples {
            let factor = i as f32 / fade_samples as f32;
            self.samples[i] *= factor;
        }
    }

    pub fn fade_out(&mut self, duration_ms: u32) {
        let fade_samples = (duration_ms as f32 * self.sample_rate as f32 / 1000.0) as usize;
        let fade_samples = fade_samples.min(self.samples.len());

        let start = self.samples.len() - fade_samples;
        for i in 0..fade_samples {
            let factor = 1.0 - (i as f32 / fade_samples as f32);
            self.samples[start + i] *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_duration() {
        let buffer = AudioBuffer::new(vec![0.0; 24000], 24000, 1);
        assert!((buffer.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_buffer_normalize() {
        let mut buffer = AudioBuffer::new(vec![0.5, -0.5, 0.25], 24000, 1);
        buffer.normalize();
        assert!((buffer.samples[0] - 1.0).abs() < 0.01);
        assert!((buffer.samples[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_audio_buffer_resample() {
        let buffer = AudioBuffer::new(vec![1.0, 2.0, 3.0, 4.0], 24000, 1);
        let resampled = buffer.resample(48000);
        assert_eq!(resampled.sample_rate, 48000);
        assert!(resampled.len() >= buffer.len());
    }
}
