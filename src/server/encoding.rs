//! Buffered speech encoders. Codec work runs outside the async executor.

use anyhow::{ensure, Result};
use gpt_sovits_rs::AudioBuffer;

use super::{audio::samples_to_pcm, response::SpeechOutputFormat};

mod mp3;
mod opus;

pub(super) struct EncodedSpeech {
    pub bytes: Vec<u8>,
    pub sample_rate: u32,
}

pub(super) fn encode(audio: &AudioBuffer, format: SpeechOutputFormat) -> Result<EncodedSpeech> {
    ensure!(audio.channels == 1, "speech encoding requires mono audio");
    ensure!(audio.sample_rate > 0, "invalid audio sample rate");
    ensure!(!audio.samples.is_empty(), "cannot encode empty audio");
    ensure!(
        audio.samples.iter().all(|sample| sample.is_finite()),
        "cannot encode non-finite audio samples"
    );
    let (bytes, sample_rate) = match format {
        SpeechOutputFormat::Wav => (audio.to_wav_bytes()?, audio.sample_rate),
        SpeechOutputFormat::Pcm => (samples_to_pcm(&audio.samples), audio.sample_rate),
        SpeechOutputFormat::Mp3 => (mp3::encode(audio)?, audio.sample_rate),
        SpeechOutputFormat::Opus => (opus::encode(audio)?, 48_000),
    };
    Ok(EncodedSpeech { bytes, sample_rate })
}

#[cfg(test)]
mod tests;
