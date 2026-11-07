use anyhow::{ensure, Context, Result};
use gpt_sovits_rs::AudioBuffer;
use lame_sys as lame;
use std::ptr::NonNull;

struct Encoder(NonNull<lame::lame_global_flags>);

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: this context came from lame_init and has one owner.
        unsafe { lame::lame_close(self.0.as_ptr()) };
    }
}

pub(super) fn encode(audio: &AudioBuffer) -> Result<Vec<u8>> {
    ensure!(
        matches!(
            audio.sample_rate,
            8_000 | 11_025 | 12_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000
        ),
        "unsupported MP3 sample rate: {}",
        audio.sample_rate
    );
    // SAFETY: LAME allocates a new context; the RAII owner closes it on every exit.
    let encoder =
        Encoder(NonNull::new(unsafe { lame::lame_init() }).context("LAME allocation failed")?);
    let ptr = encoder.0.as_ptr();
    // SAFETY: the context is live and exclusively owned throughout configuration.
    unsafe {
        ensure!(lame::lame_set_num_channels(ptr, 1) == 0, "LAME channels");
        ensure!(
            lame::lame_set_in_samplerate(ptr, audio.sample_rate as i32) == 0,
            "LAME input rate"
        );
        ensure!(
            lame::lame_set_out_samplerate(ptr, audio.sample_rate as i32) == 0,
            "LAME output rate"
        );
        ensure!(lame::lame_set_brate(ptr, 64) == 0, "LAME bitrate");
        ensure!(lame::lame_set_quality(ptr, 2) == 0, "LAME quality");
        ensure!(lame::lame_set_bWriteVbrTag(ptr, 1) == 0, "LAME gapless tag");
        ensure!(
            lame::lame_init_params(ptr) == 0,
            "LAME initialization failed"
        );
    }

    let mut output = Vec::new();
    // LAME requires at least 1.25 * input_samples + 7200 bytes per encode call.
    let mut buffer = [0u8; 12_320];
    for chunk in audio.samples.chunks(4096) {
        let samples: Vec<f32> = chunk.iter().map(|sample| sample.clamp(-1.0, 1.0)).collect();
        // SAFETY: slices stay alive for the call; both input pointers cover the
        // advertised sample count. The bounded output buffer exceeds LAME's limit.
        // LAME's C API takes const float pointers; lame-sys declares them mut.
        let written = unsafe {
            lame::lame_encode_buffer_ieee_float(
                ptr,
                samples.as_ptr().cast_mut(),
                samples.as_ptr().cast_mut(),
                samples.len() as i32,
                buffer.as_mut_ptr(),
                buffer.len() as i32,
            )
        };
        ensure!(written >= 0, "LAME encoding failed: {written}");
        output.extend_from_slice(&buffer[..written as usize]);
    }
    // SAFETY: the context is live and the buffer exceeds the 7200-byte flush minimum.
    let written = unsafe { lame::lame_encode_flush(ptr, buffer.as_mut_ptr(), buffer.len() as i32) };
    ensure!(written >= 0, "LAME flush failed: {written}");
    output.extend_from_slice(&buffer[..written as usize]);

    // Replace the reserved first frame with delay/padding and duration metadata.
    // SAFETY: LAME receives the actual output capacity and does not retain the pointer.
    let tag_len = unsafe { lame::lame_get_lametag_frame(ptr, buffer.as_mut_ptr(), buffer.len()) };
    ensure!(
        tag_len > 0 && tag_len <= buffer.len() && tag_len <= output.len(),
        "invalid LAME tag length"
    );
    output[..tag_len].copy_from_slice(&buffer[..tag_len]);
    Ok(output)
}
