//! Test SoVITS audio generation

use gpt_sovits_rs::utils::SpectrogramExtractor;
use gpt_sovits_rs::AudioBuffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing SoVITS audio generation...");

    // Load SoVITS model
    println!("Loading SoVITS model from models/sovits-model.safetensors...");
    let model = gpt_sovits_rs::models::SoVITSModel::load_with_device(
        "models/sovits-model.safetensors",
        &candle_core::Device::new_cuda(0).expect("CUDA not available"),
    )?;
    println!("Model loaded successfully!");
    println!("  Sampling rate: {} Hz", model.sampling_rate());

    // Test with more tokens (simulating realistic GPT output)
    println!("\n=== Test: enc_p path with 50 tokens ===");
    // Generate more semantic tokens with a pattern that simulates real speech
    let semantic_tokens: Vec<usize> = (0..50).map(|i| 100 + (i % 500)).collect();
    let text_tokens: Vec<usize> = (1..21).collect(); // 20 text tokens

    let audio_samples = model.synthesize(&semantic_tokens, &text_tokens, None, 0.5)?;
    println!(
        "  Generated {} samples ({:.2}s)",
        audio_samples.len(),
        audio_samples.len() as f32 / model.sampling_rate() as f32
    );
    print_audio_stats(&audio_samples);

    let audio_buffer = AudioBuffer::new(audio_samples, model.sampling_rate(), 1);
    audio_buffer.save("test_output_50tok.wav")?;
    println!("  Saved to: test_output_50tok.wav");

    // Test: enc_q with reference audio
    let ref_audio_path = "/home/ric/gpt-sovits/test_zh.wav";
    println!("\n=== Test: enc_q path (reference audio) ===");
    println!("Loading reference audio: {}", ref_audio_path);

    let ref_audio_buf = AudioBuffer::load(ref_audio_path)?;
    println!(
        "  Reference audio: {} samples ({:.2}s) at {} Hz",
        ref_audio_buf.samples.len(),
        ref_audio_buf.duration(),
        ref_audio_buf.sample_rate
    );

    // Resample to model's 24kHz if needed
    let samples_24k = if ref_audio_buf.sample_rate != 24000 {
        let ratio = 24000.0 / ref_audio_buf.sample_rate as f64;
        let new_len = (ref_audio_buf.samples.len() as f64 * ratio) as usize;
        println!(
            "  Resampling from {} Hz to 24000 Hz ({} → {} samples)",
            ref_audio_buf.sample_rate,
            ref_audio_buf.samples.len(),
            new_len
        );
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let src_idx = src_pos.floor() as usize;
            let frac = (src_pos - src_pos.floor()) as f32;
            let v0 = ref_audio_buf.samples.get(src_idx).copied().unwrap_or(0.0);
            let v1 = ref_audio_buf
                .samples
                .get(src_idx + 1)
                .copied()
                .unwrap_or(0.0);
            resampled.push(v0 * (1.0 - frac) + v1 * frac);
        }
        resampled
    } else {
        ref_audio_buf.samples.clone()
    };

    // Extract STFT magnitude spectrum matching Python's spectrogram_torch
    // n_fft=2048, hop=512, center=False, reflect padding
    println!("Extracting STFT magnitude spectrum...");
    let extractor = SpectrogramExtractor::new(24000, 2048, 512, 100);
    let stft_mag =
        extractor.extract_spectrogram_batched(&samples_24k, &candle_core::Device::Cpu)?;
    println!("  STFT magnitude shape: {:?}", stft_mag.dims());
    // Print some stats about STFT magnitude values
    let stft_stats: Vec<f32> = stft_mag.flatten_all()?.to_vec1()?;
    let mean_mag = stft_stats.iter().sum::<f32>() / stft_stats.len() as f32;
    let max_mag = stft_stats.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!(
        "  STFT magnitude - mean: {:.6}, max: {:.6}",
        mean_mag, max_mag
    );

    // Try enc_q with flow (requires proper speaker embedding for good quality)
    match model.synthesize(
        &semantic_tokens,
        &text_tokens,
        Some(&stft_mag.to_device(model.device())?),
        0.5,
    ) {
        Ok(audio_samples_encq) => {
            println!(
                "  Generated {} samples ({:.2}s)",
                audio_samples_encq.len(),
                audio_samples_encq.len() as f32 / model.sampling_rate() as f32
            );
            print_audio_stats(&audio_samples_encq);
            let audio_buffer_encq = AudioBuffer::new(audio_samples_encq, model.sampling_rate(), 1);
            audio_buffer_encq.save("test_output_encq.wav")?;
            println!("  Saved to: test_output_encq.wav");
        }
        Err(e) => {
            println!("  ⚠️ enc_q path failed: {}", e);
            println!(
                "  Note: enc_q requires SSL speaker embedding (512-dim) for proper voice cloning"
            );
        }
    }

    Ok(())
}

fn print_audio_stats(samples: &[f32]) {
    let mut finite_count = 0;
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum_sq = 0.0f64;

    for &sample in samples {
        if sample.is_nan() {
            nan_count += 1;
        } else if sample.is_infinite() {
            inf_count += 1;
        } else {
            finite_count += 1;
            if sample < min {
                min = sample;
            }
            if sample > max {
                max = sample;
            }
            sum_sq += sample as f64 * sample as f64;
        }
    }

    let rms = if finite_count > 0 {
        (sum_sq / finite_count as f64).sqrt() as f32
    } else {
        0.0
    };

    println!(
        "  Finite: {}, NaN: {}, Inf: {}",
        finite_count, nan_count, inf_count
    );
    println!("  Min: {:.6}, Max: {:.6}, RMS: {:.6}", min, max, rms);

    let is_silent = samples
        .iter()
        .filter(|&&x| x.is_finite())
        .all(|&x| x.abs() < 0.01);
    if is_silent {
        println!("  ⚠️ WARNING: Audio appears to be silent!");
    } else {
        println!("  ✓ Audio contains non-silent data");
    }
}
