/// Test: Compare Rust vs Python SoVITS decoder output using same inputs
/// Uses real semantic tokens from a previous GPT run
use candle_core::{DType, Device};
use gpt_sovits_rs::models::sovits::SoVITSModel;
use gpt_sovits_rs::utils::audio_features::SpectrogramExtractor;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SoVITS Decoder Comparison ===\n");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    // Load SoVITS
    let sovits =
        SoVITSModel::load_with_device("models/sovits-model.safetensors", &device, DType::F32)?;
    println!("[OK] SoVITS");

    // Load reference audio
    let ref_audio_path = "/home/ric/gpt-sovits/test_zh.wav";
    let (ref_samples, ref_sr) = load_wav(ref_audio_path)?;
    println!(
        "[OK] Ref audio: {} samples @ {} Hz",
        ref_samples.len(),
        ref_sr
    );

    // Load real semantic tokens from previous GPT run (not argmax - these are varied)
    let semantic_tokens: Vec<usize> = fs::read_to_string("gpt_py_output_tokens.txt")?
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    println!(
        "Loaded {} semantic tokens from previous run",
        semantic_tokens.len()
    );

    // Load real phoneme IDs
    let phoneme_ids: Vec<usize> = fs::read_to_string("gpt_py_phoneme_ids.txt")?
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    println!("Loaded {} phoneme IDs", phoneme_ids.len());

    // Process reference audio - ref_enc needs STFT magnitude (1025 channels)
    let extractor = SpectrogramExtractor::new(32000, 2048, 512, 128);
    let ref_stft = extractor.extract_spectrogram_batched(&ref_samples, &device)?;
    println!("Ref STFT: {:?}", ref_stft.shape());

    // Save intermediates for Python comparison
    let stft_cpu = ref_stft
        .to_device(&Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?;
    let stft_vec: Vec<f32> = stft_cpu.flatten_all()?.to_vec1()?;
    save_f32_file("sovits_ref_stft.txt", &stft_vec);
    println!(
        "Saved sovits_ref_stft.txt ({} elements, shape: [1, 1025, {}])",
        stft_vec.len(),
        stft_vec.len() / 1025
    );

    save_f32_file("sovits_ref_audio.txt", &ref_samples);
    println!("Saved sovits_ref_audio.txt ({} samples)", ref_samples.len());

    // Save tokens
    let tokens_i64: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
    save_i64_file("sovits_semantic_tokens.txt", &tokens_i64);
    let phoneme_i64: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
    save_i64_file("sovits_phoneme_ids.txt", &phoneme_i64);
    println!("Saved tokens");

    // Run SoVITS synthesis with debug output (saves decoder input)
    println!("\nRunning SoVITS synthesis...");
    let (dec_input, audio) =
        sovits.synthesize_debug(&semantic_tokens, &phoneme_ids, Some(&ref_stft), 0.5)?;

    // Save decoder input for Python comparison
    let dec_cpu = dec_input
        .to_device(&Device::Cpu)?
        .to_dtype(candle_core::DType::F32)?;
    let dec_vec: Vec<f32> = dec_cpu.flatten_all()?.to_vec1()?;
    let dec_shape = dec_cpu.dims().to_vec();
    {
        let header = dec_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let data = dec_vec
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write("sovits_dec_input.txt", format!("{}\n{}\n", header, data)).unwrap();
    }
    println!("Saved sovits_dec_input.txt (shape: {:?})", dec_shape);

    // Save Rust audio
    let output_path = "sovits_rust_test.wav";
    save_wav(&audio, output_path, 32000)?;
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    let duration = audio.len() as f32 / 32000.0;
    println!(
        "Saved Rust audio: {} samples, {:.2}s, RMS={:.6}",
        audio.len(),
        duration,
        rms
    );

    // Also save the decoder output (post-tanh) for comparison
    // Save the full audio as text for Python comparison
    save_f32_file("sovits_rust_audio.txt", &audio);
    println!("Saved sovits_rust_audio.txt ({} samples)", audio.len());

    Ok(())
}

fn save_f32_file(path: &str, data: &[f32]) {
    let content = data
        .iter()
        .map(|v| format!("{:.10}", v))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, format!("{}\n", content)).unwrap();
}

fn save_i64_file(path: &str, data: &[i64]) {
    let content = data
        .iter()
        .map(|v| format!("{}", v))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, format!("{}\n", content)).unwrap();
}

fn load_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| v as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((samples, spec.sample_rate))
}

fn save_wav(
    samples: &[f32],
    path: &str,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
