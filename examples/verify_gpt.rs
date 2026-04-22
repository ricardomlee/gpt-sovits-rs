/// Verify GPT model forward pass works on GPU

use gpt_sovits_rs::{Config, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPT Model Verification (GPU) ===\n");

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    let mut pipeline = Pipeline::new(config)?;

    println!("Loading models...");
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    println!("  [OK] GPT");
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    println!("  [OK] SoVITS");

    // Try loading optional models
    match pipeline.load_bert("models/onnx/bert.onnx") {
        Ok(_) => println!("  [OK] BERT (ONNX)"),
        Err(e) => println!("  [SKIP] BERT: {}", e),
    }
    match pipeline.load_hubert("models/onnx/hubert.onnx") {
        Ok(_) => println!("  [OK] HuBERT (ONNX)"),
        Err(e) => println!("  [SKIP] HuBERT: {}", e),
    }

    // Step 1: Text frontend
    println!("\n=== Text Frontend ===");
    let frontend = pipeline.text_frontend_mut();
    let phoneme_ids = frontend.process("你好", Language::Chinese)?;
    println!("Input: \"你好\"");
    println!("Phoneme IDs: {} tokens", phoneme_ids.len());

    // Step 2: GPT generation (limited by EOS naturally)
    println!("\n=== GPT Generation ===");
    let start = Instant::now();
    let gpt = pipeline.gpt_model().as_ref().unwrap();
    let semantic_tokens = gpt.generate_with_features(
        &phoneme_ids, None, None, 5, 0.9, 0.8,
    )?;
    let elapsed = start.elapsed();
    println!("Generated {} tokens in {:.2?}", semantic_tokens.len(), elapsed);
    if elapsed.as_secs_f64() > 0.0 {
        println!("Speed: {:.2} tokens/sec", semantic_tokens.len() as f64 / elapsed.as_secs_f64());
    }

    // Step 3: SoVITS WITHOUT reference audio (enc_p path with dummy ge)
    println!("\n=== SoVITS Inference (no reference audio) ===");
    let sovits = pipeline.sovits_model().as_ref().unwrap();
    let start = Instant::now();
    let audio = sovits.synthesize(&semantic_tokens, &phoneme_ids, None, 0.5)?;
    let elapsed = start.elapsed();
    println!("Generated {} samples ({:.2}s) in {:.2?}",
        audio.len(), audio.len() as f64 / sovits.sampling_rate() as f64, elapsed);
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!("Audio RMS: {:.6}", rms);

    let output = "out_gpt_verify_no_ref.wav";
    let audio_buf = gpt_sovits_rs::AudioBuffer::new(audio, sovits.sampling_rate(), 1);
    audio_buf.save(output)?;
    println!("Saved to: {}", output);

    // Step 4: SoVITS WITH reference audio (enc_p + speaker embedding from enc_q)
    let ref_audio = "/home/ric/gpt-sovits/test_zh.wav";
    if std::path::Path::new(ref_audio).exists() {
        println!("\n=== SoVITS Inference (with reference audio) ===");

        // Extract reference mel via pipeline
        use candle_core::Device;
        let device = sovits.device();
        let audio_data = load_wav(ref_audio, 24000)?;
        use gpt_sovits_rs::utils::SpectrogramExtractor;
        let mel_extractor = SpectrogramExtractor::new(24000, 2048, 512, sovits.n_mels());
        let ref_mel = mel_extractor.extract_spectrogram_batched(&audio_data, device)?;
        println!("Reference STFT magnitude: {:?}", ref_mel.dims());

        let start = Instant::now();
        let audio = sovits.synthesize(&semantic_tokens, &phoneme_ids, Some(&ref_mel), 0.5)?;
        let elapsed = start.elapsed();
        println!("Generated {} samples ({:.2}s) in {:.2?}",
            audio.len(), audio.len() as f64 / sovits.sampling_rate() as f64, elapsed);
        let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
        println!("Audio RMS: {:.6}", rms);

        let output = "out_gpt_verify_with_ref.wav";
        let audio_buf = gpt_sovits_rs::AudioBuffer::new(audio, sovits.sampling_rate(), 1);
        audio_buf.save(output)?;
        println!("Saved to: {}", output);
    } else {
        println!("\n[SKIP] Reference audio not found: {}", ref_audio);
    }

    println!("\n[OK] Full pipeline verified: GPT → SoVITS → Audio");
    Ok(())
}

fn load_wav(path: &str, target_sr: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound::WavReader;
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = match spec.bits_per_sample {
                8 => i8::MAX as f32,
                16 => i16::MAX as f32,
                24 => (1 << 23) as f32,
                _ => i16::MAX as f32,
            };
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    // Convert to mono if stereo
    let samples = if spec.channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / spec.channels as usize);
        for chunk in samples.chunks(spec.channels as usize) {
            mono.push(chunk.iter().sum::<f32>() / spec.channels as f32);
        }
        mono
    } else {
        samples
    };

    // Resample if needed
    let samples = if spec.sample_rate != target_sr {
        let ratio = target_sr as f64 / spec.sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = vec![0.0f32; new_len];
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx = src_idx.floor() as usize;
            let frac = src_idx - idx as f64;
            let v0 = samples.get(idx).copied().unwrap_or(0.0);
            let v1 = samples.get(idx + 1).copied().unwrap_or(0.0);
            resampled[i] = v0 + (v1 - v0) * frac as f32;
        }
        resampled
    } else {
        samples
    };

    Ok(samples)
}
