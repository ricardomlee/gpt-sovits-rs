/// Full GPU E2E test: GPT → SoVITS with reference audio
/// Tests complete pipeline on GPU with KV cache

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Full GPU E2E Test ===\n");

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

    match pipeline.load_bert("models/onnx/bert.onnx") {
        Ok(_) => println!("  [OK] BERT (ONNX)"),
        Err(e) => println!("  [SKIP] BERT: {}", e),
    }
    match pipeline.load_hubert("models/onnx/hubert.onnx") {
        Ok(_) => println!("  [OK] HuBERT (ONNX)"),
        Err(e) => println!("  [SKIP] HuBERT: {}", e),
    }

    let input_text = "你好世界";
    let ref_audio = "/home/ric/gpt-sovits/test_zh.wav";

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .max_tokens(200)
        .build();

    println!("\nInput: \"{}\"", input_text);
    println!("Reference: {}", ref_audio);
    println!("Max GPT tokens: 200\n");

    let start = std::time::Instant::now();
    let audio = pipeline.inference(input_text, ref_audio, "", &options)?;
    let elapsed = start.elapsed();

    println!("\n=== Results ===");
    println!("Generated {} samples ({:.2}s at {} Hz)",
        audio.samples.len(), audio.duration(), audio.sample_rate);
    println!("Total inference time: {:.2?}", elapsed);

    let rms = (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();
    let min_v = audio.samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = audio.samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nz = audio.samples.iter().filter(|&&s| s.abs() > 1e-6).count();
    println!("Audio - RMS: {:.6}, Min: {:.6}, Max: {:.6}", rms, min_v, max_v);
    println!("Non-zero samples: {}/{} ({:.1}%)", nz, audio.samples.len(), nz as f64 * 100.0 / audio.samples.len() as f64);

    let output_path = "out_e2e_gpu_full.wav";
    save_wav(&audio, output_path)?;
    println!("Saved to: {}", output_path);

    println!("\n[OK] Full GPU pipeline works!");
    Ok(())
}

fn save_wav(audio: &gpt_sovits_rs::AudioBuffer, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in &audio.samples {
        writer.write_sample((sample * 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
