/// Test Hubert feature fusion in GPT-SoVITS pipeline
use gpt_sovits_rs::{Config, Pipeline, InferenceOptions, Language};

fn main() {
    if let Err(e) = run_test() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Hubert Feature Fusion ===\n");

    // Initialize pipeline with GPU (CUDA) preference
    let config = Config::builder()
        .with_device("cuda")  // Prefer GPU
        .with_half_precision(true)  // Use FP16 for better performance
        .build();

    let mut pipeline = Pipeline::new(config)?;
    println!("[OK] Pipeline created");

    // Load models
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    println!("[OK] GPT model loaded");

    pipeline.load_sovits("models/sovits-model.safetensors")?;
    println!("[OK] SoVITS model loaded");

    pipeline.load_bigvgan("models/bigvgan.safetensors")?;
    println!("[OK] BigVGAN model loaded");

    // Load optional BERT model
    match pipeline.load_bert("models/onnx/bert.onnx") {
        Ok(_) => println!("[OK] BERT model loaded (ONNX)"),
        Err(e) => println!("[WARN] BERT model not loaded: {}", e),
    }

    // Load optional Hubert model
    match pipeline.load_hubert("models/onnx/hubert.onnx") {
        Ok(_) => println!("[OK] Hubert model loaded (ONNX)"),
        Err(e) => println!("[WARN] Hubert model not loaded: {}", e),
    }

    if !pipeline.is_ready() {
        eprintln!("ERROR: Pipeline not ready (GPT and SoVITS required)");
        return Err("Pipeline not ready".into());
    }

    println!("\n[OK] Pipeline ready for inference\n");

    // Test inference with all features
    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .max_tokens(200)
        .build();

    let reference_audio = "/home/ric/gpt-sovits/test_zh.wav";
    let reference_text = "这是一个测试文本。";
    let input_text = "你好，世界！";

    println!("Input text: {}", input_text);
    println!("Reference audio: {}", reference_audio);
    println!("Reference text: {}", reference_text);
    println!("\nRunning inference with BERT + Hubert features...\n");

    let audio = pipeline.inference(
        input_text,
        reference_audio,
        reference_text,
        &options,
    )?;

    println!("[OK] Generated audio: {} samples at {} Hz", audio.samples.len(), audio.sample_rate);
    println!("[OK] Duration: {:.2} seconds", audio.duration());

    // Save to file
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("out_hubert_fusion.wav", spec)?;
    for sample in &audio.samples {
        writer.write_sample((sample * 32767.0) as i16)?;
    }
    writer.finalize()?;

    println!("\n[OK] Audio saved to out_hubert_fusion.wav");
    println!("\n=== Test Complete ===");

    Ok(())
}
