//! CLI Inference Example
//!
//! This example demonstrates how to use the GPT-SoVITS Rust library
//! for command-line TTS inference.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPT-SoVITS Rust Inference Example ===\n");

    // Configuration
    let config = Config::builder()
        .with_device("cpu")
        .with_half_precision(false)
        .build();

    println!("Initializing pipeline...");
    let mut pipeline = Pipeline::new(config)?;

    // Note: In a real application, you would load actual model files:
    // pipeline.load_gpt("models/gpt-model.safetensors")?;
    // pipeline.load_sovits("models/sovits-model.safetensors")?;
    // pipeline.load_bert("models/bert.onnx")?;
    // pipeline.load_hubert("models/hubert.onnx")?;
    // pipeline.load_bigvgan("models/bigvgan.safetensors")?;

    // For this example, we'll show the API usage with placeholder data
    println!("\nAPI Usage Example:");
    println!("------------------");
    println!(r#"
    // Load models
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;

    // Configure inference options
    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .build();

    // Run inference
    let audio = pipeline.inference(
        "你好，这是测试文本",
        "reference.wav",
        "参考文本",
        &options
    )?;

    // Save output
    audio.save("output.wav")?;
    "#);

    // Inference options example
    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .speed(1.0)
        .language(Language::Chinese)
        .max_tokens(500)
        .build();

    println!("\nInference Options:");
    println!("  top_k: {}", options.top_k);
    println!("  top_p: {}", options.top_p);
    println!("  temperature: {}", options.temperature);
    println!("  speed: {}", options.speed);
    println!("  language: {:?}", options.language);
    println!("  max_tokens: {}", options.max_tokens);

    println!("\n=== Example Complete ===");
    println!("\nNote: This example shows the API usage.");
    println!("To run actual inference, download and convert the models first:");
    println!("  python scripts/download_and_convert.py --output-dir models");
    println!("\nThen run with:");
    println!("  cargo run --release -- --gpt-model models/gpt-s1bert.safetensors \\");
    println!("    --sovits-model models/sovits-s2G.safetensors \\");
    println!("    --text '你好世界' \\");
    println!("    --reference-audio ref.wav \\");
    println!("    --reference-text '参考文本' \\");
    println!("    --output output.wav");

    Ok(())
}
