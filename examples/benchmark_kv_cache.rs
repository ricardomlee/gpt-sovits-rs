/// Benchmark comparing GPT inference with and without KV cache
///
/// This demonstrates the performance improvement from KV cache optimization.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run_benchmark() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== KV Cache Benchmark ===\n");

    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    println!("CUDA available: {}", cuda_available);

    if !cuda_available {
        return Err("CUDA not available. Please install CUDA Toolkit and rebuild with --features cuda".into());
    }

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    println!("Using device: {:?}\n", config.device);

    let mut pipeline = Pipeline::new(config)?;

    // Load models
    println!("Loading models...");
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    pipeline.load_bigvgan("models/bigvgan.safetensors")?;
    let _ = pipeline.load_bert("models/onnx/bert.onnx");
    let _ = pipeline.load_hubert("models/onnx/hubert.onnx");

    if !pipeline.is_ready() {
        return Err("Pipeline not ready".into());
    }

    println!("Models loaded.\n");

    // Test input
    let input_text = "你好，世界！";
    let ref_audio = "/home/ric/gpt-sovits/test_zh.wav";
    let ref_text = "这是一个测试文本。";

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .max_tokens(500)
        .build();

    println!("Benchmarking GPT generation for: \"{}\"\n", input_text);

    // Run benchmark WITHOUT KV cache (3 iterations)
    println!("--- Without KV Cache (3 iterations) ---");
    let mut without_cache_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();

        // Use standard inference (no KV cache)
        let _audio = pipeline.inference(input_text, ref_audio, ref_text, &options)?;

        let duration = start.elapsed();
        without_cache_times.push(duration);
        println!("  Iteration {}: {:.2}s", i + 1, duration.as_secs_f64());
    }
    let avg_without_cache = without_cache_times.iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>() / without_cache_times.len() as f64;

    // Run benchmark WITH KV cache (3 iterations)
    println!("\n--- With KV Cache (3 iterations) ---");
    let mut with_cache_times = Vec::new();
    for i in 0..3 {
        let start = Instant::now();

        // Use KV cache optimized inference
        let _audio = pipeline.inference_kv_cache(input_text, ref_audio, ref_text, &options)?;

        let duration = start.elapsed();
        with_cache_times.push(duration);
        println!("  Iteration {}: {:.2}s", i + 1, duration.as_secs_f64());
    }
    let avg_with_cache = with_cache_times.iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>() / with_cache_times.len() as f64;

    // Print results
    println!("\n=== Results ===");
    println!("Without KV Cache: {:.2}s (average)", avg_without_cache);
    println!("With KV Cache:    {:.2}s (average)", avg_with_cache);

    let speedup = avg_without_cache / avg_with_cache;
    let improvement = (avg_without_cache - avg_with_cache) / avg_without_cache * 100.0;

    println!("\nSpeedup: {:.2}x faster", speedup);
    println!("Improvement: {:.1}% reduction in inference time", improvement);

    Ok(())
}
