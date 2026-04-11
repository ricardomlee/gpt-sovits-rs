/// Benchmark comparing GPU inference with and without KV Cache
///
/// Requires CUDA. Compares performance with and without KV cache optimization.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run_benchmark() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU KV Cache Comparison Benchmark ===\n");

    // Force CUDA check
    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    if !cuda_available {
        return Err("CUDA not available. Please install CUDA Toolkit and rebuild with --features cuda".into());
    }
    println!("CUDA available: true");
    println!("Using device: Cuda\n");

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

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

    // --- Without KV Cache ---
    println!("--- Without KV Cache (3 iterations) ---");
    let mut without_kv_times = Vec::new();
    for i in 1..=3 {
        let phoneme_ids = pipeline.text_frontend_mut().process(input_text, options.language)?;
        let bert_features = pipeline.bert_model().as_mut().and_then(|b| b.extract(input_text).ok());
        let hubert_features = pipeline.hubert_model().as_mut().and_then(|h| h.extract(ref_audio).ok());

        let start = Instant::now();
        let gpt = pipeline.gpt_model().as_ref().unwrap();
        let tokens = gpt.generate_with_features(
            &phoneme_ids,
            bert_features.as_ref(),
            hubert_features.as_ref(),
            options.top_k,
            options.top_p,
            options.temperature,
        )?;
        let elapsed = start.elapsed();
        without_kv_times.push(elapsed.as_secs_f64());
        println!("  Iteration {}: {} tokens in {:.2}s", i, tokens.len(), elapsed.as_secs_f64());
    }
    let avg_without_kv = without_kv_times.iter().sum::<f64>() / without_kv_times.len() as f64;

    // --- With KV Cache ---
    println!("\n--- With KV Cache (3 iterations) ---");
    let mut with_kv_times = Vec::new();
    for i in 1..=3 {
        let phoneme_ids = pipeline.text_frontend_mut().process(input_text, options.language)?;
        let bert_features = pipeline.bert_model().as_mut().and_then(|b| b.extract(input_text).ok());
        let hubert_features = pipeline.hubert_model().as_mut().and_then(|h| h.extract(ref_audio).ok());

        let start = Instant::now();
        let gpt = pipeline.gpt_model().as_ref().unwrap();
        let tokens = gpt.generate_with_features_kv_cache(
            &phoneme_ids,
            bert_features.as_ref(),
            hubert_features.as_ref(),
            options.top_k,
            options.top_p,
            options.temperature,
        )?;
        let elapsed = start.elapsed();
        with_kv_times.push(elapsed.as_secs_f64());
        println!("  Iteration {}: {} tokens in {:.2}s", i, tokens.len(), elapsed.as_secs_f64());
    }
    let avg_with_kv = with_kv_times.iter().sum::<f64>() / with_kv_times.len() as f64;

    // Results
    println!("\n=== Results ===");
    println!("Without KV Cache: {:.2}s (average)", avg_without_kv);
    println!("With KV Cache:    {:.2}s (average)", avg_with_kv);

    let speedup = avg_without_kv / avg_with_kv;
    let improvement = (1.0 - avg_with_kv / avg_without_kv) * 100.0;
    println!("\nSpeedup: {:.2}x faster", speedup);
    println!("Improvement: {:.1}% reduction in inference time", improvement);

    Ok(())
}
