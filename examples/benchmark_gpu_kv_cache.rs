/// Benchmark comparing inference() vs inference_kv_cache() on GPU
///
/// Measures wall-clock time for full end-to-end synthesis with and without KV cache.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run_benchmark() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPU KV Cache Benchmark ===\n");

    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    if !cuda_available {
        return Err("CUDA not available".into());
    }

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    let mut pipeline = Pipeline::new(config)?;
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    let _ = pipeline.load_bert("models/onnx/bert.onnx");
    let _ = pipeline.load_hubert("models/onnx/hubert.onnx");
    println!("Models loaded.\n");

    let input_text = "你好，世界！";
    let ref_audio = "/home/ric/gpt-sovits-rs/test_zh_py_wav16k.wav";
    let ref_text = "先帝创业未半而中道崩殂";

    let options = InferenceOptions::builder()
        .top_k(15).top_p(0.95).temperature(0.8)
        .language(Language::Chinese).max_tokens(500)
        .build();

    println!("Input: \"{}\"\n", input_text);

    // --- Without KV Cache ---
    println!("--- inference() — 3 iterations ---");
    let mut times_plain = Vec::new();
    for i in 1..=3 {
        let start = Instant::now();
        let audio = pipeline.inference(input_text, ref_audio, ref_text, &options)?;
        let elapsed = start.elapsed();
        times_plain.push(elapsed.as_secs_f64());
        println!("  Run {}: {:.2}s  ({} samples, {:.2}s audio)",
            i, elapsed.as_secs_f64(), audio.samples.len(), audio.duration());
    }

    // --- With KV Cache ---
    println!("\n--- inference_kv_cache() — 3 iterations ---");
    let mut times_kv = Vec::new();
    for i in 1..=3 {
        let start = Instant::now();
        let audio = pipeline.inference_kv_cache(input_text, ref_audio, ref_text, &options)?;
        let elapsed = start.elapsed();
        times_kv.push(elapsed.as_secs_f64());
        println!("  Run {}: {:.2}s  ({} samples, {:.2}s audio)",
            i, elapsed.as_secs_f64(), audio.samples.len(), audio.duration());
    }

    let avg_plain = times_plain.iter().sum::<f64>() / times_plain.len() as f64;
    let avg_kv    = times_kv.iter().sum::<f64>() / times_kv.len() as f64;

    println!("\n=== Results ===");
    println!("inference():          {:.2}s avg", avg_plain);
    println!("inference_kv_cache(): {:.2}s avg", avg_kv);

    if avg_kv < avg_plain {
        println!("KV cache speedup: {:.2}x", avg_plain / avg_kv);
    } else {
        println!("(no speedup on this input length — try longer text)");
    }

    Ok(())
}
