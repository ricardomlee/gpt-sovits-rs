/// Benchmark speaker feature caching.
/// Runs inference_kv_cache twice with the same reference — second call should skip
/// HuBERT + BERT + ref_mel and go straight to GPT generation.
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Speaker Cache Benchmark ===\n");

    let config = Config::builder().with_device("cuda").with_half_precision(true).build();
    let mut pipeline = Pipeline::new(config)?;

    let t0 = Instant::now();
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    let _ = pipeline.load_bert("models/onnx/bert.onnx");
    let _ = pipeline.load_hubert("models/onnx/hubert.onnx");
    println!("Model load: {:.2?}\n", t0.elapsed());

    let ref_audio = "/home/ric/gpt-sovits-rs/test_zh_py_wav16k.wav";
    let ref_text = "先帝创业未半而中道崩殂";
    let opts = InferenceOptions::builder()
        .top_k(15).top_p(0.95).temperature(0.8)
        .language(Language::Chinese).max_tokens(500)
        .build();

    // First call — cache miss (HuBERT + BERT + ref_mel all computed)
    let t1 = Instant::now();
    let _ = pipeline.inference_kv_cache("你好，世界！", ref_audio, ref_text, &opts)?;
    let t1e = t1.elapsed();
    println!("1st call (cache miss):  {:.2?}", t1e);

    // Second call — cache hit (only GPT + SoVITS run)
    let t2 = Instant::now();
    let _ = pipeline.inference_kv_cache("今天天气真不错。", ref_audio, ref_text, &opts)?;
    let t2e = t2.elapsed();
    println!("2nd call (cache hit):   {:.2?}", t2e);

    println!("\nCache saved: {:.0}ms ({:.1}%)",
        (t1e.as_secs_f64() - t2e.as_secs_f64()) * 1000.0,
        (1.0 - t2e.as_secs_f64() / t1e.as_secs_f64()) * 100.0,
    );

    Ok(())
}
