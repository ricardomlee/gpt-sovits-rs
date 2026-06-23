/// Benchmark fused QKV projection vs baseline.
/// The GPT transformer now uses a single [3*hidden, hidden] matmul for Q+K+V
/// instead of 3 separate matmuls. This measures the wall-clock impact.
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::{Duration, Instant};

const TEXTS: &[&str] = &[
    "你好，这是一句测试文本。",
    "今天天气晴朗，阳光明媚，适合出门散步。",
    "人工智能技术正在快速发展，深刻改变着人类的生活方式。",
];
const REF_AUDIO: &str = "/home/ric/gpt-sovits-rs/test_zh_py_wav16k.wav";
const REF_TEXT: &str = "先帝创业未半而中道崩殂";
const WARMUP: usize = 1;
const RUNS: usize = 5;

fn make_pipeline() -> Result<Pipeline, Box<dyn std::error::Error>> {
    let config = Config::builder().with_device("cuda").with_half_precision(false).build();
    let mut p = Pipeline::new(config)?;
    p.load_gpt("models/gpt-model.safetensors")?;
    p.load_sovits("models/sovits-model.safetensors")?;
    let _ = p.load_bert("models/onnx/bert.onnx");
    let _ = p.load_hubert("models/onnx/hubert.onnx");
    Ok(p)
}

fn bench(pipeline: &mut Pipeline) -> Vec<(usize, Duration, f32)> {
    let opts = InferenceOptions::builder()
        .top_k(15).top_p(0.95).temperature(0.8)
        .language(Language::Chinese).max_tokens(500)
        .build();

    TEXTS.iter().map(|text| {
        let chars = text.chars().count();
        for _ in 0..WARMUP {
            let _ = pipeline.inference_kv_cache(text, REF_AUDIO, REF_TEXT, &opts);
        }
        let mut times = Vec::new();
        let mut audio_dur = 0f32;
        for _ in 0..RUNS {
            let t = Instant::now();
            let audio = pipeline.inference_kv_cache(text, REF_AUDIO, REF_TEXT, &opts)
                .expect("inference failed");
            times.push(t.elapsed());
            audio_dur = audio.samples.len() as f32 / audio.sample_rate as f32;
        }
        times.sort();
        let median = times[times.len() / 2];
        (chars, median, audio_dur)
    }).collect()
}

fn main() {
    if let Err(e) = run() { eprintln!("Error: {e}"); std::process::exit(1); }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Fused QKV Benchmark ===");
    println!("  warmup={WARMUP}  runs={RUNS}  texts={}", TEXTS.len());
    println!("  GPT: F32, 16 layers, hidden=512, heads=16\n");

    println!("[Loading pipeline with fused QKV...]");
    let t = Instant::now();
    let mut pipeline = make_pipeline()?;
    println!("  load: {:.2?}\n", t.elapsed());

    println!("Results (fused QKV, single matmul per layer):");
    let results = bench(&mut pipeline);
    for (chars, median, audio_dur) in &results {
        let rtf = median.as_secs_f32() / audio_dur;
        println!("  {chars:2}chars  median={:.0}ms  audio={:.2}s  RTF={rtf:.3}",
            median.as_secs_f64() * 1000.0, audio_dur);
    }

    let avg_ms: f64 = results.iter().map(|(_, t, _)| t.as_secs_f64() * 1000.0).sum::<f64>() / results.len() as f64;
    println!("\n  avg latency: {avg_ms:.0}ms");
    println!("\nNote: compare with baseline by checking git log for pre-fused timings.");
    println!("Fused QKV reduces attention QKV from 3 matmuls → 1 per layer (×16 layers).");

    Ok(())
}
