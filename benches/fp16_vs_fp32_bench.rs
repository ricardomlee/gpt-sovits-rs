/// Benchmark FP16 vs FP32 inference on CUDA.
/// Loads models twice (once per dtype) and runs N warm+timed calls each,
/// reporting median latency, throughput (chars/s), and RTF.
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const TEXTS: &[&str] = &[
    "你好，这是一句测试文本。",
    "今天天气晴朗，阳光明媚，适合出门散步。",
    "人工智能技术正在快速发展，深刻改变着人类的生活方式。",
];
const DEFAULT_REF_TEXT: &str = "会战兵力是八十万对六十万，优势在我";
const WARMUP: usize = 1;
const RUNS: usize = 3;

fn make_pipeline(half: bool) -> Result<Pipeline, Box<dyn std::error::Error>> {
    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(half)
        .build();
    let mut p = Pipeline::new(config)?;
    p.load_gpt(model_path(
        "GPT_SOVITS_GPT_MODEL",
        &["models/gpt-model.safetensors"],
    )?)?;
    p.load_sovits(model_path(
        "GPT_SOVITS_SOVITS_MODEL",
        &["models/sovits-model.safetensors"],
    )?)?;
    let _ = p.load_bert(model_path(
        "GPT_SOVITS_BERT_MODEL",
        &["models/bert.safetensors", "models/bert/bert.safetensors"],
    )?);
    let _ = p.load_hubert(model_path(
        "GPT_SOVITS_HUBERT_MODEL",
        &[
            "models/hubert.safetensors",
            "models/hubert/hubert.safetensors",
        ],
    )?);
    Ok(p)
}

fn run_bench(pipeline: &mut Pipeline, label: &str) -> Vec<(usize, Duration, f32)> {
    let ref_audio = ref_audio_path().expect("reference audio not found");
    let ref_text =
        std::env::var("GPT_SOVITS_REF_TEXT").unwrap_or_else(|_| DEFAULT_REF_TEXT.to_string());
    let opts = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .max_tokens(500)
        .build();

    let mut results = Vec::new();

    for text in TEXTS {
        let chars = text.chars().count();

        // Warmup (also populates speaker cache)
        for _ in 0..WARMUP {
            let _ = pipeline.inference_kv_cache(text, &ref_audio, &ref_text, &opts);
        }

        // Timed runs
        let mut times: Vec<Duration> = Vec::new();
        let mut audio_dur_s = 0f32;
        for _ in 0..RUNS {
            let t = Instant::now();
            let audio = pipeline
                .inference_kv_cache(text, &ref_audio, &ref_text, &opts)
                .expect("inference failed");
            times.push(t.elapsed());
            audio_dur_s = audio.samples.len() as f32 / audio.sample_rate as f32;
        }
        times.sort();
        let median = times[times.len() / 2];
        results.push((chars, median, audio_dur_s));

        let rtf = median.as_secs_f32() / audio_dur_s;
        println!(
            "  [{label}] {chars:2}chars  median={:.0}ms  audio={:.2}s  RTF={:.3}",
            median.as_secs_f64() * 1000.0,
            audio_dur_s,
            rtf
        );
    }

    results
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== FP16 vs FP32 Inference Benchmark ===");
    println!(
        "  warmup={WARMUP}  timed_runs={RUNS}  texts={}\n",
        TEXTS.len()
    );

    // --- FP32 ---
    println!("[Loading FP32 pipeline...]");
    let t = Instant::now();
    let mut p32 = make_pipeline(false)?;
    println!("  load: {:.2?}\n", t.elapsed());
    println!("FP32 results:");
    let r32 = run_bench(&mut p32, "F32");
    drop(p32);

    // --- FP16 ---
    println!("\n[Loading FP16 pipeline...]");
    let t = Instant::now();
    let mut p16 = make_pipeline(true)?;
    println!("  load: {:.2?}\n", t.elapsed());
    println!("FP16 results:");
    let r16 = run_bench(&mut p16, "F16");
    drop(p16);

    // --- Summary ---
    println!("\n{:-<60}", "");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "chars", "F32(ms)", "F16(ms)", "speedup", "RTF(F16)"
    );
    println!("{:-<60}", "");
    let mut total_speedup = 0f64;
    for (i, ((chars, t32, _), (_, t16, aud16))) in r32.iter().zip(r16.iter()).enumerate() {
        let speedup = t32.as_secs_f64() / t16.as_secs_f64();
        let rtf16 = t16.as_secs_f32() / aud16;
        total_speedup += speedup;
        println!(
            "{:>10}  {:>10.0}  {:>10.0}  {:>9.2}x  {:>8.3}",
            chars,
            t32.as_secs_f64() * 1000.0,
            t16.as_secs_f64() * 1000.0,
            speedup,
            rtf16,
        );
        let _ = i;
    }
    let avg = total_speedup / r32.len() as f64;
    println!("{:-<60}", "");
    println!("{:>43}{:>9.2}x", "avg speedup: ", avg);

    Ok(())
}

fn model_path(env_key: &str, candidates: &[&str]) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(path) = std::env::var(env_key) {
        return Ok(PathBuf::from(path));
    }
    candidates
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .ok_or_else(|| {
            format!("missing model; set {env_key} or create one of {candidates:?}").into()
        })
}

fn ref_audio_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(path) = std::env::var("GPT_SOVITS_REF_AUDIO") {
        return Ok(PathBuf::from(path));
    }
    ["mao.wav", "ref.wav"]
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .ok_or_else(|| "missing reference audio; set GPT_SOVITS_REF_AUDIO".into())
}
