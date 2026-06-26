/// Benchmark comparing inference() vs inference_kv_cache() on GPU.
///
/// Measures wall-clock time for full end-to-end synthesis with and without KV cache.
/// Uses multiple text lengths to show how speedup scales with sequence length.
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_REF_TEXT: &str = "会战兵力是八十万对六十万，优势在我";

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
    pipeline.load_gpt(model_path(
        "GPT_SOVITS_GPT_MODEL",
        &["models/gpt-model.safetensors"],
    )?)?;
    pipeline.load_sovits(model_path(
        "GPT_SOVITS_SOVITS_MODEL",
        &["models/sovits-model.safetensors"],
    )?)?;
    let _ = pipeline.load_bert(model_path(
        "GPT_SOVITS_BERT_MODEL",
        &["models/bert.safetensors", "models/bert/bert.safetensors"],
    )?);
    let _ = pipeline.load_hubert(model_path(
        "GPT_SOVITS_HUBERT_MODEL",
        &[
            "models/hubert.safetensors",
            "models/hubert/hubert.safetensors",
        ],
    )?);
    println!("Models loaded.\n");

    let ref_audio = ref_audio_path()?;
    let ref_text =
        std::env::var("GPT_SOVITS_REF_TEXT").unwrap_or_else(|_| DEFAULT_REF_TEXT.to_string());

    // Test with different text lengths to show KV cache scaling
    let test_cases: &[(&str, &str)] = &[
        ("short",  "你好世界"),
        ("medium", "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。"),
        ("long",   "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"),
    ];

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(1.0)
        .temperature(1.0)
        .language(Language::Chinese)
        .max_tokens(1000)
        .build();

    println!(
        "{:<8} {:<8} {:<10} {:<10} {:<8}",
        "length", "tokens", "plain(s)", "kv(s)", "speedup"
    );
    println!("{}", "-".repeat(52));

    for (label, input_text) in test_cases {
        // 1 warmup + 2 timed runs each
        let _ = pipeline.inference(input_text, &ref_audio, &ref_text, &options)?;
        let _ = pipeline.inference_kv_cache(input_text, &ref_audio, &ref_text, &options)?;

        let mut t_plain = 0.0f64;
        let mut t_kv = 0.0f64;
        let mut token_count = 0usize;

        for _ in 0..2 {
            let t = Instant::now();
            let audio = pipeline.inference(input_text, &ref_audio, &ref_text, &options)?;
            t_plain += t.elapsed().as_secs_f64();
            token_count = audio.samples.len() / (audio.sample_rate as usize / 25);
        }
        for _ in 0..2 {
            let t = Instant::now();
            let _ = pipeline.inference_kv_cache(input_text, &ref_audio, &ref_text, &options)?;
            t_kv += t.elapsed().as_secs_f64();
        }

        let avg_plain = t_plain / 2.0;
        let avg_kv = t_kv / 2.0;
        let speedup = avg_plain / avg_kv;

        println!(
            "{:<8} {:<8} {:<10.2} {:<10.2} {:.2}x",
            label, token_count, avg_plain, avg_kv, speedup
        );
    }

    println!("\nNote: speedup grows with sequence length (O(n²) vs O(n) attention cost).");
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
