/// Benchmark comparing plain, dynamic KV, and CUDA Graph modes on GPU.
///
/// BERT stays enabled so every mode uses the same quality path.
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::Path;
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

    if std::env::var("GPT_SOVITS_EXPERIMENTAL_CUDA_GRAPH").as_deref() != Ok("1") {
        return Err(
            "CUDA Graph is experimental; set GPT_SOVITS_EXPERIMENTAL_CUDA_GRAPH=1 to benchmark it"
                .into(),
        );
    }

    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    if !cuda_available {
        return Err("CUDA not available".into());
    }

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    let paths = benchmark_model_paths()?;
    let mut pipeline = Pipeline::new(config)?;
    pipeline.load_gpt(&paths.gpt)?;
    pipeline.load_sovits(&paths.sovits)?;
    if let Some(path) = paths.bert.as_ref() {
        pipeline.load_bert(path)?;
    }
    if let Some(path) = paths.hubert.as_ref() {
        pipeline.load_hubert(path)?;
        pipeline.load_semantic_tokenizer(&paths.sovits)?;
    }
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
        .top_k(1)
        .top_p(1.0)
        .temperature(1.0)
        .language(Language::Chinese)
        .max_tokens(300)
        .build();

    println!(
        "{:<8} {:<9} {:<10} {:<10} {:<10} {:<9} {:<9}",
        "length", "audio(s)", "plain(s)", "kv(s)", "graph(s)", "kv/plain", "graph/kv"
    );
    println!("{}", "-".repeat(82));

    for (label, input_text) in test_cases {
        pipeline.preload_speaker(&ref_audio, &ref_text, options.language)?;

        // One warmup per mode, followed by two timed runs.
        for mode in ["plain", "kv", "cuda-graph"] {
            let _ =
                pipeline.inference_with_mode(mode, input_text, &ref_audio, &ref_text, &options)?;
        }

        let mut t_plain = 0.0f64;
        let mut t_kv = 0.0f64;
        let mut t_static = 0.0f64;
        let mut audio_duration = 0.0f32;

        for _ in 0..2 {
            let t = Instant::now();
            let audio = pipeline.inference(input_text, &ref_audio, &ref_text, &options)?;
            t_plain += t.elapsed().as_secs_f64();
            audio_duration = audio.duration();
        }
        for _ in 0..2 {
            let t = Instant::now();
            let _ = pipeline.inference_kv_cache(input_text, &ref_audio, &ref_text, &options)?;
            t_kv += t.elapsed().as_secs_f64();
        }
        for _ in 0..2 {
            let t = Instant::now();
            let _ = pipeline.inference_cuda_graph(input_text, &ref_audio, &ref_text, &options)?;
            t_static += t.elapsed().as_secs_f64();
        }

        let avg_plain = t_plain / 2.0;
        let avg_kv = t_kv / 2.0;
        let avg_static = t_static / 2.0;

        println!(
            "{:<8} {:<9.2} {:<10.2} {:<10.2} {:<10.2} {:>8.2}x {:>8.2}x",
            label,
            audio_duration,
            avg_plain,
            avg_kv,
            avg_static,
            avg_plain / avg_kv,
            avg_kv / avg_static,
        );
    }

    println!(
        "\ngraph(s) uses the same BERT features as plain and dynamic KV modes; validate its audio separately."
    );
    Ok(())
}

fn benchmark_model_paths() -> Result<ModelPaths, Box<dyn std::error::Error>> {
    Ok(ModelPaths::discover(
        Path::new("models"),
        ModelPathOverrides {
            gpt: std::env::var_os("GPT_SOVITS_GPT_MODEL").map(PathBuf::from),
            sovits: std::env::var_os("GPT_SOVITS_SOVITS_MODEL").map(PathBuf::from),
            bert: std::env::var_os("GPT_SOVITS_BERT_MODEL").map(PathBuf::from),
            hubert: std::env::var_os("GPT_SOVITS_HUBERT_MODEL").map(PathBuf::from),
        },
    )?)
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
