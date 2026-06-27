/// Full GPU E2E test: GPT → SoVITS with reference audio
/// Tests complete pipeline on GPU with KV cache
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Full GPU E2E Test ===\n");

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    let mut pipeline = Pipeline::new(config)?;
    let paths = ModelPaths::discover(Path::new("models"), ModelPathOverrides::default())?;

    println!("Loading models...");
    pipeline.load_gpt(&paths.gpt)?;
    println!("  [OK] GPT");
    pipeline.load_sovits(&paths.sovits)?;
    println!("  [OK] SoVITS");

    match paths.bert.as_ref().map(|path| pipeline.load_bert(path)) {
        Some(Ok(_)) => println!("  [OK] BERT"),
        Some(Err(e)) => println!("  [SKIP] BERT: {}", e),
        None => println!("  [SKIP] BERT: model not found"),
    }
    match paths.hubert.as_ref().map(|path| pipeline.load_hubert(path)) {
        Some(Ok(_)) => {
            pipeline.load_semantic_tokenizer(&paths.sovits)?;
            println!("  [OK] HuBERT + semantic tokenizer");
        }
        Some(Err(e)) => println!("  [SKIP] HuBERT: {}", e),
        None => println!("  [SKIP] HuBERT: model not found"),
    }

    let input_text = "你好世界";
    let ref_audio = "mao.wav";
    let ref_text = "会战兵力是八十万对六十万，优势在我";

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(1.0)
        .temperature(1.0)
        .language(Language::Chinese)
        .max_tokens(200)
        .build();

    println!("\nInput: \"{}\"", input_text);
    println!("Reference: {}", ref_audio);
    println!("Ref text: \"{}\"", ref_text);
    println!("Max GPT tokens: 200\n");

    // Test both paths and compare
    let start = std::time::Instant::now();
    let audio = pipeline.inference(input_text, ref_audio, ref_text, &options)?;
    let t1 = start.elapsed();

    let start2 = std::time::Instant::now();
    let audio_kv = pipeline.inference_kv_cache(input_text, ref_audio, ref_text, &options)?;
    let t2 = start2.elapsed();

    println!("\n=== Results ===");
    println!(
        "inference()         {:>4} tokens, {:.2}s audio, {:.2?} time",
        audio.samples.len() / (audio.sample_rate as usize / 25),
        audio.duration(),
        t1
    );
    println!(
        "inference_kv_cache() {:>4} tokens, {:.2}s audio, {:.2?} time",
        audio_kv.samples.len() / (audio_kv.sample_rate as usize / 25),
        audio_kv.duration(),
        t2
    );

    let rms1 =
        (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();
    let rms2 = (audio_kv.samples.iter().map(|s| s * s).sum::<f32>()
        / audio_kv.samples.len() as f32)
        .sqrt();
    println!("RMS: inference={:.4}, inference_kv_cache={:.4}", rms1, rms2);

    let output_path = "out_e2e_gpu_full.wav";
    save_wav(&audio, output_path)?;
    println!("Saved inference() → {}", output_path);

    let output_path_kv = "out_e2e_kv.wav";
    save_wav(&audio_kv, output_path_kv)?;
    println!("Saved inference_kv_cache() → {}", output_path_kv);

    println!("\n[OK] Full GPU pipeline works!");
    Ok(())
}

fn save_wav(
    audio: &gpt_sovits_rs::AudioBuffer,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in &audio.samples {
        writer.write_sample((sample * 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
