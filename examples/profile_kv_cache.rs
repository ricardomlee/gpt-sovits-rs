/// Profile tool for detailed timing analysis with KV Cache optimization
///
/// Compares performance with and without KV Cache.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline, AudioBuffer};
use std::time::Instant;
use hound::{WavSpec, SampleFormat, WavWriter};

fn main() {
    if let Err(e) = run_profile() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

#[derive(Clone)]
struct TimingStats {
    name: &'static str,
    duration_ms: f64,
}

fn run_profile() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPT-SoVITS Pipeline Profiler (with KV Cache) ===\n");

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
    let model_load_start = Instant::now();

    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    pipeline.load_bigvgan("models/bigvgan.safetensors")?;
    let _ = pipeline.load_bert("models/onnx/bert.onnx");
    let _ = pipeline.load_hubert("models/onnx/hubert.onnx");

    let model_load_time = model_load_start.elapsed();
    println!("Model loading: {:.2?}\n", model_load_time);

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

    println!("Running pipeline profile for: \"{}\"\n", input_text);

    let mut timings: Vec<TimingStats> = Vec::new();
    let total_start = Instant::now();

    // Step 1: Text Frontend
    let tf_start = Instant::now();
    let phoneme_ids = pipeline.text_frontend_mut().process(input_text, options.language)?;
    let tf_time = tf_start.elapsed();
    timings.push(TimingStats { name: "1. Text Frontend", duration_ms: tf_time.as_secs_f64() * 1000.0 });
    println!("  [{}] Text Frontend", format_duration(tf_time));

    // Step 2: Load reference audio
    let audio_start = Instant::now();
    let _ref_audio_buffer = AudioBuffer::load(ref_audio)?;
    let audio_time = audio_start.elapsed();
    timings.push(TimingStats { name: "2. Audio Load", duration_ms: audio_time.as_secs_f64() * 1000.0 });
    println!("  [{}] Audio Load", format_duration(audio_time));

    // Step 3: Hubert Feature Extraction
    let hubert_start = Instant::now();
    let hubert_features = pipeline.hubert_model().as_mut().and_then(|hubert| {
        hubert.extract(ref_audio).ok()
    });
    let hubert_time = hubert_start.elapsed();
    timings.push(TimingStats { name: "3. Hubert (ONNX)", duration_ms: hubert_time.as_secs_f64() * 1000.0 });
    println!("  [{}] Hubert (ONNX)", format_duration(hubert_time));

    // Step 4: BERT Feature Extraction
    let bert_start = Instant::now();
    let bert_features = pipeline.bert_model().as_mut().and_then(|bert| {
        bert.extract(input_text).ok()
    });
    let bert_time = bert_start.elapsed();
    timings.push(TimingStats { name: "4. BERT (ONNX)", duration_ms: bert_time.as_secs_f64() * 1000.0 });
    println!("  [{}] BERT (ONNX)", format_duration(bert_time));

    // Step 5: GPT Generation WITH KV CACHE
    let gpt_start = Instant::now();
    let gpt = pipeline.gpt_model().as_ref().unwrap();
    let semantic_tokens = gpt.generate_with_features_kv_cache(
        &phoneme_ids,
        bert_features.as_ref(),
        hubert_features.as_ref(),
        options.top_k,
        options.top_p,
        options.temperature,
    )?;
    let gpt_time = gpt_start.elapsed();
    timings.push(TimingStats { name: "5. GPT (KV Cache)", duration_ms: gpt_time.as_secs_f64() * 1000.0 });
    println!("  [{}] GPT (KV Cache) - {} tokens", format_duration(gpt_time), semantic_tokens.len());

    // Step 6: SoVITS Inference
    let sovits_start = Instant::now();
    let sovits = pipeline.sovits_model().as_ref().unwrap();
    let mel_spec = sovits.synthesize(&semantic_tokens, None)?;
    let sovits_time = sovits_start.elapsed();
    timings.push(TimingStats { name: "6. SoVITS", duration_ms: sovits_time.as_secs_f64() * 1000.0 });
    println!("  [{}] SoVITS", format_duration(sovits_time));

    // Step 7: BigVGAN Vocoder
    let vocoder_start = Instant::now();
    let bigvgan = pipeline.bigvgan_model().as_ref().unwrap();
    let audio_samples = bigvgan.generate(&mel_spec)?;
    let vocoder_time = vocoder_start.elapsed();
    timings.push(TimingStats { name: "7. BigVGAN", duration_ms: vocoder_time.as_secs_f64() * 1000.0 });
    println!("  [{}] BigVGAN", format_duration(vocoder_time));

    let audio = AudioBuffer::new(audio_samples, bigvgan.sampling_rate(), 1);
    let total_time = total_start.elapsed();
    println!("\n  [{}] TOTAL", format_duration(total_time));

    // Print summary
    let total_ms = total_time.as_secs_f64() * 1000.0;
    println!("\n=== Profile Summary ===");
    println!("{:<30} {:>12} {:>8}", "Stage", "Time (ms)", "%");
    println!("{}", "-".repeat(52));
    for timing in &timings {
        let pct = (timing.duration_ms / total_ms) * 100.0;
        println!("{:<30} {:>12.2} {:>7.1}%", timing.name, timing.duration_ms, pct);
    }
    println!("{}", "-".repeat(52));
    println!("{:<30} {:>12.2} {:>7.1}%", "TOTAL", total_ms, 100.0);

    // Bottleneck analysis
    println!("\n=== Bottleneck Analysis ===");
    let mut sorted = timings.clone();
    sorted.sort_by(|a, b| b.duration_ms.partial_cmp(&a.duration_ms).unwrap());

    for (i, t) in sorted.iter().take(3).enumerate() {
        let pct = (t.duration_ms / total_ms) * 100.0;
        println!("  {}. {} ({:.1}%)", i + 1, t.name, pct);
    }

    // Save output
    save_wav(&audio, "out_kv_cache_profile.wav")?;
    println!("\nAudio saved to: out_kv_cache_profile.wav");

    Ok(())
}

fn format_duration(d: std::time::Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms >= 1000.0 {
        format!("{:.2}s", ms / 1000.0)
    } else if ms >= 1.0 {
        format!("{:.2}ms", ms)
    } else {
        format!("{:.2}µs", ms * 1000.0)
    }
}

fn save_wav(audio: &AudioBuffer, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    for sample in &audio.samples {
        writer.write_sample((sample * 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
