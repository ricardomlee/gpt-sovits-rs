/// Profile tool for detailed timing analysis of GPT-SoVITS pipeline
///
/// This tool measures each stage of the TTS pipeline to identify bottlenecks.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run_profile() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

struct TimingStats {
    name: &'static str,
    duration_ms: f64,
    percentage: f64,
}

impl Clone for TimingStats {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            duration_ms: self.duration_ms,
            percentage: self.percentage,
        }
    }
}

fn run_profile() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPT-SoVITS Pipeline Profiler ===\n");

    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    println!("CUDA available: {}", cuda_available);

    let config = Config::builder()
        .with_device(if cuda_available { "cuda" } else { "cpu" })
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
    let _ref_text = "这是一个测试文本。";

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .max_tokens(500)
        .build();

    println!("Running pipeline profile for: \"{}\"\n", input_text);

    // Manual step-by-step profiling
    let mut timings: Vec<TimingStats> = Vec::new();
    let total_start = Instant::now();

    // Step 1: Text Frontend
    let tf_start = Instant::now();
    let phoneme_ids = pipeline.text_frontend_mut().process(input_text, options.language)?;
    let tf_time = tf_start.elapsed();
    timings.push(TimingStats {
        name: "1. Text Frontend (phoneme conversion)",
        duration_ms: tf_time.as_secs_f64() * 1000.0,
        percentage: 0.0,
    });
    println!("  [{}] {}", format_duration(tf_time), "Text Frontend (phoneme conversion)");

    // Step 2: Load reference audio
    let audio_start = Instant::now();
    let ref_audio_buffer = gpt_sovits_rs::AudioBuffer::load(ref_audio)?;
    let audio_time = audio_start.elapsed();
    timings.push(TimingStats {
        name: "2. Reference Audio Loading",
        duration_ms: audio_time.as_secs_f64() * 1000.0,
        percentage: 0.0,
    });
    println!("  [{}] {}", format_duration(audio_time), "Reference Audio Loading");

    // Step 3: Hubert Feature Extraction
    let hubert_start = Instant::now();
    let hubert_features = pipeline.hubert_model().as_mut().and_then(|hubert| {
        // Hubert extract needs file path, use the original ref_audio path
        hubert.extract(ref_audio).ok()
    });
    let hubert_time = hubert_start.elapsed();
    timings.push(TimingStats {
        name: "3. Hubert Feature Extraction (ONNX)",
        duration_ms: hubert_time.as_secs_f64() * 1000.0,
        percentage: 0.0,
    });
    println!("  [{}] {}", format_duration(hubert_time), "Hubert Feature Extraction (ONNX)");

    // Step 4: BERT Feature Extraction
    let bert_start = Instant::now();
    let bert_features = pipeline.bert_model().as_mut().and_then(|bert| {
        bert.extract(input_text).ok()
    });
    let bert_time = bert_start.elapsed();
    timings.push(TimingStats {
        name: "4. BERT Feature Extraction (ONNX)",
        duration_ms: bert_time.as_secs_f64() * 1000.0,
        percentage: 0.0,
    });
    println!("  [{}] {}", format_duration(bert_time), "BERT Feature Extraction (ONNX)");

    // Step 5: GPT Generation (with features)
    let gpt_start = Instant::now();
    let gpt = pipeline.gpt_model().as_ref().unwrap();
    let semantic_tokens = gpt.generate_with_features(
        &phoneme_ids,
        bert_features.as_ref(),
        hubert_features.as_ref(),
        options.top_k,
        options.top_p,
        options.temperature,
    )?;
    let gpt_time = gpt_start.elapsed();
    timings.push(TimingStats {
        name: "5. GPT Generation (semantic tokens)",
        duration_ms: gpt_time.as_secs_f64() * 1000.0,
        percentage: 0.0,
    });
    println!("  [{}] {}", format_duration(gpt_time), "GPT Generation (semantic tokens)");

    // Step 6: SoVITS Inference (includes decoder → audio)
    let sovits_start = Instant::now();
    let sovits = pipeline.sovits_model().as_ref().unwrap();
    let audio_samples = sovits.synthesize(&semantic_tokens, &[], None, 0.5)?;
    let sovits_time = sovits_start.elapsed();
    timings.push(TimingStats {
        name: "6. SoVITS Inference (with decoder)",
        duration_ms: sovits_time.as_secs_f64() * 1000.0,
        percentage: 0.0,
    });
    println!("  [{}] {}", format_duration(sovits_time), "SoVITS Inference (with decoder)");

    // Convert samples to AudioBuffer
    let audio = gpt_sovits_rs::AudioBuffer::new(audio_samples, sovits.sampling_rate(), 1);

    let total_time = total_start.elapsed();
    println!("\n  [{}] TOTAL", format_duration(total_time));

    // Calculate percentages
    let total_ms = total_time.as_secs_f64() * 1000.0;
    for timing in &mut timings {
        timing.percentage = (timing.duration_ms / total_ms) * 100.0;
    }

    // Print summary
    println!("\n=== Profile Summary ===");
    println!("{:<45} {:>12} {:>8}", "Stage", "Time (ms)", "%");
    println!("{}", "-".repeat(67));
    for timing in &timings {
        println!("{:<45} {:>12.2} {:>7.1}%", timing.name, timing.duration_ms, timing.percentage);
    }
    println!("{}", "-".repeat(67));
    println!("{:<45} {:>12.2} {:>7.1}%", "TOTAL", total_ms, 100.0);

    // Identify bottlenecks
    println!("\n=== Bottleneck Analysis ===");
    let mut sorted = timings.clone();
    sorted.sort_by(|a, b| b.duration_ms.partial_cmp(&a.duration_ms).unwrap());

    println!("Top bottlenecks:");
    for (i, timing) in sorted.iter().take(3).enumerate() {
        println!("  {}. {} ({:.1}%)", i + 1, timing.name, timing.percentage);
    }

    println!("\n=== Optimization Recommendations ===");
    for timing in &sorted {
        if timing.percentage > 50.0 {
            println!("  - {}: Consider optimization or offloading", timing.name);
        }
    }

    // Save audio output
    let output_path = "out_profile.wav";
    save_wav(&audio, output_path)?;
    println!("\nAudio saved to: {}", output_path);

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

fn save_wav(audio: &gpt_sovits_rs::AudioBuffer, path: &str) -> Result<(), Box<dyn std::error::Error>> {
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
