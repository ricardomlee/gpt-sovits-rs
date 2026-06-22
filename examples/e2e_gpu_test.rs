/// End-to-End GPU Inference Test
///
/// This test runs the complete TTS pipeline on GPU with real audio files.

use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};

fn main() {
    if let Err(e) = run_test() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== End-to-End GPU Inference Test ===\n");

    // Check if CUDA is available
    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    println!("CUDA available: {}", cuda_available);

    if !cuda_available {
        return Err("CUDA not available. Please install CUDA Toolkit and rebuild with --features cuda".into());
    }

    // Initialize pipeline with GPU preference
    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    println!("Using device: {:?}", config.device);
    println!();

    let mut pipeline = Pipeline::new(config)?;
    println!("[OK] Pipeline created");

    // Load models
    let gpt_path = "models/gpt-model.safetensors";
    let sovits_path = "models/sovits-model.safetensors";
    let bigvgan_path = "models/bigvgan.safetensors";
    let bert_path = "models/onnx/bert_hs22.onnx";
    let hubert_path = "models/onnx/hubert.onnx";

    println!("\nLoading models...");

    match pipeline.load_gpt(gpt_path) {
        Ok(_) => println!("[OK] GPT model loaded: {}", gpt_path),
        Err(e) => {
            println!("[SKIP] GPT model not found: {}", e);
            println!("Skipping end-to-end test (models required)");
            return Ok(());
        }
    }

    pipeline.load_sovits(sovits_path)?;
    println!("[OK] SoVITS model loaded: {}", sovits_path);

    pipeline.load_bigvgan(bigvgan_path)?;
    println!("[OK] BigVGAN model loaded: {}", bigvgan_path);

    // Load optional BERT model
    match pipeline.load_bert(bert_path) {
        Ok(_) => println!("[OK] BERT model loaded (ONNX): {}", bert_path),
        Err(e) => println!("[WARN] BERT model not loaded: {}", e),
    }

    // Load optional Hubert model
    match pipeline.load_hubert(hubert_path) {
        Ok(_) => println!("[OK] Hubert model loaded (ONNX): {}", hubert_path),
        Err(e) => println!("[WARN] Hubert model not loaded: {}", e),
    }

    if !pipeline.is_ready() {
        eprintln!("ERROR: Pipeline not ready (GPT and SoVITS required)");
        return Err("Pipeline not ready".into());
    }

    println!("\n[OK] Pipeline ready for inference\n");

    // Test cases
    let test_cases = vec![
        (
            "用户句子测试",
            "如果确实没问题的话，运行一遍完整的流程生成一段音频，文字就是我的这句话",
            "/home/ric/gpt-sovits/test_zh.wav",
            "这是一个测试文本。",
        ),
    ];

    // Match Python TTS defaults (top_k=5, top_p=1.0, temperature=1.0, repetition_penalty=1.35)
    let options = InferenceOptions::builder()
        .top_k(5)
        .top_p(1.0)
        .temperature(1.0)
        .language(Language::Chinese)
        .max_tokens(1500)
        .build();

    for (test_name, input_text, ref_audio, ref_text) in test_cases {
        println!("\n=== {} ===", test_name);
        println!("Input: {}", input_text);
        println!("Reference: {}", ref_audio);

        let start = std::time::Instant::now();

        let audio = pipeline.inference(
            input_text,
            ref_audio,
            ref_text,
            &options,
        )?;

        let duration = start.elapsed();

        println!("[OK] Generated {} samples at {} Hz", audio.samples.len(), audio.sample_rate);
        println!("[OK] Audio duration: {:.2}s", audio.duration());
        println!("[OK] Inference time: {:.2?}", duration);

        // Save to file
        let output_path = format!("out_e2e_test_{}.wav", test_name.chars().filter(|c| c.is_alphanumeric()).collect::<String>());
        save_wav(&audio, &output_path)?;
        println!("[OK] Saved to: {}", output_path);
    }

    println!("\n=== All Tests Complete ===");
    Ok(())
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
