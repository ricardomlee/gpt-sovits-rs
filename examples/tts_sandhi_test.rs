/// GPU KV cache TTS test with tone sandhi content
///
/// 测试文本覆盖：三声连读（你好/所有/美好/可以/理想）、"不"变调（不要/不是）、
/// "一"变调（一起/一天）、轻声（的/吧/们）
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TTS Sandhi Test (GPU KV Cache) ===\n");

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    let mut pipeline = Pipeline::new(config)?;
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    pipeline.load_bert("models/onnx/bert.onnx")?;
    pipeline.load_hubert("models/onnx/hubert.onnx")?;
    println!("Models loaded.\n");

    // 参考音频：先帝创业未半而中道崩殂
    let ref_audio = "/home/ric/gpt-sovits-rs/test_zh_py_wav16k.wav";
    let ref_text  = "先帝创业未半而中道崩殂";

    // 变调覆盖：
    //   三声连读：你好(3+3)、所有(3+3)、美好(3+3)、可以(3+3)、理想(3+3)
    //   "不"变调：不要(不+4→2)、不是(不+4→2)
    //   "一"变调：一起(一+3→4)、一天(一+1→4)
    //   轻声：的、们、吧
    let input_text = "你好，所有的美好都值得努力争取。\
                      一起加油，不要轻易放弃，\
                      每个人都可以实现自己的理想。\
                      不是做不到，一天一天慢慢来，我们一起加油吧。";

    println!("目标文本（{}字）：", input_text.chars().count());
    println!("  {}\n", input_text);
    println!("变调场景：");
    println!("  三声连读：你好 所有 美好 可以 理想");
    println!("  不变调：  不要(bu2 yao4) 不是(bu2 shi4)");
    println!("  一变调：  一起(yi4 qi3) 一天(yi4 tian1)");
    println!("  轻声：    的 们 吧\n");

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(1.0)
        .temperature(1.0)
        .language(Language::Chinese)
        .max_tokens(1000)
        .build();

    // 先验证 phoneme 变调
    let frontend = gpt_sovits_rs::text_frontend::TextFrontend::new()?;
    let phonemes = frontend.get_phonemes(input_text, Language::Chinese)?;
    println!("Phonemes: {}\n", phonemes);

    // GPU KV cache 推理
    let t = Instant::now();
    let audio = pipeline.inference_kv_cache(input_text, ref_audio, ref_text, &options)?;
    let elapsed = t.elapsed();

    let token_count = audio.samples.len() / (audio.sample_rate as usize / 25);
    let rms = (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();

    println!("=== 结果 ===");
    println!("生成 token 数：{}", token_count);
    println!("音频时长：{:.2}s", audio.duration());
    println!("耗时：{:.2?}", elapsed);
    println!("RMS：{:.4}", rms);

    let out = "out_sandhi_kv.wav";
    save_wav(&audio, out)?;
    println!("\n已保存 → {}", out);

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
    for &s in &audio.samples {
        writer.write_sample((s * 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
