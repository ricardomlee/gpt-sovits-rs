/// Profile inference_kv_cache() end-to-end timing
///
/// Runs inference() and inference_kv_cache() back-to-back for comparison.
use gpt_sovits_rs::{AudioBuffer, Config, InferenceOptions, Language, Pipeline};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::time::Instant;

fn main() {
    if let Err(e) = run_profile() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_profile() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GPT-SoVITS KV Cache Profile ===\n");

    let cuda_available = candle_core::Device::new_cuda(0).is_ok();
    if !cuda_available {
        return Err("CUDA not available".into());
    }

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(false)
        .build();

    let mut pipeline = Pipeline::new(config)?;

    let t0 = Instant::now();
    pipeline.load_gpt("models/gpt-model.safetensors")?;
    pipeline.load_sovits("models/sovits-model.safetensors")?;
    let _ = pipeline.load_bert("models/bert/bert.safetensors");
    let _ = pipeline.load_hubert("models/hubert/hubert.safetensors");
    println!("Model load: {:.2?}\n", t0.elapsed());

    let input_text = "你好，世界！";
    let ref_audio = "ref.wav";
    let ref_text = "先帝创业未半而中道崩殂";

    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .max_tokens(500)
        .build();

    println!("Input: \"{}\"", input_text);
    println!("Ref:   \"{}\"\n", ref_text);

    // inference() baseline
    let t1 = Instant::now();
    let audio1 = pipeline.inference(input_text, ref_audio, ref_text, &options)?;
    let t1e = t1.elapsed();
    let rms1 = rms(&audio1.samples);
    println!(
        "inference():          {:.2?}  {:.2}s audio  RMS={:.4}",
        t1e,
        audio1.duration(),
        rms1
    );
    save_wav(&audio1, "out_profile_plain.wav")?;

    // inference_kv_cache()
    let t2 = Instant::now();
    let audio2 = pipeline.inference_kv_cache(input_text, ref_audio, ref_text, &options)?;
    let t2e = t2.elapsed();
    let rms2 = rms(&audio2.samples);
    println!(
        "inference_kv_cache(): {:.2?}  {:.2}s audio  RMS={:.4}",
        t2e,
        audio2.duration(),
        rms2
    );
    save_wav(&audio2, "out_profile_kv.wav")?;

    println!("\nSaved: out_profile_plain.wav, out_profile_kv.wav");
    Ok(())
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
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
