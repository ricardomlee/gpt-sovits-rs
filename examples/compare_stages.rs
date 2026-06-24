//! Save intermediate tensors from Rust pipeline for comparison with Python
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline};
use candle_core::{Device, Tensor};
use std::io::Write;

fn save_npy(name: &str, tensor: &Tensor) {
    let flat: Vec<f32> = tensor.to_dtype(candle_core::DType::F32).unwrap()
        .flatten_all().unwrap().to_vec1().unwrap();
    let dims = tensor.dims();
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
    let std_val: f32 = (flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32).sqrt();
    eprintln!("  [RS] {}: shape={:?}, mean={:.6}, std={:.6}", name, dims, mean, std_val);

    let path = format!("/home/ric/gpt-sovits-rs/debug_rs/{}.npy", name);
    let mut f = std::fs::File::create(&path).unwrap();

    // Write .npy header (numpy format)
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        if dims.len() == 1 {
            format!("({},)", dims[0])
        } else {
            format!("({})", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        }
    );
    // Pad header to be aligned to 64 bytes
    let magic = b"\x93NUMPY";
    let version = [1u8, 0u8];
    let header_len = header.len() + 1; // +1 for newline
    let total = magic.len() + version.len() + 2 + header_len; // 2 for header_len bytes
    let padding = (64 - (total % 64)) % 64;
    let padded_header = format!("{}{}\n", header, " ".repeat(padding));
    let actual_header_len = padded_header.len() as u16;

    f.write_all(magic).unwrap();
    f.write_all(&version).unwrap();
    f.write_all(&actual_header_len.to_le_bytes()).unwrap();
    f.write_all(padded_header.as_bytes()).unwrap();
    for &v in &flat {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn main() -> gpt_sovits_rs::Result<()> {
    std::fs::create_dir_all("/home/ric/gpt-sovits-rs/debug_rs").unwrap();

    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(false)
        .build();

    let mut pipeline = Pipeline::new(config)?;

    eprintln!("Loading models...");
    pipeline.load_gpt("/home/ric/gpt-sovits-rs/models/gpt-model.safetensors")?;
    pipeline.load_sovits("/home/ric/gpt-sovits-rs/models/sovits-model.safetensors")?;
    pipeline.load_bigvgan("/home/ric/gpt-sovits-rs/models/bigvgan.safetensors")?;
    pipeline.load_bert("/home/ric/gpt-sovits-rs/models/onnx/bert.onnx")?;
    pipeline.load_hubert("/home/ric/gpt-sovits-rs/models/onnx/hubert.onnx")?;
    pipeline.load_semantic_tokenizer("/home/ric/gpt-sovits-rs/models/sovits-model.safetensors")?;

    let text = "今天天气真不错";
    let ref_audio = "/home/ric/gpt-sovits/mao.wav";
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    // Stage 1: G2P
    eprintln!("Stage 1: G2P");
    let phone_ids = pipeline.text_frontend_mut().process(text, Language::Chinese)?;
    eprintln!("  phone_ids ({}): {:?}", phone_ids.len(), phone_ids);

    // Stage 2: BERT
    eprintln!("Stage 2: BERT");
    if let Some(bert) = pipeline.bert_model().as_mut() {
        match bert.extract(text) {
            Ok(feats) => {
                let feats = feats.to_device(&device)?;
                save_npy("02_bert_features", &feats);
            }
            Err(e) => eprintln!("  [RS] BERT failed: {}", e),
        }
    }

    // Stage 3: Hubert
    eprintln!("Stage 3: Hubert");
    if let Some(hubert) = pipeline.hubert_model().as_mut() {
        match hubert.extract(ref_audio) {
            Ok(feats) => {
                let feats = feats.to_device(&device)?;
                save_npy("03_hubert_features", &feats);
            }
            Err(e) => eprintln!("  [RS] Hubert failed: {}", e),
        }
    }

    // Stage 5: Full inference
    eprintln!("Stage 5: Full inference");
    let options = InferenceOptions::builder()
        .top_k(15)
        .top_p(0.95)
        .temperature(0.8)
        .language(Language::Chinese)
        .build();

    let audio = pipeline.inference(text, ref_audio, "测试参考音频", &options)?;
    eprintln!("  Audio: {} samples", audio.samples.len());

    let audio_tensor = Tensor::from_vec(audio.samples.clone(), &[audio.samples.len()], &Device::Cpu)?;
    save_npy("05_audio_output", &audio_tensor);

    eprintln!("\nSaved to /home/ric/gpt-sovits-rs/debug_rs/");
    Ok(())
}
