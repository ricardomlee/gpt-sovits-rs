/// Debug: Save decoder input and compare Rust vs Python decoder output
use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::{Config, Language, Pipeline};

fn main() {
    let config = Config::builder()
        .with_device("cuda")
        .with_half_precision(true)
        .build();

    let mut pipeline = Pipeline::new(config).unwrap();
    pipeline.load_gpt("models/gpt-model.safetensors").unwrap();
    pipeline
        .load_sovits("models/sovits-model.safetensors")
        .unwrap();

    // Text frontend
    let frontend = pipeline.text_frontend_mut();
    let phoneme_ids = frontend.process("你好", Language::Chinese).unwrap();

    // GPT generation
    let gpt = pipeline.gpt_model().as_ref().unwrap();
    let semantic_tokens = gpt
        .generate_with_features(&phoneme_ids, None, None, 5, 0.9, 0.8, 1.35, 500)
        .unwrap();
    println!("GPT: {} semantic tokens", semantic_tokens.len());

    // Load ref audio
    let ref_audio = load_wav("/home/ric/gpt-sovits/test_zh.wav", 24000).unwrap();
    let device = Device::new_cuda(0).unwrap();
    use gpt_sovits_rs::utils::SpectrogramExtractor;
    let mel_extractor = SpectrogramExtractor::new(24000, 2048, 512, 100);
    let ref_mel = mel_extractor
        .extract_spectrogram_batched(&ref_audio, &device)
        .unwrap();
    println!("Ref mel: {:?}", ref_mel.dims());

    // Save ref mel and semantic tokens
    save_tensor("sovits_ref_mel_debug", &ref_mel).unwrap();
    std::fs::write(
        "sovits_semantic_tokens_debug.txt",
        semantic_tokens
            .iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    std::fs::write(
        "sovits_phoneme_ids_debug.txt",
        phoneme_ids
            .iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();

    // Run SoVITS with debug
    let sovits = pipeline.sovits_model().as_ref().unwrap();

    // Prepare inputs like synthesize() does
    let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
    let codes = Tensor::from_vec(codes_vec.clone(), (1, semantic_tokens.len()), &device).unwrap();
    let text_vec: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
    let text = Tensor::from_vec(text_vec.clone(), (1, phoneme_ids.len()), &device).unwrap();

    // Compute ge
    let time = ref_mel.dims()[2];
    let refer_mask = Tensor::full(1.0f32, &[1, 1, time], &device).unwrap();
    let mel_masked = ref_mel.broadcast_mul(&refer_mask).unwrap();
    let ge = sovits.ref_enc().forward(&mel_masked, &refer_mask).unwrap();
    save_tensor("sovits_debug_ge", &ge).unwrap();

    // Quantize and upsample
    let quantized = sovits.quantizer().decode(&codes).unwrap();
    save_tensor("sovits_debug_quantized", &quantized).unwrap();
    let quantized_up = upsample_2x(&quantized).unwrap();
    save_tensor("sovits_debug_quantized_up", &quantized_up).unwrap();

    // EncP forward
    let y_lengths = vec![quantized_up.dims()[2] as i64];
    let text_lengths = vec![text.dims()[1] as i64];
    let time_len = quantized_up.dims()[2];
    let y_mask = build_sequence_mask(&y_lengths, time_len, 1, &device).unwrap();

    let (_y, m_p, logs_p, _y_mask_enc) = sovits
        .enc_p()
        .forward(&quantized_up, &y_lengths, &text, &text_lengths, &ge, 1.0)
        .unwrap();
    save_tensor("sovits_debug_encp_m", &m_p).unwrap();
    save_tensor("sovits_debug_encp_logs", &logs_p).unwrap();

    // Sampling
    let noise = Tensor::randn(0.0f32, 1.0, m_p.dims(), &device).unwrap();
    let logs_exp = logs_p.exp().unwrap();
    let z_p = m_p
        .add(
            &noise
                .broadcast_mul(&logs_exp)
                .unwrap()
                .broadcast_mul(&Tensor::full(0.5f32, m_p.dims(), &device).unwrap())
                .unwrap(),
        )
        .unwrap();
    save_tensor("sovits_debug_zp", &z_p).unwrap();

    // Flow inverse
    let z = sovits
        .flow()
        .forward(&z_p, &y_mask, Some(&ge), true)
        .unwrap();
    save_tensor("sovits_debug_flow_z", &z).unwrap();
    let z_masked = z.broadcast_mul(&y_mask).unwrap();
    save_tensor("sovits_dec_input", &z_masked).unwrap();

    println!(
        "\nDec input: {:?}, mean={:.6}",
        z_masked.dims(),
        tensor_mean(&z_masked).unwrap()
    );

    // Decoder
    let audio = sovits.decoder().forward(&z_masked, Some(&ge)).unwrap();
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!("Rust decoder audio RMS: {:.6}", rms);

    // Save audio
    std::fs::write(
        "sovits_rust_audio.txt",
        audio
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    println!("Saved sovits_rust_audio.txt");
}

fn upsample_2x(x: &Tensor) -> candle_core::Result<Tensor> {
    let dims = x.dims();
    let (batch, channels, time) = (dims[0], dims[1], dims[2]);
    let new_time = time * 2;
    let mut result = Vec::with_capacity(batch * channels * new_time);
    let flat: Vec<f32> = x.flatten_all()?.to_vec1()?;
    for b in 0..batch {
        for c in 0..channels {
            for t in 0..time {
                let idx = b * channels * time + c * time + t;
                let val = flat[idx];
                result.push(val);
                result.push(val);
            }
        }
    }
    Tensor::from_vec(result, (batch, channels, new_time), x.device())
}

fn build_sequence_mask(
    lengths: &[i64],
    time: usize,
    batch: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let indices: Vec<i64> = (0..time as i64).collect();
    let idx_tensor = Tensor::from_vec(indices, (1, 1, time), device)?;
    let len_tensor = Tensor::from_vec(lengths.to_vec(), (batch, 1, 1), device)?;
    let lengths_b = len_tensor.broadcast_as((batch, 1, time))?;
    let mask = idx_tensor.broadcast_lt(&lengths_b)?;
    mask.to_dtype(DType::F32)
}

fn tensor_mean(t: &Tensor) -> candle_core::Result<f32> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().sum::<f32>() / flat.len() as f32)
}

fn save_tensor(name: &str, t: &Tensor) -> std::io::Result<()> {
    let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    let dims = t.dims();
    let header = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let data = flat
        .iter()
        .map(|v| format!("{:.10}", v))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(format!("{}.txt", name), format!("{}\n{}\n", header, data))
}

fn load_wav(path: &str, target_sr: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use hound::WavReader;
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = match spec.bits_per_sample {
                8 => i8::MAX as f32,
                16 => i16::MAX as f32,
                24 => (1 << 23) as f32,
                _ => i16::MAX as f32,
            };
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };
    let samples = if spec.channels > 1 {
        let mut mono = Vec::with_capacity(samples.len() / spec.channels as usize);
        for chunk in samples.chunks(spec.channels as usize) {
            mono.push(chunk.iter().sum::<f32>() / spec.channels as f32);
        }
        mono
    } else {
        samples
    };
    let samples = if spec.sample_rate != target_sr {
        let ratio = target_sr as f64 / spec.sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = vec![0.0f32; new_len];
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx = src_idx.floor() as usize;
            let frac = src_idx - idx as f64;
            let v0 = samples.get(idx).copied().unwrap_or(0.0);
            let v1 = samples.get(idx + 1).copied().unwrap_or(0.0);
            resampled[i] = v0 + (v1 - v0) * frac as f32;
        }
        resampled
    } else {
        samples
    };
    Ok(samples)
}
