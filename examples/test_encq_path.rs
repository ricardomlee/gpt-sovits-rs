/// Test enc_q (reference-driven) synthesis path
use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::utils::SpectrogramExtractor;
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

    let frontend = pipeline.text_frontend_mut();
    let phoneme_ids = frontend.process("你好", Language::Chinese).unwrap();

    let gpt = pipeline.gpt_model().as_ref().unwrap();
    let semantic_tokens = gpt
        .generate_with_features(&phoneme_ids, None, None, 5, 0.9, 0.8, 1.35, 500)
        .unwrap();
    println!("GPT: {} semantic tokens", semantic_tokens.len());

    let ref_audio = load_wav("/home/ric/gpt-sovits/test_zh.wav", 24000).unwrap();
    let device = Device::new_cuda(0).unwrap();
    let mel_extractor = SpectrogramExtractor::new(24000, 2048, 512, 100);
    let ref_mel = mel_extractor
        .extract_spectrogram_batched(&ref_audio, &device)
        .unwrap();
    println!("Ref mel (STFT magnitude): {:?}", ref_mel.dims());

    let sovits = pipeline.sovits_model().as_ref().unwrap();

    // Compute ge from ref_enc
    let time = ref_mel.dims()[2];
    let refer_mask = Tensor::full(1.0f32, &[1, 1, time], &device).unwrap();
    let mel_masked = ref_mel.broadcast_mul(&refer_mask).unwrap();
    let ge = sovits.ref_enc().forward(&mel_masked, &refer_mask).unwrap();
    println!("ge: {:?}, mean={:.6}", ge.dims(), tensor_mean(&ge).unwrap());

    // enc_q: ref_mel → enc_q → (m, logs)
    let (m_q, logs_q, mask_q) = sovits.enc_q().forward(&ref_mel, Some(&ge)).unwrap();
    println!("\n=== enc_q path ===");
    println!(
        "enc_q m: {:?}, mean={:.6}, std={:.6}",
        m_q.dims(),
        tensor_mean(&m_q).unwrap(),
        tensor_std(&m_q).unwrap()
    );
    println!(
        "enc_q logs: {:?}, mean={:.6}, min={:.6}, max={:.6}",
        logs_q.dims(),
        tensor_mean(&logs_q).unwrap(),
        tensor_min(&logs_q).unwrap(),
        tensor_max(&logs_q).unwrap()
    );

    // Sample z_p
    let noise = Tensor::randn(0.0f32, 1.0, m_q.dims(), &device).unwrap();
    let logs_exp = logs_q.exp().unwrap();
    let z_p = m_q
        .add(
            &noise
                .broadcast_mul(&logs_exp)
                .unwrap()
                .broadcast_mul(&Tensor::full(0.5f32, m_q.dims(), &device).unwrap())
                .unwrap(),
        )
        .unwrap();
    println!(
        "z_p: mean={:.6}, std={:.6}",
        tensor_mean(&z_p).unwrap(),
        tensor_std(&z_p).unwrap()
    );

    // Flow inverse
    let z = sovits
        .flow()
        .forward(&z_p, &mask_q, Some(&ge), true)
        .unwrap();
    println!(
        "flow z: mean={:.6}, std={:.6}",
        tensor_mean(&z).unwrap(),
        tensor_std(&z).unwrap()
    );
    let z_masked = z.broadcast_mul(&mask_q).unwrap();

    // Decoder
    let audio = sovits.decoder().forward(&z_masked, Some(&ge)).unwrap();
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!(
        "\nenc_q audio: {} samples ({:.2}s), RMS={:.6}",
        audio.len(),
        audio.len() as f64 / 32000.0,
        rms
    );

    let audio_buf = gpt_sovits_rs::AudioBuffer::new(audio, sovits.sampling_rate(), 1);
    audio_buf.save("out_encq.wav").unwrap();
    println!("Saved to out_encq.wav");

    // Also test enc_p path with reference audio
    println!("\n=== enc_p path (with ref audio) ===");
    let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
    let codes = Tensor::from_vec(codes_vec, (1, semantic_tokens.len()), &device).unwrap();
    let quantized = sovits.quantizer().decode(&codes).unwrap();
    let quantized_up = upsample_2x(&quantized).unwrap();

    let y_lengths = vec![quantized_up.dims()[2] as i64];
    let text_vec: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
    let text = Tensor::from_vec(text_vec, (1, phoneme_ids.len()), &device).unwrap();
    let text_lengths = vec![text.dims()[1] as i64];
    let time_len = quantized_up.dims()[2];
    let y_mask = build_sequence_mask(&y_lengths, time_len, 1, &device).unwrap();

    let (_y, m_p, logs_p, _y_mask_enc) = sovits
        .enc_p()
        .forward(&quantized_up, &y_lengths, &text, &text_lengths, &ge, 1.0)
        .unwrap();
    println!(
        "enc_p m: {:?}, mean={:.6}, std={:.6}",
        m_p.dims(),
        tensor_mean(&m_p).unwrap(),
        tensor_std(&m_p).unwrap()
    );
    println!(
        "enc_p logs: mean={:.6}, min={:.6}, max={:.6}",
        tensor_mean(&logs_p).unwrap(),
        tensor_min(&logs_p).unwrap(),
        tensor_max(&logs_p).unwrap()
    );

    let noise2 = Tensor::randn(0.0f32, 1.0, m_p.dims(), &device).unwrap();
    let logs_exp2 = logs_p.exp().unwrap();
    let z_p2 = m_p
        .add(
            &noise2
                .broadcast_mul(&logs_exp2)
                .unwrap()
                .broadcast_mul(&Tensor::full(0.5f32, m_p.dims(), &device).unwrap())
                .unwrap(),
        )
        .unwrap();

    let z2 = sovits
        .flow()
        .forward(&z_p2, &y_mask, Some(&ge), true)
        .unwrap();
    let z_masked2 = z2.broadcast_mul(&y_mask).unwrap();
    println!(
        "flow z: mean={:.6}, std={:.6}",
        tensor_mean(&z_masked2).unwrap(),
        tensor_std(&z_masked2).unwrap()
    );

    let audio2 = sovits.decoder().forward(&z_masked2, Some(&ge)).unwrap();
    let rms2 = (audio2.iter().map(|s| s * s).sum::<f32>() / audio2.len() as f32).sqrt();
    println!(
        "enc_p audio: {} samples ({:.2}s), RMS={:.6}",
        audio2.len(),
        audio2.len() as f64 / 32000.0,
        rms2
    );

    let audio_buf2 = gpt_sovits_rs::AudioBuffer::new(audio2, sovits.sampling_rate(), 1);
    audio_buf2.save("out_encp_with_ref.wav").unwrap();
    println!("Saved to out_encp_with_ref.wav");
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

fn tensor_std(t: &Tensor) -> candle_core::Result<f32> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let var = flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    Ok(var.sqrt())
}

fn tensor_min(t: &Tensor) -> candle_core::Result<f32> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().fold(f32::INFINITY, |a, &b| a.min(b)))
}

fn tensor_max(t: &Tensor) -> candle_core::Result<f32> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
}

fn load_wav(path: &str, target_sr: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    
    let mut reader = hound::WavReader::open(path)?;
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
