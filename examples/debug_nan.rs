/// Debug NaN in SoVITS with reference audio

use gpt_sovits_rs::models::SoVITSModel;
use gpt_sovits_rs::utils::SpectrogramExtractor;
use candle_core::{Device, Tensor, DType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0).expect("CUDA");
    let model = SoVITSModel::load_with_device("models/sovits-model.safetensors", &device)?;

    let semantic_tokens: Vec<usize> = (0..50).map(|i| 100 + (i % 500)).collect();
    let text_tokens: Vec<usize> = (1..21).collect();

    // Load reference audio and extract mel
    let audio_data = load_wav("/home/ric/gpt-sovits/test_zh.wav", 24000)?;
    let extractor = SpectrogramExtractor::new(24000, 2048, 512, 100);
    let ref_mel = extractor.extract_spectrogram_batched(&audio_data, &Device::Cpu)?;
    let ref_mel = ref_mel.to_device(&device)?;

    println!("Reference mel shape: {:?}", ref_mel.dims());

    // Step 1: Prepare tensors
    let codes_vec: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
    let codes = Tensor::from_vec(codes_vec, (1, semantic_tokens.len()), &device)?;
    let text_vec: Vec<i64> = text_tokens.iter().map(|&x| x as i64).collect();
    let text = Tensor::from_vec(text_vec, (1, text_tokens.len()), &device)?;

    // Step 2: Compute ge via ref_enc
    let mel_in = ref_mel.clone();
    let time = mel_in.dims()[2];
    let refer_mask = Tensor::full(1.0f32, &[1, 1, time], &device)?;
    let mel_masked = mel_in.broadcast_mul(&refer_mask)?;

    println!("Computing ge from ref_enc...");
    let ge = model.ref_enc().forward(&mel_masked, &refer_mask)?;
    println!("ge shape: {:?}", ge.dims());
    print_tensor_stats(&ge, "ge")?;

    // Step 3: Quantizer decode
    let quantized = model.quantizer().decode(&codes)?;
    print_tensor_stats(&quantized, "quantized")?;

    // Step 4: Upsample
    let quantized_up = nearest_upsample_2x(&quantized)?;
    print_tensor_stats(&quantized_up, "quantized_up")?;

    // Step 5: enc_p forward
    let y_lengths = vec![quantized_up.dims()[2] as i64];
    let text_lengths = vec![text.dims()[1] as i64];

    let (_y, m_p, logs_p, y_mask) = model.enc_p().forward(
        &quantized_up, &y_lengths, &text, &text_lengths, &ge, 1.0,
    )?;
    print_tensor_stats(&m_p, "m_p")?;
    print_tensor_stats(&logs_p, "logs_p")?;

    // Step 6: Sample z_p
    let logs_p_clamped = logs_p.clamp(-4.0, 4.0)?;
    let logs_exp = logs_p_clamped.exp()?;
    let noise = Tensor::randn(0.0f32, 1.0, m_p.dims(), &device)?;
    let z_p = m_p.add(&noise.broadcast_mul(&logs_exp)?)?;
    print_tensor_stats(&z_p, "z_p")?;

    // Step 7: Flow reverse
    let z = model.flow().forward(&z_p, &y_mask, Some(&ge), true)?;
    print_tensor_stats(&z, "z (flow output)")?;

    let z_masked = z.broadcast_mul(&y_mask)?;
    print_tensor_stats(&z_masked, "z_masked")?;

    // Step 8: Decoder
    let output = model.decoder().forward(&z_masked, Some(&ge))?;
    // output is Vec<f32>, not Tensor
    let nan_count = output.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = output.iter().filter(|&&x| x.is_infinite()).count();
    let mean = output.iter().copied().sum::<f32>() / output.len() as f32;
    let max = output.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let min = output.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    println!("decoder output: {} samples, mean={:.4}, min={:.4}, max={:.4}, NaN={}, Inf={}",
        output.len(), mean, min, max, nan_count, inf_count);
    println!("NaN count: {}, Inf count: {}", nan_count, inf_count);

    Ok(())
}

fn print_tensor_stats(t: &Tensor, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let finite: Vec<f32> = flat.iter().filter(|x| x.is_finite()).copied().collect();
    let nan_count = flat.iter().filter(|x| x.is_nan()).count();
    let inf_count = flat.iter().filter(|x| x.is_infinite()).count();
    if finite.is_empty() {
        println!("{}: ALL NaN/Inf (NaN={}, Inf={})", name, nan_count, inf_count);
    } else {
        let mean = finite.iter().sum::<f32>() / finite.len() as f32;
        let max = finite.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let min = finite.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        println!("{}: shape={:?}, mean={:.4}, min={:.4}, max={:.4}, NaN={}, Inf={}",
            name, t.dims(), mean, min, max, nan_count, inf_count);
    }
    Ok(())
}

fn nearest_upsample_2x(x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
    let dims = x.dims();
    let batch = dims[0];
    let channels = dims[1];
    let time = dims[2];
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

    Ok(Tensor::from_vec(result, (batch, channels, new_time), x.device())?)
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
            reader.samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader.samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
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
