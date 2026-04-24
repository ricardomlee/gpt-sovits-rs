/// Debug: Compare layer 0 intermediate outputs between Python and Rust
use candle_core::{Device, Tensor};
use gpt_sovits_rs::models::gpt::GPTModel;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Layer 0 Debug: Python vs Rust ===\n");

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    // Load model
    let gpt = GPTModel::load_with_device("models/gpt-model.safetensors", &device)?;

    // Load inputs
    let phoneme_ids: Vec<usize> = fs::read_to_string("gpt_py_phoneme_ids.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let bert_data: Vec<f32> = fs::read_to_string("gpt_py_bert_feature.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let prompt_tokens: Vec<usize> = fs::read_to_string("gpt_py_prompts.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();

    let text_seq = phoneme_ids.len();
    let prompt_seq = prompt_tokens.len();

    let text_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
    let text_tensor = Tensor::new(text_ids.as_slice(), &device)?.unsqueeze(0)?;
    let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let bert_flat = Tensor::from_vec(bert_data, (1, 1024, phoneme_ids.len()), &device)?;
    let bert_feature = bert_flat.transpose(1, 2)?;

    // Build xy_pos (same as Python)
    let x_emb = gpt.lookup_text_tokens(&text_tensor, text_seq)?;

    // BERT projection
    let x_emb = if let Some((proj_w, proj_b)) = gpt.bert_proj_ref() {
        let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
        let projected = bert_feature.matmul(&proj_w_3d)?;
        let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
        x_emb.broadcast_add(&projected)?
    } else {
        x_emb
    };
    let x_emb = gpt.add_sine_positional_pub(&x_emb, "text")?;

    let y_emb = gpt.lookup_audio_tokens(&prompt_tensor, prompt_seq)?;
    let y_pos = gpt.add_sine_positional_pub(&y_emb, "audio")?;
    let xy_pos = Tensor::cat(&[&x_emb, &y_pos], 1)?;

    println!("xy_pos shape: {:?}", xy_pos.dims());

    // Compare with Python xy_pos
    let py_xy_pos_data: Vec<f32> = fs::read_to_string("gpt_py_xy_pos.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let py_xy_pos = Tensor::from_vec(py_xy_pos_data, (1, 15, 512), &device)?;
    let diff = (&xy_pos - &py_xy_pos)?;
    let flat_diff: Vec<f32> = diff.flatten_all()?.to_vec1()?;
    let mean_diff = flat_diff.iter().map(|v| v.abs()).sum::<f32>() / flat_diff.len() as f32;
    println!("xy_pos match Python: {:.2e}", mean_diff);

    // Create hybrid attention mask (same as Python)
    let total_seq = text_seq + prompt_seq;
    let mask = create_hybrid_mask(text_seq, total_seq, &device)?;

    // Save layer 0 intermediates
    println!("\n--- Layer 0 Intermediates ---");
    let (attn_out, norm1_out, linear1_out, relu_out, final_out) =
        gpt.debug_layer0_intermediates(&xy_pos, &mask)?;

    save_tensor("rust_layer0_attn.txt", &attn_out.flatten_all()?.to_vec1()?)?;
    save_tensor("rust_layer0_norm1.txt", &norm1_out.flatten_all()?.to_vec1()?)?;
    save_tensor("rust_layer0_linear1.txt", &linear1_out.flatten_all()?.to_vec1()?)?;
    save_tensor("rust_layer0_relu.txt", &relu_out.flatten_all()?.to_vec1()?)?;
    save_tensor("rust_layer0_final.txt", &final_out.flatten_all()?.to_vec1()?)?;

    // Compare with Python layer 0 intermediates
    for name in &["attn", "norm1", "linear1", "relu", "final"] {
        let rust_data: Vec<f32> = fs::read_to_string(&format!("rust_layer0_{}.txt", name))?
            .lines().map(|l| l.trim().parse().unwrap()).collect();
        let py_data: Vec<f32> = fs::read_to_string(&format!("py_layer0_{}.txt", name))?
            .lines().map(|l| l.trim().parse().unwrap()).collect();
        let d = elementwise_diff(&rust_data, &py_data);
        println!("  {}: mean_diff={:.2e}, max_diff={:.2e}", name, d.mean, d.max);
    }

    // Run full transformer with hybrid mask
    println!("\n--- Full Transformer (hybrid mask) ---");
    let hidden = gpt.run_transformer_with_mask(&xy_pos, &mask)?;
    let hidden_flat: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
    println!("Rust transformer output: mean={:.6}, std={:.6}",
        mean(&hidden_flat), std_dev(&hidden_flat));
    save_tensor("rust_transformer_out.txt", &hidden_flat)?;

    let py_hidden_data: Vec<f32> = fs::read_to_string("py_transformer_out.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let td = elementwise_diff(&hidden_flat, &py_hidden_data);
    println!("  Transformer mean diff: {:.2e}, max diff: {:.2e}", td.mean, td.max);

    // Compute logits
    let last_hidden = hidden.narrow(1, xy_pos.dims()[1] - 1, 1)?.squeeze(0)?;
    let ar_predict_layer = gpt.ar_predict_layer_ref()?;
    let logits = last_hidden.matmul(&ar_predict_layer.t()?)?.squeeze(0)?;
    let logits_flat: Vec<f32> = logits.flatten_all()?.to_vec1()?;
    println!("\nRust logits: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
        mean(&logits_flat), std_dev(&logits_flat),
        logits_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::min),
        logits_flat.iter().cloned().fold(f32::INFINITY, f32::max));
    save_tensor("rust_logits.txt", &logits_flat)?;

    let py_logits: Vec<f32> = fs::read_to_string("py_logits.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let ld = elementwise_diff(&logits_flat, &py_logits);
    println!("\n--- Logits Comparison ---");
    println!("  Mean abs diff: {:.2e}", ld.mean);
    println!("  Max abs diff:  {:.2e}", ld.max);

    // Top 10 comparison
    let mut rust_top: Vec<(usize, f32)> = logits_flat.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut py_top: Vec<(usize, f32)> = py_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    rust_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    py_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nRust top 5:");
    for (idx, val) in &rust_top[..5] { println!("  token {}: {:.4}", idx, val); }
    println!("Python top 5:");
    for (idx, val) in &py_top[..5] { println!("  token {}: {:.4}", idx, val); }

    Ok(())
}

fn create_hybrid_mask(text_seq: usize, total_seq: usize, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut mask = vec![0.0f32; total_seq * total_seq];
    for i in text_seq..total_seq {
        for j in text_seq..total_seq {
            if j > i {
                mask[i * total_seq + j] = 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(mask, (total_seq, total_seq), device)?)
}

#[derive(Debug)]
struct DiffStats {
    mean: f32,
    max: f32,
}

fn elementwise_diff(a: &[f32], b: &[f32]) -> DiffStats {
    let len = a.len().min(b.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..len {
        let d = (a[i] - b[i]).abs();
        sum_diff += d;
        if d > max_diff {
            max_diff = d;
        }
    }
    DiffStats { mean: sum_diff / len as f32, max: max_diff }
}

fn mean(data: &[f32]) -> f32 {
    data.iter().sum::<f32>() / data.len() as f32
}

fn std_dev(data: &[f32]) -> f32 {
    let m = mean(data);
    let var = data.iter().map(|x| (x - m).powi(2)).sum::<f32>() / data.len() as f32;
    var.sqrt()
}

fn save_tensor(path: &str, data: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let content = data.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n");
    fs::write(path, format!("{}\n", content))?;
    println!("  Saved {} ({} elements)", path, data.len());
    Ok(())
}
