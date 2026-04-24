/// Debug positional encoding: compare Rust PE computation with Python
use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::models::gpt::GPTModel;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let gpt = GPTModel::load_with_device("models/gpt-model.safetensors", &device)?;

    // Load inputs
    let phoneme_ids: Vec<usize> = fs::read_to_string("gpt_py_phoneme_ids.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let prompt_tokens: Vec<usize> = fs::read_to_string("gpt_py_prompts.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();

    let text_seq = phoneme_ids.len();
    let prompt_seq = prompt_tokens.len();

    let text_ids: Vec<i64> = phoneme_ids.iter().map(|&x| x as i64).collect();
    let text_tensor = Tensor::new(text_ids.as_slice(), &device)?.unsqueeze(0)?;
    let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Step 1: Text embedding
    let x_emb = gpt.lookup_text_tokens(&text_tensor, text_seq)?;
    println!("x_emb shape: {:?}", x_emb.dims());
    let x_emb_flat: Vec<f32> = x_emb.flatten_all()?.to_vec1()?;
    println!("x_emb[0,0,:5]: {:?}", &x_emb_flat[..5]);

    // Step 2: BERT projection
    let bert_data: Vec<f32> = fs::read_to_string("gpt_py_bert_feature.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    let bert_flat = Tensor::from_vec(bert_data, (1, 1024, text_seq), &device)?;
    let bert_feature = bert_flat.transpose(1, 2)?;

    let x_emb2 = if let Some((proj_w, proj_b)) = gpt.bert_proj_ref() {
        let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
        let projected = bert_feature.matmul(&proj_w_3d)?;
        let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
        let out = x_emb.broadcast_add(&projected)?;
        println!("After BERT proj [0,0,:5]:");
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        println!("  {:?}", &flat[..5]);
        out
    } else {
        x_emb.clone()
    };

    // Step 3: Text positional encoding
    let x_emb3 = gpt.add_sine_positional_pub(&x_emb2, "text")?;
    println!("After text PE [0,0,:5]:");
    let flat: Vec<f32> = x_emb3.flatten_all()?.to_vec1()?;
    println!("  {:?}", &flat[..5]);

    // Step 4: Audio embedding
    let y_emb = gpt.lookup_audio_tokens(&prompt_tensor, prompt_seq)?;
    println!("y_emb shape: {:?}", y_emb.dims());
    let y_flat: Vec<f32> = y_emb.flatten_all()?.to_vec1()?;
    println!("y_emb[0,0,:5]: {:?}", &y_flat[..5]);

    // Step 5: Audio positional encoding
    let y_pos = gpt.add_sine_positional_pub(&y_emb, "audio")?;
    println!("After audio PE [0,0,:5]:");
    let y_flat: Vec<f32> = y_pos.flatten_all()?.to_vec1()?;
    println!("  {:?}", &y_flat[..5]);

    // Step 6: Concat
    let xy_pos = Tensor::cat(&[&x_emb3, &y_pos], 1)?;
    println!("\nxy_pos shape: {:?}", xy_pos.dims());
    let xy_flat: Vec<f32> = xy_pos.flatten_all()?.to_vec1()?;
    println!("xy_pos[0,0,:5]: {:?}", &xy_flat[..5]);
    println!("xy_pos[0,10,:5]: {:?}", &xy_flat[10*512..10*512+5]);

    // Compare with Python
    let py_xy: Vec<f32> = fs::read_to_string("gpt_py_xy_pos.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();

    // Also load Python PE buffer for comparison
    let py_pe: Vec<f32> = fs::read_to_string("py_pe_full.txt")?
        .lines().map(|l| l.trim().parse().unwrap()).collect();
    println!("\nPython PE buffer shape: [1, 15, 512]");
    println!("Python PE[0,0,:5]: {:?}", &py_pe[..5]);
    println!("Python PE[0,1,:5]: {:?}", &py_pe[512..517]);

    // Compute diff
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..xy_flat.len().min(py_xy.len()) {
        let d = (xy_flat[i] - py_xy[i]).abs();
        sum_diff += d;
        if d > max_diff { max_diff = d; }
    }
    let n = xy_flat.len().min(py_xy.len());
    println!("\n=== xy_pos Comparison ===");
    println!("Rust elements: {}, Python elements: {}", xy_flat.len(), py_xy.len());
    println!("Mean abs diff: {:.2e}", sum_diff / n as f32);
    println!("Max abs diff:  {:.2e}", max_diff);

    // Also compare element by element at key positions
    println!("\nElement-by-element comparison (first 10):");
    for i in 0..10 {
        println!("  [{}]: Rust={:.10}, Python={:.10}, diff={:.2e}",
            i, xy_flat[i], py_xy[i], (xy_flat[i] - py_xy[i]).abs());
    }

    // Check Rust PE computation directly
    // Python pe[0, 0, :5] = [0, 1, 0, 1, 0]
    // Python pe[0, 1, :5] = [0.8415, 0.5403, 0.8219, 0.5697, 0.8020]
    println!("\nExpected Python PE at pos 0: [0, 1, 0, 1, 0]");
    println!("Expected Python PE at pos 1: [0.8415, 0.5403, 0.8219, 0.5697, 0.8020]");

    // Compute what Rust's PE should be (same sinusoidal formula)
    let hidden = 512;
    let half_dim = hidden / 2;
    let div_term: Vec<f64> = (0..half_dim)
        .map(|i| (-((i as f64) * 2.0) * (10000.0f64.ln()) / (hidden as f64)).exp())
        .collect();
    println!("\nRust div_term[0:5]: {:?}", &div_term[..5]);

    // For position 0: all sin(0)=0, cos(0)=1
    println!("Rust PE at pos 0 (computed): all sin(0)=0, cos(0)=1 -> [0, 1, 0, 1, 0]");
    // For position 1
    for pos in 0..=1 {
        let val0 = (pos as f64 * div_term[0]).sin();
        let val1 = (pos as f64 * div_term[0]).cos();
        println!("Rust PE at pos {} dim 0,1: sin={:.4}, cos={:.4}", pos, val0, val1);
    }

    Ok(())
}
