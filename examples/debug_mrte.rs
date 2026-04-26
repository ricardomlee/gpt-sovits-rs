/// Debug MRTE cross-attention intermediates
use candle_core::{DType, Device, Module, Tensor};
use gpt_sovits_rs::utils::{load_safetensors, StateDict};
use std::collections::HashMap;

fn load_tensor_file(path: &str) -> Result<(Vec<usize>, Vec<f32>), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.trim().split('\n').collect();
    let dims: Vec<usize> = lines[0].split(',').map(|d| d.parse().unwrap()).collect();
    let data: Vec<f32> = lines[1..]
        .iter()
        .map(|s| s.trim().parse().unwrap())
        .collect();
    Ok((dims, data))
}

fn layer_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let x_perm = x.transpose(1, 2)?;
    let mean = x_perm.mean_keepdim(candle_core::D::Minus1)?;
    let centered = x_perm.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let eps = Tensor::full(1e-5f32, var.dims(), x.device())?;
    Ok(centered
        .broadcast_div(&var.add(&eps)?.sqrt()?)?
        .broadcast_mul(gamma)?
        .broadcast_add(beta)?
        .transpose(1, 2)?)
}

fn load_conv1d(
    sd: &StateDict,
    device: &Device,
    prefix: &str,
) -> Result<candle_nn::Conv1d, Box<dyn std::error::Error>> {
    let w = sd
        .get(&format!("{}.weight", prefix))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let b = sd
        .get(&format!("{}.bias", prefix))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let kernel_size = w.dims()[2];
    let padding = (kernel_size - 1) / 2;
    let config = candle_nn::Conv1dConfig {
        padding,
        stride: 1,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: Default::default(),
    };
    Ok(candle_nn::Conv1d::new(w, Some(b), config))
}

fn mean_val(t: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().sum::<f32>() / flat.len() as f32)
}

fn run_encoder_layer(
    x: &Tensor,
    mask: &Tensor,
    sd: &StateDict,
    device: &Device,
    prefix: &str,
    layer_idx: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let gamma1 = sd
        .get(&format!("{}.norm_layers_1.{}.gamma", prefix, layer_idx))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let beta1 = sd
        .get(&format!("{}.norm_layers_1.{}.beta", prefix, layer_idx))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let gamma2 = sd
        .get(&format!("{}.norm_layers_2.{}.gamma", prefix, layer_idx))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let beta2 = sd
        .get(&format!("{}.norm_layers_2.{}.beta", prefix, layer_idx))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let conv_q = load_conv1d(
        sd,
        device,
        &format!("{}.attn_layers.{}.conv_q", prefix, layer_idx),
    )?;
    let conv_k = load_conv1d(
        sd,
        device,
        &format!("{}.attn_layers.{}.conv_k", prefix, layer_idx),
    )?;
    let conv_v = load_conv1d(
        sd,
        device,
        &format!("{}.attn_layers.{}.conv_v", prefix, layer_idx),
    )?;
    let conv_o = load_conv1d(
        sd,
        device,
        &format!("{}.attn_layers.{}.conv_o", prefix, layer_idx),
    )?;
    let ff1 = load_conv1d(
        sd,
        device,
        &format!("{}.ffn_layers.{}.conv_1", prefix, layer_idx),
    )?;
    let ff2 = load_conv1d(
        sd,
        device,
        &format!("{}.ffn_layers.{}.conv_2", prefix, layer_idx),
    )?;

    let x_norm1 = layer_norm(x, &gamma1, &beta1)?;
    let q = conv_q.forward(&x_norm1)?;
    let k = conv_k.forward(&x_norm1)?;
    let v = conv_v.forward(&x_norm1)?;

    let batch = 1;
    let channels = q.dims()[1];
    let seq_len = q.dims()[2];
    let n_heads: usize = 8;
    let head_dim = channels / n_heads;

    let q_h = q.reshape((batch, n_heads, seq_len, head_dim))?;
    let k_h = k.reshape((batch, n_heads, seq_len, head_dim))?;
    let v_h = v.reshape((batch, n_heads, seq_len, head_dim))?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k_h.transpose(2, 3)?;
    let raw_scores = q_h.matmul(&k_t)?;
    let scores =
        raw_scores
            .broadcast_mul(&Tensor::full(scale as f32, raw_scores.dims(), device)?)?;

    let mask_2d = mask.reshape((1, seq_len))?;
    let mask_4d = mask_2d.unsqueeze(1)?.unsqueeze(2)?;
    let mask_bc = mask_4d.broadcast_as((batch, n_heads, seq_len, seq_len))?;
    let neg_inf = Tensor::full(-1e9f32, mask_bc.dims(), device)?;
    let ones = Tensor::ones(mask_bc.dims(), DType::F32, device)?;
    let inv_mask = ones.sub(&mask_bc)?.broadcast_mul(&neg_inf)?;
    let masked = scores.broadcast_mul(&mask_bc)?.add(&inv_mask)?;

    let attn_probs = candle_nn::ops::softmax(&masked, candle_core::D::Minus1)?;
    let attn_out_pre = attn_probs
        .matmul(&v_h)?
        .reshape((batch, channels, seq_len))?;
    let attn_out = conv_o.forward(&attn_out_pre)?;
    let x = x.add(&attn_out)?;

    let x_norm2 = layer_norm(&x, &gamma2, &beta2)?;
    let ffn1 = ff1.forward(&x_norm2)?;
    let ffn_gelu = ffn1.gelu()?;
    let ffn2 = ff2.forward(&ffn_gelu)?;
    let x = x.add(&ffn2)?;
    Ok(x.broadcast_mul(mask)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    let weights_map = load_safetensors("models/sovits-model.safetensors")?;
    let sd = StateDict::new(weights_map);

    let (dims, data) = load_tensor_file("encp_debug_quantized_up.txt")?;
    let quantized = Tensor::from_vec(data, dims.clone(), &device)?.to_dtype(DType::F32)?;
    let y_len = dims[2] as i64;
    let y_mask_data: Vec<f32> = (0..y_len).map(|_| 1.0f32).collect();
    let y_mask =
        Tensor::from_vec(y_mask_data, (1, y_len as usize), &device)?.to_dtype(DType::F32)?;
    let y_mask_exp = y_mask.unsqueeze(1)?;

    let ssl_proj = load_conv1d(&sd, &device, "enc_p.ssl_proj")?;
    let mut y = ssl_proj.forward(&quantized.broadcast_mul(&y_mask_exp)?)?;
    let gamma0 = sd
        .get("enc_p.encoder_ssl.norm_layers_1.0.gamma")?
        .to_device(&device)?
        .to_dtype(DType::F32)?;
    let beta0 = sd
        .get("enc_p.encoder_ssl.norm_layers_1.0.beta")?
        .to_device(&device)?
        .to_dtype(DType::F32)?;
    y = layer_norm(&y, &gamma0, &beta0)?.broadcast_mul(&y_mask_exp)?;

    for i in 0..3 {
        y = run_encoder_layer(&y, &y_mask_exp, &sd, &device, "enc_p.encoder_ssl", i)?;
    }

    let text_tokens: Vec<i64> = std::fs::read_to_string("encp_debug_text_ids.txt")?
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let text_max_len = text_tokens.len();
    let text_mask_data: Vec<f32> = (0..text_max_len as i64)
        .map(|j| if j < text_max_len as i64 { 1.0 } else { 0.0 })
        .collect();
    let text_mask =
        Tensor::from_vec(text_mask_data, (1, text_max_len), &device)?.to_dtype(DType::F32)?;
    let text_mask_exp = text_mask.unsqueeze(1)?;

    let text_emb_w = sd
        .get("enc_p.text_embedding.weight")?
        .to_device(&device)?
        .to_dtype(DType::F32)?;
    let mut text_emb = Tensor::zeros((1, text_max_len, text_emb_w.dims()[1]), DType::F32, &device)?;
    let indices: Vec<i64> = text_tokens.clone();
    let mut embeddings = Vec::new();
    for &idx in &indices {
        embeddings.push(text_emb_w.get(idx as usize)?);
    }
    text_emb = Tensor::stack(&embeddings, 0)?.reshape((1, text_max_len, text_emb_w.dims()[1]))?;
    text_emb = text_emb.broadcast_mul(&text_mask_exp)?;

    for i in 0..6 {
        text_emb = run_encoder_layer(
            &text_emb,
            &text_mask_exp,
            &sd,
            &device,
            "enc_p.encoder_text",
            i,
        )?;
    }

    // MRTE
    let c_pre = load_conv1d(&sd, &device, "enc_p.mrte.c_pre")?;
    let text_pre = load_conv1d(&sd, &device, "enc_p.mrte.text_pre")?;
    let c_post = load_conv1d(&sd, &device, "enc_p.mrte.c_post")?;
    let ca_q = load_conv1d(&sd, &device, "enc_p.mrte.cross_attention.conv_q")?;
    let ca_k = load_conv1d(&sd, &device, "enc_p.mrte.cross_attention.conv_k")?;
    let ca_v = load_conv1d(&sd, &device, "enc_p.mrte.cross_attention.conv_v")?;
    let ca_o = load_conv1d(&sd, &device, "enc_p.mrte.cross_attention.conv_o")?;

    let ssl_enc = y.broadcast_mul(&y_mask_exp)?;
    let ssl_proj_out = c_pre.forward(&ssl_enc)?;
    let text_enc = text_emb.broadcast_mul(&text_mask_exp)?;
    let text_proj = text_pre.forward(&text_enc)?;

    let q_ca = ca_q.forward(&ssl_proj_out.broadcast_mul(&y_mask_exp)?)?;
    let k_ca = ca_k.forward(&text_proj.broadcast_mul(&text_mask_exp)?)?;
    let v_ca = ca_v.forward(&text_proj.broadcast_mul(&text_mask_exp)?)?;

    let seq_ssl = q_ca.dims()[2];
    let seq_text = k_ca.dims()[2];
    let n_heads_ca: usize = 4;
    let head_dim_ca = 512 / n_heads_ca;

    // Direct reshape (same as Rust MultiHeadAttention::forward)
    let q_h = q_ca.reshape((1, n_heads_ca, seq_ssl, head_dim_ca))?;
    let k_h = k_ca.reshape((1, n_heads_ca, seq_text, head_dim_ca))?;
    let v_h = v_ca.reshape((1, n_heads_ca, seq_text, head_dim_ca))?;

    let raw_scores = q_h.matmul(&k_h.transpose(2, 3)?)?;
    let scale = (head_dim_ca as f64).sqrt().recip();
    let scaled_scores =
        raw_scores.broadcast_mul(&Tensor::full(scale as f32, raw_scores.dims(), &device)?)?;

    println!("Rust MRTE cross-attn:");
    println!("  q_h: {:?}, mean={:.6}", q_h.dims(), mean_val(&q_h)?);
    println!("  k_h: {:?}, mean={:.6}", k_h.dims(), mean_val(&k_h)?);
    println!("  v_h: {:?}, mean={:.6}", v_h.dims(), mean_val(&v_h)?);
    println!(
        "  raw_scores: {:?}, mean={:.6}, std={:.6}",
        raw_scores.dims(),
        mean_val(&raw_scores)?,
        {
            let f: Vec<f32> = raw_scores.flatten_all()?.to_vec1()?;
            let m = f.iter().sum::<f32>() / f.len() as f32;
            let v = f.iter().map(|x| (x - m).powi(2)).sum::<f32>() / f.len() as f32;
            v.sqrt()
        }
    );
    println!(
        "  scaled_scores: {:?}, mean={:.6}",
        scaled_scores.dims(),
        mean_val(&scaled_scores)?
    );

    // Mask
    let text_mask_3d = text_mask.reshape((1, 1, seq_text))?;
    let ssl_mask_4d = y_mask.reshape((1, 1, seq_ssl, 1))?;
    let attn_mask = text_mask_3d.broadcast_mul(&ssl_mask_4d)?;
    let mask_bc = attn_mask.broadcast_as(scaled_scores.dims())?;
    let neg_inf = Tensor::full(-1e9f32, mask_bc.dims(), &device)?;
    let ones = Tensor::ones(mask_bc.dims(), DType::F32, &device)?;
    let inv_mask = ones.sub(&mask_bc)?.broadcast_mul(&neg_inf)?;
    let masked_scores = scaled_scores.broadcast_mul(&mask_bc)?.add(&inv_mask)?;

    let attn_probs = candle_nn::ops::softmax(&masked_scores, candle_core::D::Minus1)?;
    let attn_out_ca = attn_probs.matmul(&v_h)?.reshape((1, 512, seq_ssl))?;
    let attn_out_ca = ca_o.forward(&attn_out_ca)?;

    println!("  attn_out_ca: mean={:.6}", mean_val(&attn_out_ca)?);

    let y_mrte = attn_out_ca.add(&ssl_proj_out)?;
    let ge = load_tensor_file("encp_debug_ge.txt")?;
    let ge_t = Tensor::from_vec(ge.1, ge.0.clone(), &device)?.to_dtype(DType::F32)?;
    let ge_bc = if ge_t.dims()[2] == 1 && seq_ssl != 1 {
        ge_t.broadcast_as((1, ge_t.dims()[1], seq_ssl))?
    } else {
        ge_t.clone()
    };
    let y_mrte = y_mrte.add(&ge_bc)?;
    println!("  y_mrte (before c_post): mean={:.6}", mean_val(&y_mrte)?);

    let y_out = c_post.forward(&y_mrte.broadcast_mul(&y_mask_exp)?)?;
    let y_out = y_out.broadcast_mul(&y_mask_exp)?;
    println!("  y_out (after c_post): mean={:.6}", mean_val(&y_out)?);

    // Save for comparison
    let cpu = Device::Cpu;
    let mut map = HashMap::new();
    for (name, t) in [
        ("q_h", &q_h),
        ("k_h", &k_h),
        ("v_h", &v_h),
        ("raw_scores", &raw_scores),
        ("scaled_scores", &scaled_scores),
        ("attn_probs", &attn_probs),
        ("attn_out_ca", &attn_out_ca),
        ("ssl_proj_out", &ssl_proj_out),
        ("text_proj", &text_proj),
        ("y_mrte", &y_mrte),
        ("y_out", &y_out),
    ] {
        map.insert(name.to_string(), t.to_device(&cpu)?.to_dtype(DType::F32)?);
    }
    candle_core::safetensors::save(&map, "/tmp/rust_mrte_attn.safetensors")?;
    println!("\nSaved to /tmp/rust_mrte_attn.safetensors");

    Ok(())
}
