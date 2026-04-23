/// Debug: Save ALL intermediate decoder outputs
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};

fn save_tensor(name: &str, t: &Tensor) {
    let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    let dims = t.dims();
    let header = format!("{}\n", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    let data = flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n");
    std::fs::write(format!("rust_{}.txt", name), format!("{}{}\n", header, data)).unwrap();
    println!("  Saved {} (shape: {:?})", name, dims);
}

fn get_wn(state_dict: &StateDict, prefix: &str, device: &Device) -> Tensor {
    let v = state_dict.get(&format!("{}.weight_v", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap();
    let g = state_dict.get(&format!("{}.weight_g", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap();
    let norm = v.sqr().unwrap().sum(D::Minus1).unwrap().sum(D::Minus1).unwrap().sqrt().unwrap();
    let out_ch = v.dims()[0];
    let norm_r = norm.reshape((out_ch, 1, 1)).unwrap();
    v.broadcast_div(&norm_r).unwrap().broadcast_mul(&g).unwrap()
}

fn get_wn_bias(state_dict: &StateDict, prefix: &str, device: &Device) -> Tensor {
    state_dict.get(&format!("{}.bias", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap()
}

fn resblock_forward(x: &Tensor, state_dict: &StateDict, device: &Device, block_idx: usize) -> Tensor {
    let dilations = [1, 3, 5];
    let mut xs: Option<Tensor> = None;
    for d_idx in 0..3 {
        let w1 = get_wn(state_dict, &format!("dec.resblocks.{}.convs1.{}", block_idx, d_idx), device);
        let b1 = get_wn_bias(state_dict, &format!("dec.resblocks.{}.convs1.{}", block_idx, d_idx), device);
        let w2 = get_wn(state_dict, &format!("dec.resblocks.{}.convs2.{}", block_idx, d_idx), device);
        let b2 = get_wn_bias(state_dict, &format!("dec.resblocks.{}.convs2.{}", block_idx, d_idx), device);
        let k = w1.dims()[2];
        let dilation = dilations[d_idx];
        let pad = (k * dilation - dilation) / 2;

        let zeros = Tensor::zeros_like(x).unwrap();
        let xt = x.maximum(&zeros).unwrap();
        let b1_3d = b1.reshape((1, b1.dims()[0], 1)).unwrap();
        let xt = xt.conv1d(&w1, pad, 1, dilation, 1).unwrap().broadcast_add(&b1_3d).unwrap();
        let zeros2 = Tensor::zeros_like(&xt).unwrap();
        let xt = xt.maximum(&zeros2).unwrap();
        let b2_3d = b2.reshape((1, b2.dims()[0], 1)).unwrap();
        let xt = xt.conv1d(&w2, (k - 1) / 2, 1, 1, 1).unwrap().broadcast_add(&b2_3d).unwrap();

        xs = Some(match xs {
            Some(acc) => acc.add(&xt).unwrap(),
            None => xt,
        });
    }
    let xs = xs.unwrap();
    let divisor = Tensor::full(3.0f32, xs.dims(), device).unwrap();
    xs.broadcast_div(&divisor).unwrap()
}

fn main() {
    println!("=== Full Decoder Debug ===\n");
    let device = Device::Cpu;

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Same z
    let mut z_data = Vec::with_capacity(1 * 192 * 200);
    for i in 0..(192 * 200) {
        z_data.push(((i % 1000) as f32 - 500.0) / 5000.0);
    }
    let z = Tensor::from_vec(z_data, (1, 192, 200), &device).unwrap();
    let ge = Tensor::zeros((1, 512, 1), DType::F32, &device).unwrap();

    // conv_pre
    let cp_w = state_dict.get("dec.conv_pre.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cp_b = state_dict.get("dec.conv_pre.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cp_b_3d = cp_b.reshape((1, 512, 1)).unwrap();
    let mut x = z.conv1d(&cp_w, 3, 1, 1, 1).unwrap().broadcast_add(&cp_b_3d).unwrap();
    save_tensor("conv_pre_out", &x);

    // cond
    let cd_w = state_dict.get("dec.cond.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cd_b = state_dict.get("dec.cond.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cd_b_3d = cd_b.reshape((1, 512, 1)).unwrap();
    let g_proj = ge.conv1d(&cd_w, 0, 1, 1, 1).unwrap().broadcast_add(&cd_b_3d).unwrap();
    x = x.broadcast_add(&g_proj).unwrap();
    save_tensor("cond_out", &x);

    let UPSAMPLE_RATES = [10, 8, 2, 2, 2];

    for i in 0..5 {
        // LeakyReLU
        let zeros = Tensor::zeros_like(&x).unwrap();
        let positive = x.maximum(&zeros).unwrap();
        let negative = x.minimum(&zeros).unwrap();
        let slope = Tensor::full(0.1f32, x.dims(), &device).unwrap();
        x = positive.add(&negative.broadcast_mul(&slope).unwrap()).unwrap();

        // Upsample
        let up_w = get_wn(&state_dict, &format!("dec.ups.{}", i), &device);
        let up_b = get_wn_bias(&state_dict, &format!("dec.ups.{}", i), &device);
        let k = up_w.dims()[2];
        let stride = UPSAMPLE_RATES[i];
        let padding = (k - stride) / 2;
        let up_b_3d = up_b.reshape((1, up_b.dims()[0], 1)).unwrap();
        x = x.conv_transpose1d(&up_w, padding, 0, stride, 1, 1).unwrap().broadcast_add(&up_b_3d).unwrap();
        save_tensor(&format!("ups{}_out", i), &x);

        // Resblocks
        let mut xs_acc: Option<Tensor> = None;
        for j in 0..3 {
            let block_idx = i * 3 + j;
            if block_idx >= 15 { break; }
            let xs = resblock_forward(&x, &state_dict, &device, block_idx);
            xs_acc = Some(match xs_acc {
                Some(acc) => acc.add(&xs).unwrap(),
                None => xs,
            });
        }
        if let Some(xs) = xs_acc {
            let divisor = Tensor::full(3.0f32, xs.dims(), &device).unwrap();
            x = xs.broadcast_div(&divisor).unwrap();
        }
        save_tensor(&format!("resblock{}_group_out", i), &x);
    }

    // Final
    let zeros = Tensor::zeros_like(&x).unwrap();
    let positive = x.maximum(&zeros).unwrap();
    let negative = x.minimum(&zeros).unwrap();
    let slope = Tensor::full(0.1f32, x.dims(), &device).unwrap();
    x = positive.add(&negative.broadcast_mul(&slope).unwrap()).unwrap();

    let cp2_w = state_dict.get("dec.conv_post.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    x = x.conv1d(&cp2_w, 3, 1, 1, 1).unwrap();
    x = x.tanh().unwrap();

    let audio: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
    std::fs::write("decoder_audio_rust_debug.txt",
        audio.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n")
    ).unwrap();

    println!("\nFinal audio: {} samples, RMS: {:.8}", audio.len(),
        (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt());
}
