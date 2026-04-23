/// Debug: Save intermediate decoder outputs for step-by-step comparison
use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};

fn save_tensor(name: &str, t: &Tensor) {
    let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    let dims = t.dims();
    let header = format!("{}\n", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    let data = flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n");
    std::fs::write(format!("rust_{}.txt", name), format!("{}{}\n", header, data)).unwrap();
    println!("  Saved {} (shape: {:?})", name, dims);
}

fn main() {
    println!("=== Rust Decoder Debug ===\n");
    let device = Device::Cpu;

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Create fixed input z [1, 192, 200]
    let mut z_data = Vec::with_capacity(1 * 192 * 200);
    for i in 0..(192 * 200) {
        z_data.push(((i % 1000) as f32 - 500.0) / 5000.0);
    }
    let z = Tensor::from_vec(z_data, (1, 192, 200), &device).unwrap();
    let ge = Tensor::zeros((1, 512, 1), DType::F32, &device).unwrap();

    println!("Input z: {:?}", z.dims());

    // Step 1: conv_pre
    println!("\nStep 1: conv_pre");
    let cp_w = state_dict.get("dec.conv_pre.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cp_b = state_dict.get("dec.conv_pre.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cp_b_3d = cp_b.reshape((1, 512, 1)).unwrap();
    let x1 = z.conv1d(&cp_w, 3, 1, 1, 1).unwrap().broadcast_add(&cp_b_3d).unwrap();
    save_tensor("conv_pre_out", &x1);

    // Step 2: cond
    println!("\nStep 2: cond");
    let cd_w = state_dict.get("dec.cond.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cd_b = state_dict.get("dec.cond.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cd_b_3d = cd_b.reshape((1, 512, 1)).unwrap();
    let g_proj = ge.conv1d(&cd_w, 0, 1, 1, 1).unwrap().broadcast_add(&cd_b_3d).unwrap();
    save_tensor("g_proj", &g_proj);

    let x2 = x1.broadcast_add(&g_proj).unwrap();
    save_tensor("cond_out", &x2);

    // Step 3: LeakyReLU
    println!("\nStep 3: LeakyReLU");
    let zeros = Tensor::zeros_like(&x2).unwrap();
    let positive = x2.maximum(&zeros).unwrap();
    let negative = x2.minimum(&zeros).unwrap();
    let slope = Tensor::full(0.1f32, x2.dims(), &device).unwrap();
    let x3 = positive.add(&negative.broadcast_mul(&slope).unwrap()).unwrap();
    save_tensor("lrelu0", &x3);

    // Step 4: Upsample 0
    println!("\nStep 4: Upsample 0");
    use candle_core::D;

    fn get_wn_weight(state_dict: &StateDict, prefix: &str, device: &Device) -> Tensor {
        let v = state_dict.get(&format!("{}.weight_v", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap();
        let g = state_dict.get(&format!("{}.weight_g", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap();
        let norm = v.sqr().unwrap().sum(D::Minus1).unwrap().sum(D::Minus1).unwrap().sqrt().unwrap();
        let out_ch = v.dims()[0];
        let norm_reshaped = norm.reshape((out_ch, 1, 1)).unwrap();
        let v_norm = v.broadcast_div(&norm_reshaped).unwrap();
        v_norm.broadcast_mul(&g).unwrap()
    }

    fn get_wn_bias(state_dict: &StateDict, prefix: &str, device: &Device) -> Option<Tensor> {
        state_dict.get(&format!("{}.bias", prefix)).ok()
            .cloned()
            .map(|t| t.to_device(device).unwrap().to_dtype(DType::F32).unwrap())
    }

    let up_w = get_wn_weight(&state_dict, "dec.ups.0", &device);
    let up_b = get_wn_bias(&state_dict, "dec.ups.0", &device);
    let k = up_w.dims()[2];
    let stride = 10;
    let padding = (k - stride) / 2;
    println!("  kernel={}, stride={}, padding={}", k, stride, padding);

    let mut x = x3;
    if let Some(b) = up_b {
        let b_3d = b.reshape((1, b.dims()[0], 1)).unwrap();
        x = x.conv_transpose1d(&up_w, padding, 0, stride, 1, 1).unwrap().broadcast_add(&b_3d).unwrap();
    } else {
        x = x.conv_transpose1d(&up_w, padding, 0, stride, 1, 1).unwrap();
    }
    save_tensor("ups0_out", &x);

    // Step 5: Resblocks 0-2
    println!("\nStep 5: Resblocks 0-2");
    let dilations = [1, 3, 5];

    fn get_wn_conv(state_dict: &StateDict, prefix: &str, device: &Device) -> Tensor {
        let v = state_dict.get(&format!("{}.weight_v", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap();
        let g = state_dict.get(&format!("{}.weight_g", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap();
        let norm = v.sqr().unwrap().sum(D::Minus1).unwrap().sum(D::Minus1).unwrap().sqrt().unwrap();
        let out_ch = v.dims()[0];
        let norm_reshaped = norm.reshape((out_ch, 1, 1)).unwrap();
        v.broadcast_div(&norm_reshaped).unwrap().broadcast_mul(&g).unwrap()
    }

    fn get_wn_conv_bias(state_dict: &StateDict, prefix: &str, device: &Device) -> Tensor {
        state_dict.get(&format!("{}.bias", prefix)).unwrap().to_device(device).unwrap().to_dtype(DType::F32).unwrap()
    }

    let x_input = x.clone();
    let mut xs_acc: Option<Tensor> = None;
    for j in 0..3 {
        let w1 = get_wn_conv(&state_dict, &format!("dec.resblocks.0.convs1.{}", j), &device);
        let b1 = get_wn_conv_bias(&state_dict, &format!("dec.resblocks.0.convs1.{}", j), &device);
        let w2 = get_wn_conv(&state_dict, &format!("dec.resblocks.0.convs2.{}", j), &device);
        let b2 = get_wn_conv_bias(&state_dict, &format!("dec.resblocks.0.convs2.{}", j), &device);
        let k = w1.dims()[2];
        let dilation = dilations[j];
        let pad = (k * dilation - dilation) / 2;

        let zeros = Tensor::zeros_like(&x_input).unwrap();
        let xt = x_input.maximum(&zeros).unwrap(); // relu
        let b1_3d = b1.reshape((1, b1.dims()[0], 1)).unwrap();
        let xt = xt.conv1d(&w1, pad, 1, dilation, 1).unwrap().broadcast_add(&b1_3d).unwrap();
        let zeros2 = Tensor::zeros_like(&xt).unwrap();
        let xt = xt.maximum(&zeros2).unwrap(); // relu
        let b2_3d = b2.reshape((1, b2.dims()[0], 1)).unwrap();
        let xt = xt.conv1d(&w2, (k - 1) / 2, 1, 1, 1).unwrap().broadcast_add(&b2_3d).unwrap();

        println!("  Resblock 0.{}: kernel={}, dilation={}, pad={}", j, k, dilation, pad);

        xs_acc = Some(match xs_acc {
            Some(acc) => acc.add(&xt).unwrap(),
            None => xt,
        });
    }
    let xs = xs_acc.unwrap();
    let divisor = Tensor::full(3.0f32, xs.dims(), &device).unwrap();
    let x_res = xs.broadcast_div(&divisor).unwrap();
    save_tensor("resblock0_out", &x_res);

    println!("\nDone! Saved files:");
    println!("  rust_conv_pre_out.txt");
    println!("  rust_g_proj.txt");
    println!("  rust_cond_out.txt");
    println!("  rust_lrelu0.txt");
    println!("  rust_ups0_out.txt");
    println!("  rust_resblock0_out.txt");
}
