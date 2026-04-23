/// Debug: Run the full resblock group (0,1,2) step by step and save intermediates
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};
use gpt_sovits_rs::models::sovits_decoder::ResBlock1;

fn leaky_relu(x: &Tensor, slope: f32) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let positive = x.maximum(&zeros)?;
    let negative = x.minimum(&zeros)?;
    let slope_t = Tensor::full(slope, x.dims(), x.device())?;
    Ok(positive.add(&negative.broadcast_mul(&slope_t)?)?)
}

fn save_tensor(name: &str, t: &Tensor) {
    let flat: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
    let dims = t.dims();
    let header = format!("{}\n", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    let data = flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n");
    std::fs::write(format!("{}.txt", name), format!("{}{}", header, data)).unwrap();
}

fn main() {
    let device = Device::Cpu;

    // Load weights
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Create same z input
    let mut z_data = Vec::with_capacity(1 * 192 * 200);
    for i in 0..(192 * 200) {
        z_data.push(((i % 1000) as f32 - 500.0) / 5000.0);
    }
    let z = Tensor::from_vec(z_data, (1, 192, 200), &device).unwrap();
    let ge = Tensor::zeros((1, 512, 1), DType::F32, &device).unwrap();

    // Run full pipeline like the decoder
    let conv_pre = gpt_sovits_rs::utils::Conv1d::new(
        state_dict.get("dec.conv_pre.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap(),
        Some(state_dict.get("dec.conv_pre.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap()),
        1, 3, 1
    );
    let mut x = conv_pre.forward(&z).unwrap();
    save_tensor("rust_debug_conv_pre", &x);

    let cond_w = state_dict.get("dec.cond.weight").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cond_b = state_dict.get("dec.cond.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let cond = gpt_sovits_rs::utils::Conv1d::new(cond_w, Some(cond_b), 1, 0, 1);
    let g_proj = cond.forward(&ge).unwrap();
    x = x.broadcast_add(&g_proj).unwrap();
    save_tensor("rust_debug_cond", &x);

    // Load ups layers
    let dilations = [1, 3, 5];
    let upsample_rates = [10, 8, 2, 2, 2];

    for i in 0..5 {
        let prefix = format!("dec.ups.{}", i);
        let wg = state_dict.get(&format!("{}.weight_g", prefix)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let wv = state_dict.get(&format!("{}.weight_v", prefix)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let bias = state_dict.get(&format!("{}.bias", prefix)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let k = wv.dims()[2];
        let pad = (k - upsample_rates[i]) / 2;
        let conv = Conv1dWeightNorm::new(wg, wv, Some(bias), 1, pad, 1);

        x = leaky_relu(&x, 0.1).unwrap();

        let weight = conv.get_weight().unwrap();
        let out = x.conv_transpose1d(&weight, pad, 0, upsample_rates[i], 1, 1).unwrap();
        let bias_r = bias.reshape(&[1, bias.dims()[0], 1]).unwrap();
        x = out.broadcast_add(&bias_r).unwrap();
        save_tensor(&format!("rust_debug_ups{}", i), &x);

        // Resblock group
        let resblock_start = i * 3;
        let resblock_end = (resblock_start + 3).min(15);

        if resblock_start < 15 {
            let mut xs_acc: Option<Tensor> = None;
            for j in resblock_start..resblock_end {
                let block = ResBlock1::load(&state_dict, &format!("dec.resblocks.{}", j), &device).unwrap();
                let xs = block.forward(&x).unwrap();
                if j == resblock_start && i == 0 {
                    // Save first block output
                    save_tensor(&format!("rust_debug_block{}_out", j), &xs);
                }
                xs_acc = Some(match xs_acc {
                    Some(acc) => acc.add(&xs).unwrap(),
                    None => xs,
                });
            }
            if let Some(xs) = xs_acc {
                let divisor = Tensor::full((resblock_end - resblock_start) as f32, xs.dims(), &device).unwrap();
                x = xs.broadcast_div(&divisor).unwrap();
            }
            save_tensor(&format!("rust_debug_resblock{}", i), &x);
        }
    }

    println!("Debug files saved.");
}
