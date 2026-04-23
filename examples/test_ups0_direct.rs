/// Direct test: same input + same weights → compare with Python ups0 output
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_txt(path: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();
    let shape_line = lines.next().unwrap()?;
    let shape: Vec<usize> = shape_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let mut data = Vec::new();
    for line in lines {
        let line = line?;
        if line.is_empty() { continue; }
        data.push(line.trim().parse::<f32>()?);
    }
    Ok(Tensor::from_vec(data, &*shape, device)?)
}

fn main() {
    let device = Device::Cpu;

    // Load input from debug_cond.txt
    let input = load_txt("debug_cond.txt", &device).unwrap();
    println!("Input shape: {:?}", input.dims()); // [1, 512, 200]

    // Load Python expected output
    let py_out = load_txt("debug_ups0.txt", &device).unwrap();
    println!("Python output shape: {:?}", py_out.dims()); // [1, 256, 2000]
    let py_flat: Vec<f32> = py_out.flatten_all().unwrap().to_vec1().unwrap();
    println!("Python first 5: {:?}", &py_flat[..5]);

    // Load weights
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);
    let wv = state_dict.get("dec.ups.0.weight_v").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let wg = state_dict.get("dec.ups.0.weight_g").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let bias = state_dict.get("dec.ups.0.bias").unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();

    // Apply weight norm
    let v_norm = wv.sqr().unwrap().sum(D::Minus1).unwrap().sum(D::Minus1).unwrap().sqrt().unwrap()
        .reshape((512, 1, 1)).unwrap();
    let weight = wv.broadcast_div(&v_norm).unwrap().broadcast_mul(&wg).unwrap();

    // Run conv_transpose1d
    let out = input.conv_transpose1d(&weight, 3, 0, 10, 1, 1).unwrap();
    let bias_r = bias.reshape((1, 256, 1)).unwrap();
    let out = out.broadcast_add(&bias_r).unwrap();

    let out_flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    println!("Candle first 5: {:?}", &out_flat[..5]);

    // Compare element by element - check if it's a layout issue
    println!("\nFirst 10 comparison:");
    for i in 0..10 {
        let diff = (out_flat[i] - py_flat[i]).abs();
        println!("  [{}]: candle={:.8} python={:.8} diff={:.8}", i, out_flat[i], py_flat[i], diff);
    }

    // Check max diff
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..out_flat.len() {
        let d = (out_flat[i] - py_flat[i]).abs();
        max_diff = max_diff.max(d);
        sum_diff += d;
    }
    println!("\nMax diff: {:.8}, Mean diff: {:.8}", max_diff, sum_diff / out_flat.len() as f32);

    // Try a small sub-problem: take a tiny slice of input and weight,
    // compute manually, compare with Python doing the same tiny slice
    println!("\n=== Small sub-problem test ===");
    // Take input[:, :4, :5] (channels 0-3, time 0-4) and weight[:4, :4, :4]
    let small_input = input.narrow(1, 0, 4).unwrap().narrow(2, 0, 5).unwrap();
    let small_weight = weight.narrow(0, 0, 4).unwrap().narrow(1, 0, 4).unwrap().narrow(2, 0, 4).unwrap();
    println!("Small input shape: {:?}", small_input.dims()); // [1, 4, 5]
    println!("Small weight shape: {:?}", small_weight.dims()); // [4, 4, 4]
    
    let small_out = small_input.conv_transpose1d(&small_weight, 3, 0, 10, 1, 1).unwrap();
    println!("Small output shape: {:?}", small_out.dims());
    let small_flat: Vec<f32> = small_out.flatten_all().unwrap().to_vec1().unwrap();
    println!("Small output first 10: {:?}", &small_flat[..10]);

    // Manual computation:
    // output_time = (5-1)*10 - 2*3 + 4 = 40 - 6 + 4 = 38
    // For t_out=0: k must satisfy t_out = t_in*stride - pad + k => 0 = t_in*10 - 3 + k => k = 3 - t_in*10
    //   t_in=0 => k=3; t_in>=1 => k<0 (out of range)
    //   y[c_out, 0] = sum_c_in x[c_in, 0] * w[c_in, c_out, 3]
    let si_flat: Vec<f32> = small_input.flatten_all().unwrap().to_vec1().unwrap();
    let sw_flat: Vec<f32> = small_weight.flatten_all().unwrap().to_vec1().unwrap();
    // small_input is [1, 4, 5], stored as [c0t0, c0t1, c0t2, c0t3, c0t4, c1t0, ..., c3t4]
    // small_weight is [4, 4, 4], stored as [c_in0,out0,k0, c_in0,out0,k1, ..., c_in3,out3,k3]
    // For c_out=0, t_out=0: sum over c_in of x[c_in,0] * w[c_in, 0, 3]
    // x[c_in, 0] = si_flat[c_in * 5 + 0]
    // w[c_in, 0, 3] = sw_flat[c_in * 4 * 4 + 0 * 4 + 3]
    let mut manual_0_0 = 0.0f32;
    for c_in in 0..4 {
        let x_val = si_flat[c_in * 5];
        let w_val = sw_flat[c_in * 16 + 3];
        manual_0_0 += x_val * w_val;
    }
    println!("Manual y[c_out=0, t_out=0] = {:.8}", manual_0_0);
    println!("Candle y[c_out=0, t_out=0] = {:.8} (small_flat[0])", small_flat[0]);
}
