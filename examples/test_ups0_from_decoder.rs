/// Test: Load the actual decoder's ups0 layer, run it on debug_cond, compare with Python
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};

fn load_debug_txt(path: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let mut lines = content.lines();
    // First line might be shape or data
    let first_line = lines.next().unwrap();
    // Check if it's a shape line (contains commas)
    if first_line.contains(',') {
        let shape: Vec<usize> = first_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
        let data: Vec<f32> = lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
        Ok(Tensor::from_vec(data, &*shape, device)?)
    } else {
        // No header, just raw data - shape provided separately
        let mut data: Vec<f32> = vec![first_line.trim().parse::<f32>().unwrap()];
        data.extend(lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()));
        Ok(Tensor::from_vec(data, (1, 256, 2000), device)?)
    }
}

fn leaky_relu(x: &Tensor, slope: f32) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    let positive = x.maximum(&zeros)?;
    let negative = x.minimum(&zeros)?;
    let slope_t = Tensor::full(slope, x.dims(), x.device())?;
    Ok(positive.add(&negative.broadcast_mul(&slope_t)?)?)
}

fn main() {
    let device = Device::Cpu;

    // Load weights from safetensors
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    let prefix = "dec.ups.0";
    let weight_g = state_dict.get(&format!("{}.weight_g", prefix)).unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let weight_v = state_dict.get(&format!("{}.weight_v", prefix)).unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let bias = state_dict.get(&format!("{}.bias", prefix)).unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();

    // Get normalized weight via Conv1dWeightNorm::get_weight (same as decoder uses)
    let conv = Conv1dWeightNorm::new(weight_g, weight_v, Some(bias), 1, 7, 1);
    let weight = conv.get_weight().unwrap();
    println!("Weight shape: {:?}", weight.dims());

    // Load input
    let input = load_debug_txt("debug_cond.txt", &device).unwrap();
    println!("Input shape: {:?}", input.dims());

    // Apply leaky_relu
    let x = leaky_relu(&input, 0.1).unwrap();

    // Run conv_transpose1d (same params as decoder's upsample_forward)
    let kernel_size = weight.dims()[2];
    let stride = 10;
    let pad = (kernel_size - stride) / 2;
    println!("conv_transpose1d: kernel={}, stride={}, padding={}", kernel_size, stride, pad);

    let output = x.conv_transpose1d(&weight, pad, 0, stride, 1, 1).unwrap();
    println!("Output shape: {:?}", output.dims());

    // Compare with Python
    let py_out = load_debug_txt("py_ups0_output.txt", &device).unwrap();
    let out_flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    let py_flat: Vec<f32> = py_out.flatten_all().unwrap().to_vec1().unwrap();

    println!("\nFirst 5 comparison:");
    for i in 0..5 {
        let diff = (out_flat[i] - py_flat[i]).abs();
        println!("  [{}]: rust={:.10} py={:.10} diff={:.2e}", i, out_flat[i], py_flat[i], diff);
    }

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..out_flat.len() {
        let d = (out_flat[i] - py_flat[i]).abs();
        max_diff = max_diff.max(d);
        sum_diff += d;
    }
    println!("\nMax diff: {:.2e}, Mean diff: {:.2e}", max_diff, sum_diff / out_flat.len() as f32);

    // Also compare with debug_ups0.txt
    let debug_out = load_debug_txt("debug_ups0.txt", &device).unwrap();
    let debug_flat: Vec<f32> = debug_out.flatten_all().unwrap().to_vec1().unwrap();
    let mut max_diff2 = 0.0f32;
    let mut sum_diff2 = 0.0f32;
    for i in 0..out_flat.len() {
        let d = (out_flat[i] - debug_flat[i]).abs();
        max_diff2 = max_diff2.max(d);
        sum_diff2 += d;
    }
    println!("Rust (this test) vs debug_ups0.txt: Max diff: {:.2e}, Mean diff: {:.2e}", max_diff2, sum_diff2 / out_flat.len() as f32);
}
