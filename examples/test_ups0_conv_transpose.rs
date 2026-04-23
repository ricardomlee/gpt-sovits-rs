/// Test: Manual ConvTranspose1d for ups0 to debug Candle vs PyTorch divergence
///
/// Loads the cond output (input to ups0) from debug_cond.txt,
/// loads ups0 weights from safetensors, applies weight normalization,
/// runs conv_transpose1d, and compares with Python's debug_ups0.txt output.
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_txt_tensor(path: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    // First line: shape
    let shape_line = lines.next().unwrap()?;
    let shape: Vec<usize> = shape_line
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;

    // Remaining lines: data
    let mut data = Vec::new();
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        data.push(line.trim().parse::<f32>()?);
    }

    let tensor = Tensor::from_vec(data, &*shape, device)?;
    Ok(tensor)
}

fn main() {
    println!("=== ConvTranspose1d ups0 Debug ===\n");
    let device = Device::Cpu;

    // 1. Load input from debug_cond.txt
    let input = load_txt_tensor("debug_cond.txt", &device).unwrap();
    println!("Input shape: {:?}", input.dims());
    let inp_flat: Vec<f32> = input.flatten_all().unwrap().to_vec1().unwrap();
    println!("Input first 5: {:?}", &inp_flat[..5]);

    // 2. Load expected output from debug_ups0.txt
    let expected = load_txt_tensor("debug_ups0.txt", &device).unwrap();
    println!("Expected shape: {:?}", expected.dims());
    let exp_flat: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();
    println!("Expected first 5: {:?}", &exp_flat[..5]);

    // 3. Load weights from safetensors
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    let weight_v = state_dict.get("dec.ups.0.weight_v").unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let weight_g = state_dict.get("dec.ups.0.weight_g").unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let bias = state_dict.get("dec.ups.0.bias").unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();

    println!("\nWeight_v shape: {:?}", weight_v.dims());  // [in_ch=512, out_ch=256, kernel=16]
    println!("Weight_g shape: {:?}", weight_g.dims());   // [in_ch=512, 1, 1]
    println!("Bias shape: {:?}", bias.dims());            // [out_ch=256]

    // 4. Apply weight normalization
    // weight = weight_v * (weight_g / ||weight_v||)
    // weight_v: [in_ch=512, out_ch=256, kernel=16]
    // weight_g: [in_ch=512, 1, 1]
    // ||weight_v|| computed per input channel (over out_ch * kernel)
    let v_squared = weight_v.sqr().unwrap();
    let v_sum = v_squared.sum(D::Minus1).unwrap().sum(D::Minus1).unwrap(); // [in_ch=512]
    let v_norm = v_sum.sqrt().unwrap();
    let v_norm_reshaped = v_norm.reshape((512, 1, 1)).unwrap();
    let v_normalized = weight_v.broadcast_div(&v_norm_reshaped).unwrap();
    let weight = v_normalized.broadcast_mul(&weight_g).unwrap();
    println!("Weight after weight_norm: {:?}", weight.dims()); // [512, 256, 16]

    // 5. Run conv_transpose1d
    // PyTorch ConvTranspose1d weight: [in_channels, out_channels, kernel_size]
    // Candle conv_transpose1d weight: [in_channels, out_channels/groups, kernel_size]
    // Same format, no transposition needed
    let kernel_size = 16;
    let stride = 10;
    let padding = 3;
    println!("\nConvTranspose1d: kernel={}, stride={}, padding={}", kernel_size, stride, padding);

    let output = input.conv_transpose1d(&weight, padding, 0, stride, 1, 1).unwrap();
    println!("Output shape: {:?}", output.dims());

    // 6. Add bias (reshape to [1, out_ch, 1] for broadcasting)
    let bias_reshaped = bias.reshape((1, 256, 1)).unwrap();
    let output = output.broadcast_add(&bias_reshaped).unwrap();
    println!("Output shape (after bias): {:?}", output.dims());

    // 7. Compare
    let output_flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    let expected_flat: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    println!("\n=== First 10 values comparison ===");
    println!("{:<8} {:<16} {:<16} {:<16}", "Index", "Candle", "Python", "Diff");
    for i in 0..10 {
        let candle_val = output_flat[i];
        let python_val = expected_flat[i];
        let diff = (candle_val - python_val).abs();
        println!("{:<8} {:<16.10} {:<16.10} {:<16.10}", i, candle_val, python_val, diff);
    }

    // Overall statistics
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..output_flat.len() {
        let diff = (output_flat[i] - expected_flat[i]).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let mean_diff = sum_diff / output_flat.len() as f32;

    println!("\n=== Statistics ===");
    println!("Max absolute diff: {:.10}", max_diff);
    println!("Mean absolute diff: {:.10}", mean_diff);

    // Save output
    let dims = output.dims();
    let header = format!("{}\n", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
    let data = output_flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n");
    std::fs::write("test_ups0_candle.txt", format!("{}{}", header, data)).unwrap();
    println!("\nSaved test_ups0_candle.txt");
}
