//! Debug: run decoder on fixed input, save intermediates for Python comparison.

use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::utils::{load_safetensors, StateDict};
use gpt_sovits_rs::models::sovits_decoder::Decoder;
use std::io::{BufRead, BufReader};

fn load_py_tensor(path: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    let f = std::fs::File::open(path)?;
    let mut lines = BufReader::new(f).lines();
    // numpy savetxt with header writes "# shape" on first line
    let shape_line = lines.next().ok_or_else(|| anyhow::anyhow!("empty"))??;
    let shape_str = shape_line.trim_start_matches("# ");
    let shape: Vec<usize> = shape_str.split(',')
        .map(|s| s.trim().parse::<usize>().unwrap())
        .collect();
    let mut data = Vec::new();
    for line in lines {
        let l = line?;
        let l = l.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        data.push(l.parse::<f32>()?);
    }
    Ok((shape, data))
}

fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);
    
    let weights_map = load_safetensors("models/sovits-model-v2.safetensors")?;
    let weights_map = weights_map.into_iter()
        .map(|(k, v)| v.to_device(&device).map(|v| (k, v)))
        .collect::<candle_core::Result<_>>()?;
    let state_dict = StateDict::new(weights_map);
    let decoder = Decoder::load(&state_dict, &device, DType::F32)?;
    println!("Decoder loaded");
    
    let (z_shape, z_data) = load_py_tensor("/tmp/py_input_z.txt")?;
    let (ge_shape, ge_data) = load_py_tensor("/tmp/py_input_ge.txt")?;
    println!("z shape: {:?}  n={}", z_shape, z_data.len());
    println!("ge shape: {:?}  n={}", ge_shape, ge_data.len());
    
    let z = Tensor::from_vec(z_data, (z_shape[0], z_shape[1], z_shape[2]), &device)?;
    let ge = Tensor::from_vec(ge_data, (ge_shape[0], ge_shape[1], ge_shape[2]), &device)?;
    
    // Saves debug_*.txt in the current working directory
    let audio = decoder.forward_debug(&z, Some(&ge))?;
    println!("Audio samples: {}", audio.len());

    test_single_conv()?;
    Ok(())
}

fn test_single_conv() -> anyhow::Result<()> {
    use gpt_sovits_rs::utils::{Conv1dWeightNorm};
    let device = candle_core::Device::cuda_if_available(0)?;
    println!("\n=== Single conv test ===");
    
    let weights_map = load_safetensors("models/sovits-model-v2.safetensors")?;
    let weights_map = weights_map.into_iter()
        .map(|(k, v)| v.to_device(&device).map(|v| (k, v)))
        .collect::<candle_core::Result<_>>()?;
    let state_dict = gpt_sovits_rs::utils::StateDict::new(weights_map);
    
    // Load resblock0.convs1.1 (dilation=3)
    let prefix = "dec.resblocks.0.convs1.1";
    let weight_g = state_dict.get(&format!("{}.weight_g", prefix))?.to_device(&device)?;
    let weight_v = state_dict.get(&format!("{}.weight_v", prefix))?.to_device(&device)?;
    let bias = state_dict.get(&format!("{}.bias", prefix)).ok().cloned();
    
    // kernel_size=3, dilation=3, padding=3
    let conv = Conv1dWeightNorm::new_with_cached(weight_g, weight_v, bias, 1, 3, 3)?;
    
    let (x_shape, x_data) = load_py_tensor("/tmp/py_resblock0_c1_input.txt")?;
    let x = Tensor::from_vec(x_data, (x_shape[0], x_shape[1], x_shape[2]), &device)?;
    
    // apply leaky_relu then conv
    let zeros = Tensor::zeros_like(&x)?;
    let pos = x.maximum(&zeros)?;
    let neg = x.minimum(&zeros)?;
    let slope = Tensor::full(0.1f32, x.dims(), &device)?.to_dtype(DType::F32)?;
    let xt = pos.add(&neg.broadcast_mul(&slope)?)?;
    let out = conv.forward(&xt)?;
    
    let out_data: Vec<f32> = out.flatten_all()?.to_vec1()?;
    let (py_shape, py_data) = load_py_tensor("/tmp/py_resblock0_c1_out.txt")?;
    
    let max_diff = out_data.iter().zip(py_data.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0_f32, f32::max);
    let rust_max = out_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let py_max = py_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("resblock0.convs1.1 max_diff={:.6} rust_max={:.4} py_max={:.4}", max_diff, rust_max, py_max);
    
    Ok(())
}
