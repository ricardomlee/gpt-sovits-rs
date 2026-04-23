/// Test: Compare ResBlock0 forward (sum of 3 branches) between Rust and Python
use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};
use gpt_sovits_rs::models::sovits_decoder::ResBlock1;

fn main() {
    let device = Device::Cpu;

    // Load debug_ups0 as input
    let content = std::fs::read_to_string("debug_ups0.txt").unwrap();
    let mut lines = content.lines();
    let shape_line = lines.next().unwrap();
    let shape: Vec<usize> = shape_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let data: Vec<f32> = lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    let x = Tensor::from_vec(data, &*shape, &device).unwrap();
    println!("Input shape: {:?}", x.dims());

    // Load resblock0 using the same method as the decoder
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);
    
    let block = ResBlock1::load(&state_dict, "dec.resblocks.0", &device).unwrap();
    let out = block.forward(&x).unwrap();
    
    let out_flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    println!("Rust resblock0 forward first 5: {:?}", &out_flat[..5]);

    // Compare with Python resblock0 sum
    let py_data: Vec<f32> = std::fs::read_to_string("py_resblock0_sum.txt").unwrap()
        .lines().filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    
    let diff: Vec<f32> = out_flat.iter().zip(py_data.iter()).map(|(a,b)| (a-b).abs()).collect();
    println!("\nRust vs Python resblock0:");
    println!("  Mean diff: {:.2e}, Max diff: {:.2e}",
        diff.iter().sum::<f32>() / diff.len() as f32,
        diff.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("  Rust first 5: {:?}", &out_flat[..5]);
    println!("  Python first 5: {:?}", &py_data[..5]);
}
