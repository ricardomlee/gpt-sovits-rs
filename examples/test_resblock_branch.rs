/// Test: Compare single resblock0 branch between Rust and Python
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};

fn load_txt_flat(path: &str, device: &Device, shape: &[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let data: Vec<f32> = content.lines().filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    Ok(Tensor::from_vec(data, shape, device)?)
}

fn relu(x: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    x.maximum(&zeros)
}

fn main() {
    let device = Device::Cpu;

    // Load same input as Python (debug_ups0)
    let content = std::fs::read_to_string("debug_ups0.txt").unwrap();
    let mut lines = content.lines();
    let shape_line = lines.next().unwrap();
    let shape: Vec<usize> = shape_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let data: Vec<f32> = lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    let x = Tensor::from_vec(data, &*shape, &device).unwrap();
    println!("Input shape: {:?}", x.dims());

    // Load weights
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Load resblock0 convs1.0
    let prefix1 = "dec.resblocks.0.convs1.0";
    let wg1 = state_dict.get(&format!("{}.weight_g", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let wv1 = state_dict.get(&format!("{}.weight_v", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let b1 = state_dict.get(&format!("{}.bias", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();

    // Create Conv1dWeightNorm with padding=1, dilation=1
    let conv1 = Conv1dWeightNorm::new(wg1, wv1, Some(b1), 1, 1, 1);

    // Step 1: relu(x)
    let xt1 = relu(&x).unwrap();
    let xt1_flat: Vec<f32> = xt1.flatten_all().unwrap().to_vec1().unwrap();
    println!("Rust relu(x) first 5: {:?}", &xt1_flat[..5]);

    // Step 2: conv1
    let xt2 = conv1.forward(&xt1).unwrap();
    let xt2_flat: Vec<f32> = xt2.flatten_all().unwrap().to_vec1().unwrap();
    println!("Rust conv1 output first 5: {:?}", &xt2_flat[..5]);

    // Compare with Python
    let py_shape = [1, 256, 2000usize];
    let py_relu = load_txt_flat("py_resblock0_branch0_relu_input.txt", &device, &py_shape).unwrap();
    let py_conv1 = load_txt_flat("py_resblock0_branch0_conv1.txt", &device, &py_shape).unwrap();

    let py_relu_flat: Vec<f32> = py_relu.flatten_all().unwrap().to_vec1().unwrap();
    let py_conv1_flat: Vec<f32> = py_conv1.flatten_all().unwrap().to_vec1().unwrap();

    let relu_diff: Vec<f32> = xt1_flat.iter().zip(py_relu_flat.iter()).map(|(a,b)| (a-b).abs()).collect();
    let conv1_diff: Vec<f32> = xt2_flat.iter().zip(py_conv1_flat.iter()).map(|(a,b)| (a-b).abs()).collect();

    println!("\nRelu diff: mean={:.2e}, max={:.2e}", 
        relu_diff.iter().sum::<f32>() / relu_diff.len() as f32,
        relu_diff.iter().fold(0.0f32, |a, &b| a.max(b)));
    println!("Conv1 diff: mean={:.2e}, max={:.2e}",
        conv1_diff.iter().sum::<f32>() / conv1_diff.len() as f32,
        conv1_diff.iter().fold(0.0f32, |a, &b| a.max(b)));

    // If relu matches but conv1 doesn't, the issue is in conv1d
    // If relu doesn't match, the issue is in relu implementation
}
