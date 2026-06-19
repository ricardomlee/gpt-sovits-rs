/// Test: Compare Rust conv1d with Python on a simple known input
use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::utils::{load_safetensors, StateDict};

fn main() {
    let device = Device::Cpu;

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    let w = state_dict
        .get("dec.conv_pre.weight")
        .unwrap()
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let b = state_dict
        .get("dec.conv_pre.bias")
        .unwrap()
        .to_device(&device)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Create same simple test input as Python
    let input_data: Vec<f32> = (0..192 * 10)
        .map(|i| ((i % 100) as f32 - 50.0) / 100.0)
        .collect();
    let input = Tensor::from_vec(input_data, (1, 192, 10), &device).unwrap();

    // Candle conv1d: input.conv1d(weight, padding, stride, dilation, groups)
    let output = input.conv1d(&w, 3, 1, 1, 1).unwrap();
    let output_with_bias = output
        .broadcast_add(&b.reshape((1, 512, 1)).unwrap())
        .unwrap();

    println!("Rust conv1d output: {:?}", output_with_bias.dims());
    println!("  mean = {:.10}", tensor_mean(&output_with_bias).unwrap());

    // Save full output for Python comparison
    save_tensor("rust_conv1d_test_output", &output_with_bias).unwrap();

    // Also test with the actual decoder input
    let dec_input = load_tensor_file("sovits_dec_input.txt", &device).unwrap();
    let x_lrelu = leaky_relu(&dec_input, 0.1);
    let dec_output = x_lrelu.conv1d(&w, 3, 1, 1, 1).unwrap();
    let dec_output_bias = dec_output
        .broadcast_add(&b.reshape((1, 512, 1)).unwrap())
        .unwrap();

    println!("\nRust dec conv_pre output: {:?}", dec_output_bias.dims());
    println!("  mean = {:.10}", tensor_mean(&dec_output_bias).unwrap());
    save_tensor("rust_conv_pre_actual", &dec_output_bias).unwrap();
}

fn leaky_relu(x: &Tensor, slope: f32) -> Tensor {
    let zeros = Tensor::zeros_like(x).unwrap();
    let positive = x.maximum(&zeros).unwrap();
    let negative = x.minimum(&zeros).unwrap();
    let slope_t = Tensor::full(slope, x.dims(), x.device()).unwrap();
    positive
        .add(&negative.broadcast_mul(&slope_t).unwrap())
        .unwrap()
}

fn tensor_mean(t: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().sum::<f32>() / flat.len() as f32)
}

fn save_tensor(name: &str, t: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let dims = t.dims();
    let header = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let data = flat
        .iter()
        .map(|v| format!("{:.10}", v))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(format!("{}.txt", name), format!("{}\n{}\n", header, data))?;
    Ok(())
}

fn load_tensor_file(path: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.trim().split('\n').collect();
    let dims: Vec<usize> = lines[0].split(',').map(|d| d.parse().unwrap()).collect();
    let data: Vec<f32> = lines[1..]
        .iter()
        .map(|s| s.trim().parse().unwrap())
        .collect();
    Ok(Tensor::from_vec(data, dims, device)?)
}
