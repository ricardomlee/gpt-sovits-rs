/// Test: Run decoder conv_pre on CPU vs CUDA and compare
use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::utils::{load_safetensors, StateDict};

fn main() {
    let cpu = Device::Cpu;
    let cuda = Device::new_cuda(0).unwrap();

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Load weights on both devices
    let w_cpu = state_dict
        .get("dec.conv_pre.weight")
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let b_cpu = state_dict
        .get("dec.conv_pre.bias")
        .unwrap()
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let w_cuda = state_dict
        .get("dec.conv_pre.weight")
        .unwrap()
        .to_device(&cuda)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let b_cuda = state_dict
        .get("dec.conv_pre.bias")
        .unwrap()
        .to_device(&cuda)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    // Load decoder input
    let dec_input_cpu = load_tensor_file("sovits_dec_input.txt", &cpu).unwrap();
    let dec_input_cuda = load_tensor_file("sovits_dec_input.txt", &cuda).unwrap();

    // LeakyReLU
    let x_lrelu_cpu = leaky_relu(&dec_input_cpu, 0.1);
    let x_lrelu_cuda = leaky_relu(&dec_input_cuda, 0.1);

    // Conv1d
    let out_cpu = x_lrelu_cpu
        .conv1d(&w_cpu, 3, 1, 1, 1)
        .unwrap()
        .broadcast_add(&b_cpu.reshape((1, 512, 1)).unwrap())
        .unwrap();
    let out_cuda = x_lrelu_cuda
        .conv1d(&w_cuda, 3, 1, 1, 1)
        .unwrap()
        .broadcast_add(&b_cuda.reshape((1, 512, 1)).unwrap())
        .unwrap();

    let out_cuda_cpu = out_cuda
        .to_device(&cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();

    let cpu_mean = tensor_mean(&out_cpu).unwrap();
    let cuda_mean = tensor_mean(&out_cuda_cpu).unwrap();
    println!(
        "CPU conv_pre:  mean={:.8}, std={:.8}",
        cpu_mean,
        tensor_std(&out_cpu).unwrap()
    );
    println!(
        "CUDA conv_pre: mean={:.8}, std={:.8}",
        cuda_mean,
        tensor_std(&out_cuda_cpu).unwrap()
    );

    let (mean_diff, max_diff) = tensor_diff(&out_cpu, &out_cuda_cpu).unwrap();
    println!("\nmean_abs_diff = {:.8e}", mean_diff);
    println!("max_abs_diff = {:.8e}", max_diff);

    if mean_diff > 0.01 {
        println!("\n*** CPU and CUDA differ significantly! ***");
    } else {
        println!("\n*** CPU and CUDA match ***");
    }
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

fn tensor_std(t: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let var = flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    Ok(var.sqrt())
}

fn tensor_diff(a: &Tensor, b: &Tensor) -> Result<(f32, f32), Box<dyn std::error::Error>> {
    let a_f: Vec<f32> = a.flatten_all()?.to_vec1()?;
    let b_f: Vec<f32> = b.flatten_all()?.to_vec1()?;
    let diffs: Vec<f32> = a_f
        .iter()
        .zip(b_f.iter())
        .map(|(a, b)| (a - b).abs())
        .collect();
    let mean_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let max_diff = diffs
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &x| if x > acc { x } else { acc });
    Ok((mean_diff, max_diff))
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
