/// Test decoder with debug output
use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::models::sovits_decoder::Decoder;
use gpt_sovits_rs::utils::{load_safetensors, StateDict};

fn main() {
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);
    let decoder = Decoder::load(&state_dict, &device).unwrap();

    // Load decoder input from Rust pipeline
    let dec_input_data = load_tensor_file("sovits_dec_input.txt", &device).unwrap();
    println!(
        "Dec input: {:?}, mean={:.6}",
        dec_input_data.dims(),
        tensor_mean(&dec_input_data).unwrap()
    );

    // Load ge
    let ge_data = load_tensor_file("sovits_debug_ge.txt", &device).unwrap();
    println!(
        "ge: {:?}, mean={:.6}",
        ge_data.dims(),
        tensor_mean(&ge_data).unwrap()
    );

    // Run decoder with debug
    let audio = decoder
        .forward_debug(&dec_input_data, Some(&ge_data))
        .unwrap();
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!("\nAudio: {} samples, RMS={:.6}", audio.len(), rms);

    // Save audio for comparison
    std::fs::write(
        "sovits_rust_audio.txt",
        audio
            .iter()
            .map(|v| format!("{:.10}", v))
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
}

fn load_tensor_file(path: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.trim().split('\n').collect();
    let dims: Vec<usize> = lines[0].split(',').map(|d| d.parse().unwrap()).collect();
    let data: Vec<f32> = lines[1..]
        .iter()
        .map(|s| s.trim().parse().unwrap())
        .collect();
    Ok(Tensor::from_vec(data, dims, device)?.to_dtype(DType::F32)?)
}

fn tensor_mean(t: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().sum::<f32>() / flat.len() as f32)
}
