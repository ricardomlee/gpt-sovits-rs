/// Test: Run actual library Decoder on ups0 input to compare with manual computation
use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};
use gpt_sovits_rs::models::sovits_decoder::Decoder;

fn main() {
    println!("=== Decoder ResBlock Test ===\n");
    let device = Device::Cpu;

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Load full decoder
    let decoder = Decoder::load(&state_dict, &device).unwrap();

    // Create same z input
    let mut z_data = Vec::with_capacity(1 * 192 * 200);
    for i in 0..(192 * 200) {
        z_data.push(((i % 1000) as f32 - 500.0) / 5000.0);
    }
    let z = Tensor::from_vec(z_data, (1, 192, 200), &device).unwrap();
    let ge = Tensor::zeros((1, 512, 1), DType::F32, &device).unwrap();

    // Run full decoder with debug output
    let audio = decoder.forward_debug(&z, Some(&ge)).unwrap();
    println!("Full decoder audio: {} samples", audio.len());
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!("RMS: {:.8}", rms);

    // Save audio for comparison
    std::fs::write("decoder_audio_library.txt",
        audio.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n")
    ).unwrap();
    println!("Saved decoder_audio_library.txt");
}
