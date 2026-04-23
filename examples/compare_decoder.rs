/// Compare Rust and Python decoder with identical inputs
///
/// This saves z (decoder input) and the audio output to npz format,
/// then the Python script loads these and compares.

use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};
use gpt_sovits_rs::models::sovits_decoder::Decoder;
use std::collections::HashMap;

fn main() {
    println!("=== Decoder Comparison Test ===\n");

    let device = Device::Cpu;

    // Load SoVITS model
    println!("Loading SoVITS model...");
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Load decoder
    let decoder = Decoder::load(&state_dict, &device).unwrap();

    // Create fixed input: z [1, 192, 200]
    // Use a fixed seed approach: deterministic values
    let mut z_data = Vec::with_capacity(1 * 192 * 200);
    for i in 0..(192 * 200) {
        // Simple deterministic pattern
        z_data.push(((i % 1000) as f32 - 500.0) / 5000.0);
    }
    let z = Tensor::from_vec(z_data, (1, 192, 200), &device).unwrap();

    // Zero ge [1, 512, 1]
    let ge = Tensor::zeros((1, 512, 1), DType::F32, &device).unwrap();

    println!("Input z: {:?}", z.dims());
    println!("Input ge: {:?}", ge.dims());

    // Run decoder
    println!("\nRunning decoder...");
    let audio = decoder.forward(&z, Some(&ge)).unwrap();
    println!("Output: {} samples", audio.len());

    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();
    println!("RMS: {:.6}", rms);
    println!("Range: [{:.6}, {:.6}]", audio.iter().cloned().fold(f32::INFINITY, f32::min), audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Save z and audio to a simple text format for Python comparison
    // Save z as CSV
    let z_vec: Vec<f32> = z.flatten_all().unwrap().to_vec1().unwrap();

    // Write to files
    std::fs::write("decoder_z.txt",
        z_vec.iter().map(|v| format!("{:.8}", v)).collect::<Vec<_>>().join("\n")
    ).unwrap();

    std::fs::write("decoder_audio_rust.txt",
        audio.iter().map(|v| format!("{:.8}", v)).collect::<Vec<_>>().join("\n")
    ).unwrap();

    println!("\nSaved decoder_z.txt and decoder_audio_rust.txt");
    println!("Run Python decoder with same z and compare with decoder_audio_rust.txt");
}
