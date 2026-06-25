//! Debug: compare Rust spectrogram with Python for mao.wav

use candle_core::Device;
use gpt_sovits_rs::utils::SpectrogramExtractor;
use std::io::{BufRead, BufReader};

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Load mao.wav
    let mut reader = hound::WavReader::open("/home/ric/gpt-sovits/mao.wav")?;
    let spec = reader.spec();
    println!(
        "WAV: sr={}, ch={}, bits={}",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    let samples: Vec<f32> = if spec.channels == 2 {
        samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        samples
    };

    // Normalize (Python does: maxx = audio.abs().max(); if maxx > 1: audio /= min(2, maxx))
    let max_abs = samples.iter().cloned().fold(0.0_f32, f32::max);
    let samples: Vec<f32> = if max_abs > 1.0 {
        let divisor = max_abs.min(2.0);
        samples.iter().map(|&s| s / divisor).collect()
    } else {
        samples
    };

    println!(
        "Samples: {} max={:.4}",
        samples.len(),
        samples.iter().cloned().fold(0.0_f32, f32::max)
    );

    // Extract spectrogram: n_fft=2048, hop=640, n_mels=128 (not used for linear spec)
    let extractor = SpectrogramExtractor::new(32000, 2048, 640, 128);
    let stft = extractor.extract_spectrogram(&samples, &device)?;
    let stft_data: Vec<f32> = stft.flatten_all()?.to_vec1()?;
    let dims = stft.dims();
    println!(
        "Rust STFT: {:?} max={:.4}",
        dims,
        stft_data.iter().cloned().fold(0.0_f32, f32::max)
    );

    // Load Python spec
    let f = std::fs::File::open("/tmp/py_mao_spec.txt")?;
    let mut lines = BufReader::new(f).lines();
    let shape_line = lines.next().unwrap()?;
    let shape_str = shape_line.trim_start_matches("# ");
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(|s| s.trim().parse::<usize>().unwrap())
        .collect();
    let py_data: Vec<f32> = lines
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
        .map(|l| l.trim().parse::<f32>().unwrap())
        .collect();

    println!("Python STFT: {:?} n={}", shape, py_data.len());

    // Compare (Rust is [n_bins, n_frames], Python is [n_bins, n_frames])
    let n = py_data.len().min(stft_data.len());
    let max_diff = (0..n)
        .map(|i| (stft_data[i] - py_data[i]).abs())
        .fold(0.0_f32, f32::max);
    let mean_diff = (0..n)
        .map(|i| (stft_data[i] - py_data[i]).abs())
        .sum::<f32>()
        / n as f32;
    println!(
        "Spec comparison: max_diff={:.6}  mean_diff={:.6}",
        max_diff, mean_diff
    );
    println!("First 5 Rust:   {:?}", &stft_data[..5]);
    println!("First 5 Python: {:?}", &py_data[..5]);

    Ok(())
}
