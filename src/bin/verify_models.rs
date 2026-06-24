//! Numerical verification: F32 vs BF16 for HuBERT and BERT.
//! Run: cargo run --release --features cuda --bin verify_models

use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::models::{Wav2Vec2Model, BertCandleModel};
use std::path::Path;
use std::time::Instant;

fn load_npy_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read npy");
    let header_end = bytes.windows(1).position(|w| w[0] == b'\n').unwrap_or(128) + 1;
    let data = &bytes[header_end..];
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    a[..n].iter().zip(b[..n].iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>() / n
}

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let text = "今天天气真的很不错，适合出去散步";

    // ── HuBERT F32 vs BF16 ──────────────────────────────────────────────────
    println!("═══ HuBERT (Wav2Vec2) ═══");

    let audio_flat = load_npy_f32("scripts/hubert_onnx_input.npy");
    let n = audio_flat.len();
    let audio = Tensor::from_vec(audio_flat, (1, n), &device)?;

    let t0 = Instant::now();
    let hub_f32 = Wav2Vec2Model::load_from_file(Path::new("models/hubert.safetensors"), &device)?;
    let load_f32 = t0.elapsed();
    let t0 = Instant::now();
    let out_f32 = hub_f32.forward(&audio)?;
    device.synchronize()?;
    let infer_f32 = t0.elapsed();
    let flat_f32: Vec<f32> = out_f32.flatten_all()?.to_vec1()?;

    let t0 = Instant::now();
    let hub_bf16 = Wav2Vec2Model::load_from_file_bf16(Path::new("models/hubert.safetensors"), &device)?;
    let load_bf16 = t0.elapsed();
    let t0 = Instant::now();
    let out_bf16 = hub_bf16.forward(&audio)?;
    device.synchronize()?;
    let infer_bf16 = t0.elapsed();
    let flat_bf16: Vec<f32> = out_bf16.flatten_all()?.to_vec1()?;

    // vs ONNX reference (if available)
    let onnx_path = "scripts/hubert_onnx_output.npy";
    let onnx_flat = if Path::new(onnx_path).exists() { load_npy_f32(onnx_path) } else { flat_f32.clone() };

    println!("  F32 : load={:.2?}  infer={:.2?}", load_f32, infer_f32);
    println!("  BF16: load={:.2?}  infer={:.2?}", load_bf16, infer_bf16);
    println!("  F32  vs ONNX:  maxdiff={:.6}  meandiff={:.6}", max_abs_diff(&flat_f32, &onnx_flat), mean_abs_diff(&flat_f32, &onnx_flat));
    println!("  BF16 vs ONNX:  maxdiff={:.6}  meandiff={:.6}", max_abs_diff(&flat_bf16, &onnx_flat), mean_abs_diff(&flat_bf16, &onnx_flat));
    println!("  BF16 vs F32:   maxdiff={:.6}  meandiff={:.6}", max_abs_diff(&flat_bf16, &flat_f32), mean_abs_diff(&flat_bf16, &flat_f32));

    // ── BERT F32 vs BF16 ────────────────────────────────────────────────────
    println!("\n═══ BERT ═══");

    let t0 = Instant::now();
    let bert_f32 = BertCandleModel::load(
        Path::new("models/bert.safetensors"),
        Path::new("models/onnx/tokenizer.json"),
        &device,
    )?;
    let load_f32 = t0.elapsed();
    let t0 = Instant::now();
    let out_f32 = bert_f32.extract(text)?;
    device.synchronize()?;
    let infer_f32 = t0.elapsed();
    let flat_f32: Vec<f32> = out_f32.flatten_all()?.to_vec1()?;

    let t0 = Instant::now();
    let bert_bf16 = BertCandleModel::load_bf16(
        Path::new("models/bert.safetensors"),
        Path::new("models/onnx/tokenizer.json"),
        &device,
    )?;
    let load_bf16 = t0.elapsed();
    let t0 = Instant::now();
    let out_bf16 = bert_bf16.extract(text)?;
    device.synchronize()?;
    let infer_bf16 = t0.elapsed();
    let flat_bf16: Vec<f32> = out_bf16.flatten_all()?.to_vec1()?;

    let onnx_path = "scripts/bert_onnx_output_f32.npy";
    let onnx_flat = if Path::new(onnx_path).exists() { load_npy_f32(onnx_path) } else { flat_f32.clone() };

    println!("  F32 : load={:.2?}  infer={:.2?}", load_f32, infer_f32);
    println!("  BF16: load={:.2?}  infer={:.2?}", load_bf16, infer_bf16);
    println!("  F32  vs ONNX:  maxdiff={:.6}  meandiff={:.6}", max_abs_diff(&flat_f32, &onnx_flat), mean_abs_diff(&flat_f32, &onnx_flat));
    println!("  BF16 vs ONNX:  maxdiff={:.6}  meandiff={:.6}", max_abs_diff(&flat_bf16, &onnx_flat), mean_abs_diff(&flat_bf16, &onnx_flat));
    println!("  BF16 vs F32:   maxdiff={:.6}  meandiff={:.6}", max_abs_diff(&flat_bf16, &flat_f32), mean_abs_diff(&flat_bf16, &flat_f32));

    // Warm inference (second run, no load overhead)
    println!("\n  Warm inference (3 runs average):");
    for (label, model) in [("F32 ", &bert_f32), ("BF16", &bert_bf16)] {
        let mut total = std::time::Duration::ZERO;
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = model.extract(text)?;
            device.synchronize()?;
            total += t0.elapsed();
        }
        println!("    BERT {label}: avg {:.2?}/inference", total / 3);
    }

    for (label, model) in [("F32 ", &hub_f32), ("BF16", &hub_bf16)] {
        let mut total = std::time::Duration::ZERO;
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = model.forward(&audio)?;
            device.synchronize()?;
            total += t0.elapsed();
        }
        println!("    HuBERT {label}: avg {:.2?}/inference", total / 3);
    }

    Ok(())
}
