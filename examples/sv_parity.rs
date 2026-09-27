//! Compare native SV against externally generated upstream F32 golden tensors.
use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};
use clap::Parser;
use gpt_sovits_rs::models::sv::{features::fbank, SvModel};
use gpt_sovits_rs::utils::load_safetensors;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 32000)]
    sovits_sample_rate: u32,
}

fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f64) -> Result<()> {
    ensure!(actual.dims() == expected.dims(), "{name}: shape mismatch");
    let a = actual.flatten_all()?.to_vec1::<f32>()?;
    let b = expected.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        a.iter().chain(&b).all(|v| v.is_finite()),
        "{name}: non-finite output"
    );
    let mut diff = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    let mut dot = 0.0;
    let mut max_abs = 0.0f64;
    for (&a, &b) in a.iter().zip(&b) {
        let (a, b) = (a as f64, b as f64);
        diff += (a - b).powi(2);
        norm_a += a * a;
        norm_b += b * b;
        dot += a * b;
        max_abs = max_abs.max((a - b).abs());
    }
    let relative_l2 = (diff / norm_b.max(1e-30)).sqrt();
    let cosine = dot / (norm_a * norm_b).sqrt().max(1e-30);
    println!("{name}: max_abs={max_abs:.8}, relative_l2={relative_l2:.8}, cosine={cosine:.10}");
    ensure!(
        relative_l2 < tolerance && cosine > 0.999999,
        "{name}: numerical mismatch"
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        _ => anyhow::bail!("expected cpu or cuda"),
    };
    let baseline = load_safetensors(args.baseline)?;
    let samples = baseline
        .get("samples")
        .ok_or_else(|| anyhow::anyhow!("missing samples"))?
        .to_vec1::<f32>()?;
    let features = baseline
        .get("fbank")
        .ok_or_else(|| anyhow::anyhow!("missing fbank"))?;
    let embedding = baseline
        .get("embedding")
        .ok_or_else(|| anyhow::anyhow!("missing embedding"))?;
    let native_features = fbank(&samples, &device)?;
    compare("fbank", &native_features, features, 1e-4)?;
    let model = SvModel::load(args.model, &device)?;
    compare(
        "encoder",
        &model.forward_features(features)?,
        embedding,
        1e-4,
    )?;
    compare(
        "samples -> embedding",
        &model.forward_features(&native_features)?,
        embedding,
        1e-4,
    )?;
    compare(
        "WAV -> embedding",
        &model.extract(args.reference, args.sovits_sample_rate)?,
        embedding,
        1e-4,
    )?;
    Ok(())
}
