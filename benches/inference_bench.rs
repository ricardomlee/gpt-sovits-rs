//! Benchmark tests for GPT-SoVITS inference

use candle_core::{DType, Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpt_sovits_rs::AudioBuffer;

fn audio_benchmark(c: &mut Criterion) {
    // Audio buffer benchmark
    c.bench_function("audio_buffer_create", |b| {
        b.iter(|| {
            let samples = vec![0.0f32; 24000];
            let _buffer = AudioBuffer::new(black_box(samples), black_box(24000), black_box(1));
        })
    });

    c.bench_function("audio_buffer_normalize", |b| {
        let buffer = AudioBuffer::new(vec![0.5f32; 24000], 24000, 1);
        b.iter(|| {
            let mut buf = buffer.clone();
            buf.normalize();
        })
    });

    c.bench_function("audio_buffer_fade", |b| {
        let buffer = AudioBuffer::new(vec![1.0f32; 24000], 24000, 1);
        b.iter(|| {
            let mut buf = buffer.clone();
            buf.fade_in(100);
            buf.fade_out(100);
        })
    });
}

fn tensor_benchmark(c: &mut Criterion) {
    // Prefer GPU for tensor operations
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    eprintln!("Using device: {:?}", device);

    // Tensor operations benchmark
    c.bench_function("tensor_matmul_512x512", |b| {
        let a = Tensor::randn(0.0f32, 0.1f32, (512, 512), &device).unwrap();
        let b_tensor = Tensor::randn(0.0f32, 0.1f32, (512, 512), &device).unwrap();
        b.iter(|| {
            let _ = a.matmul(&b_tensor).unwrap();
        })
    });

    c.bench_function("tensor_softmax", |b| {
        let x = Tensor::randn(0.0f32, 0.1f32, (1, 512), &device).unwrap();
        b.iter(|| {
            let _ = candle_nn::ops::softmax(&x, candle_core::D::Minus1).unwrap();
        })
    });

    c.bench_function("tensor_transpose", |b| {
        let x = Tensor::randn(0.0f32, 0.1f32, (1, 512, 768), &device).unwrap();
        b.iter(|| {
            let _ = x.transpose(1, 2).unwrap();
        })
    });

    c.bench_function("tensor_layer_norm", |b| {
        let x = Tensor::randn(0.0f32, 0.1f32, (1, 50, 512), &device).unwrap();
        let weight = Tensor::ones(512, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(512, DType::F32, &device).unwrap();
        b.iter(|| {
            let _ = candle_nn::ops::layer_norm(&x, &weight, &bias, 1e-5f32).unwrap();
        })
    });
}

fn feature_fusion_benchmark(c: &mut Criterion) {
    // Prefer GPU for feature operations
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    eprintln!("Using device: {:?}", device);

    // Simulate BERT feature fusion
    // BERT output: [batch, seq, 768]
    // proj_w: [768, 512] - need to transpose bert to [batch, 768, seq] for matmul
    // Or use [seq, 768] @ [768, 512] = [seq, 512] per sample
    c.bench_function("bert_feature_projection", |b| {
        let bert = Tensor::randn(0.0f32, 0.1f32, (50, 768), &device).unwrap(); // [seq, 768]
        let proj_w = Tensor::randn(0.0f32, 0.01f32, (768, 512), &device).unwrap();
        let proj_b = Tensor::zeros(512, DType::F32, &device).unwrap();
        b.iter(|| {
            let projected = bert.matmul(&proj_w).unwrap(); // [seq, 512]
            let _ = projected
                .broadcast_add(&proj_b.reshape((1, 512)).unwrap())
                .unwrap();
        })
    });

    c.bench_function("hubert_feature_projection", |b| {
        let hubert = Tensor::randn(0.0f32, 0.1f32, (10, 768), &device).unwrap(); // [frames, 768]
        let proj_w = Tensor::randn(0.0f32, 0.01f32, (768, 512), &device).unwrap();
        let proj_b = Tensor::zeros(512, DType::F32, &device).unwrap();
        b.iter(|| {
            let projected = hubert.matmul(&proj_w).unwrap(); // [frames, 512]
            let _ = projected
                .broadcast_add(&proj_b.reshape((1, 512)).unwrap())
                .unwrap();
        })
    });

    c.bench_function("feature_residual_fusion", |b| {
        let emb = Tensor::randn(0.0f32, 0.1f32, (50, 512), &device).unwrap();
        let feat = Tensor::randn(0.0f32, 0.1f32, (50, 512), &device).unwrap();
        let scale = Tensor::full(0.5f32, emb.dims(), &device).unwrap();
        b.iter(|| {
            let scaled_feat = feat.broadcast_mul(&scale).unwrap();
            let _ = emb.broadcast_add(&scaled_feat).unwrap();
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    audio_benchmark(c);
    tensor_benchmark(c);
    feature_fusion_benchmark(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
