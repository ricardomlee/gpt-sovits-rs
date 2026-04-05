//! Benchmark tests for GPT-SoVITS inference

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpt_sovits_rs::{Config, InferenceOptions, Language, Pipeline, AudioBuffer};

fn criterion_benchmark(c: &mut Criterion) {
    // Audio buffer benchmark
    c.bench_function("audio_buffer_create", |b| {
        b.iter(|| {
            let samples = vec![0.0f32; 24000];
            let _buffer = AudioBuffer::new(black_box(samples), black_box(24000), black_box(1));
        })
    });

    c.bench_function("audio_buffer_normalize", |b| {
        let mut buffer = AudioBuffer::new(vec![0.5f32; 24000], 24000, 1);
        b.iter(|| {
            let mut buf = buffer.clone();
            buf.normalize();
        })
    });

    c.bench_function("audio_buffer_fade", |b| {
        let mut buffer = AudioBuffer::new(vec![1.0f32; 24000], 24000, 1);
        b.iter(|| {
            let mut buf = buffer.clone();
            buf.fade_in(100);
            buf.fade_out(100);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
