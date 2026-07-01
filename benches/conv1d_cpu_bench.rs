//! CPU Conv1d benchmark using shapes from the SoVITS decoder residual blocks.

use candle_core::{Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

#[derive(Clone, Copy)]
struct ConvCase {
    name: &'static str,
    channels: usize,
    length: usize,
    kernel: usize,
    dilation: usize,
}

impl ConvCase {
    fn padding(self) -> usize {
        self.dilation * (self.kernel - 1) / 2
    }
}

fn im2col_serial(input: &[f32], case: ConvCase) -> Vec<f32> {
    let mut output = vec![0f32; case.length * case.channels * case.kernel];
    for output_pos in 0..case.length {
        let output_base = output_pos * case.channels * case.kernel;
        for channel in 0..case.channels {
            let input_base = channel * case.length;
            let output_base = output_base + channel * case.kernel;
            for kernel_pos in 0..case.kernel {
                let padded_pos = output_pos + kernel_pos * case.dilation;
                if padded_pos >= case.padding() && padded_pos < case.length + case.padding() {
                    output[output_base + kernel_pos] =
                        input[input_base + padded_pos - case.padding()];
                }
            }
        }
    }
    output
}

fn im2col_parallel(input: &[f32], case: ConvCase) -> Vec<f32> {
    let row_size = case.channels * case.kernel;
    let mut output = vec![0f32; case.length * row_size];
    output
        .par_chunks_mut(row_size)
        .enumerate()
        .for_each(|(output_pos, row)| {
            for channel in 0..case.channels {
                let input_base = channel * case.length;
                let output_base = channel * case.kernel;
                for kernel_pos in 0..case.kernel {
                    let padded_pos = output_pos + kernel_pos * case.dilation;
                    if padded_pos >= case.padding() && padded_pos < case.length + case.padding() {
                        row[output_base + kernel_pos] =
                            input[input_base + padded_pos - case.padding()];
                    }
                }
            }
        });
    output
}

fn benchmark_conv1d(c: &mut Criterion) {
    // Keep the crate linked so its MKL hgemm compatibility symbol is available.
    black_box(gpt_sovits_rs::Config::default());
    let cases = [
        ConvCase {
            name: "stage1_c128_l7520_k11_d5",
            channels: 128,
            length: 7_520,
            kernel: 11,
            dilation: 5,
        },
        ConvCase {
            name: "stage4_c16_l60160_k11_d5",
            channels: 16,
            length: 60_160,
            kernel: 11,
            dilation: 5,
        },
    ];

    let mut group = c.benchmark_group("sovits_conv1d_cpu");
    group.sample_size(10);

    for case in cases {
        let input = vec![0.01f32; case.channels * case.length];
        let serial = im2col_serial(&input, case);
        let parallel = im2col_parallel(&input, case);
        assert_eq!(serial, parallel);

        group.bench_with_input(
            BenchmarkId::new("im2col_serial", case.name),
            &case,
            |b, &case| b.iter(|| im2col_serial(black_box(&input), case)),
        );
        group.bench_with_input(
            BenchmarkId::new("im2col_parallel", case.name),
            &case,
            |b, &case| b.iter(|| im2col_parallel(black_box(&input), case)),
        );

        let input_tensor =
            Tensor::from_vec(input.clone(), (1, case.channels, case.length), &Device::Cpu).unwrap();
        let weight = Tensor::ones(
            (case.channels, case.channels, case.kernel),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        group.bench_with_input(
            BenchmarkId::new("candle_conv1d", case.name),
            &case,
            |b, &case| {
                b.iter(|| {
                    input_tensor
                        .conv1d(&weight, case.padding(), 1, case.dilation, 1)
                        .unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_conv1d);
criterion_main!(benches);
