use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::{cuda_launch, load_kernel_module};
use std::time::Instant;

#[kernel]
pub fn fused_leaky_relu(input: &[f32], slope: f32, mut output: DisjointSlice<f32>) {
    let index = thread::index_1d();
    let offset = index.get();
    if let Some(output_value) = output.get_mut(index) {
        let value = input[offset];
        *output_value = if value >= 0.0 { value } else { value * slope };
    }
}

fn main() {
    const ELEMENTS: usize = 1 << 20;
    const WARMUP: usize = 20;
    const ITERATIONS: usize = 500;
    const SLOPE: f32 = 0.1;

    let context = CudaContext::new(0).expect("create CUDA context");
    let stream = context.default_stream();
    let input_host: Vec<f32> = (0..ELEMENTS)
        .map(|index| index as f32 / ELEMENTS as f32 * 2.0 - 1.0)
        .collect();
    let input = DeviceBuffer::from_host(&stream, &input_host).expect("copy input");
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, ELEMENTS).expect("allocate output");
    let module =
        load_kernel_module(&context, "gpt_sovits_cuda_oxide").expect("load cuda-oxide PTX");
    let launch_config = LaunchConfig::for_num_elems(ELEMENTS as u32);

    for _ in 0..WARMUP {
        unsafe {
            cuda_launch! {
                kernel: fused_leaky_relu,
                stream: stream,
                module: module,
                config: launch_config,
                args: [slice(input), SLOPE, slice_mut(output)]
            }
            .expect("warmup launch");
        }
    }
    stream.synchronize().expect("warmup synchronize");

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        unsafe {
            cuda_launch! {
                kernel: fused_leaky_relu,
                stream: stream,
                module: module,
                config: launch_config,
                args: [slice(input), SLOPE, slice_mut(output)]
            }
            .expect("benchmark launch");
        }
    }
    stream.synchronize().expect("benchmark synchronize");
    let elapsed = start.elapsed();

    let output_host = output.to_host_vec(&stream).expect("copy output");
    let max_error = input_host
        .iter()
        .zip(output_host.iter())
        .map(|(&input, &actual)| {
            let expected = if input >= 0.0 { input } else { input * SLOPE };
            (expected - actual).abs()
        })
        .fold(0.0f32, f32::max);
    assert!(max_error <= f32::EPSILON, "maximum error: {max_error}");

    let average_us = elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
    println!(
        "cuda-oxide fused_leaky_relu: elements={ELEMENTS} average={average_us:.2}us max_error={max_error:.2e}"
    );
}
