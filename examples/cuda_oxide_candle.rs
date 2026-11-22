use anyhow::{Context, Result, bail, ensure};
use candle_core::{
    Device, Storage, Tensor,
    cuda_backend::{
        CudaStorageSlice,
        cudarc::{
            driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg},
            nvrtc::Ptx,
        },
    },
};
use gpt_sovits_rs::models::sovits_decoder::leaky_relu;
use std::{path::PathBuf, time::Instant};

const WARMUP: usize = 20;
const ITERATIONS: usize = 500;

fn cuda_ptr(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
    ensure!(layout.is_contiguous(), "expected a contiguous CUDA tensor");
    let Storage::Cuda(storage) = &*storage else {
        bail!("tensor is not on a CUDA device");
    };
    let CudaStorageSlice::F32(slice) = &storage.slice else {
        bail!("expected an F32 CUDA tensor");
    };
    let stream = storage.device.cuda_stream();
    let (pointer, _guard) = slice.device_ptr(&stream);
    Ok(pointer + (layout.start_offset() * size_of::<f32>()) as u64)
}

fn max_error(input: &[f32], output: &[f32], slope: f32) -> Result<f32> {
    ensure!(input.len() == output.len(), "output length mismatch");
    input
        .iter()
        .zip(output.iter())
        .map(|(&input, &actual)| {
            let expected = if input >= 0.0 { input } else { input * slope };
            ensure!(
                expected.is_finite() && actual.is_finite(),
                "non-finite value"
            );
            Ok((expected - actual).abs())
        })
        .try_fold(0.0f32, |max, error| Ok(max.max(error?)))
}

fn main() -> Result<()> {
    let ptx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("experiments/cuda-oxide/gpt_sovits_cuda_oxide.ptx");
    if !ptx_path.exists() {
        bail!(
            "missing {}; generate it with `cd experiments/cuda-oxide && cargo oxide build --arch sm_89`",
            ptx_path.display()
        );
    }

    let device = Device::new_cuda_with_stream(0).context("create Candle CUDA device")?;
    let Device::Cuda(cuda_device) = &device else {
        unreachable!();
    };
    // Match the pipeline's dedicated-stream policy in Config::candle_device.
    unsafe { cuda_device.disable_event_tracking() };
    let stream = cuda_device.cuda_stream();
    let module = stream
        .context()
        .load_module(Ptx::from_file(&ptx_path))
        .context("load cuda-oxide PTX into Candle's CUDA context")?;
    let function = module
        .load_function("fused_leaky_relu")
        .context("load fused_leaky_relu")?;

    for round in 1..=3 {
        for elements in [257, 1 << 14, 1 << 16, 1 << 18, 1 << 20] {
            for slope in [0.1, 0.01] {
                println!("round={round} elements={elements} slope={slope}");
                benchmark_case(&device, &function, elements, slope)?;
            }
        }
    }
    Ok(())
}

fn benchmark_case(
    device: &Device,
    function: &CudaFunction,
    elements: usize,
    slope: f32,
) -> Result<()> {
    let Device::Cuda(cuda_device) = device else {
        bail!("expected a CUDA device");
    };
    let stream = cuda_device.cuda_stream();
    let input_host: Vec<f32> = (0..elements)
        .map(|index| index as f32 / elements as f32 * 2.0 - 1.0)
        .collect();
    // Exercise a nonzero storage offset as well as partial final thread blocks.
    let padded: Vec<f32> = [vec![0.0; 7], input_host.clone()].concat();
    let input = Tensor::from_vec(padded, elements + 7, device)?.narrow(0, 7, elements)?;
    let input_ptr = cuda_ptr(&input)?;
    let length = elements as u64;
    let config = LaunchConfig::for_num_elems(elements as u32);

    let launch = |output: &Tensor| -> Result<()> {
        let output_ptr = cuda_ptr(output)?;
        let mut args = stream.launch_builder(function);
        args.arg(&input_ptr)
            .arg(&length)
            .arg(&slope)
            .arg(&output_ptr)
            .arg(&length);
        // Both tensors outlive the launch and all work uses this one stream.
        unsafe { args.launch(config) }.context("launch fused_leaky_relu")?;
        Ok(())
    };

    let preallocated_output = Tensor::zeros(elements, candle_core::DType::F32, device)?;
    for _ in 0..WARMUP {
        launch(&preallocated_output)?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        launch(&preallocated_output)?;
    }
    stream.synchronize()?;
    let preallocated_elapsed = start.elapsed();

    let mut custom_output = Tensor::zeros_like(&input)?;
    launch(&custom_output)?;
    for _ in 1..WARMUP {
        custom_output = Tensor::zeros_like(&input)?;
        launch(&custom_output)?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        custom_output = Tensor::zeros_like(&input)?;
        launch(&custom_output)?;
    }
    stream.synchronize()?;
    let allocated_elapsed = start.elapsed();

    let mut candle_output = leaky_relu(&input, slope)?;
    for _ in 1..WARMUP {
        candle_output = leaky_relu(&input, slope)?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        candle_output = leaky_relu(&input, slope)?;
    }
    stream.synchronize()?;
    let candle_elapsed = start.elapsed();

    let mut native_output = candle_nn::ops::leaky_relu(&input, slope as f64)?;
    for _ in 1..WARMUP {
        native_output = candle_nn::ops::leaky_relu(&input, slope as f64)?;
    }
    stream.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        native_output = candle_nn::ops::leaky_relu(&input, slope as f64)?;
    }
    stream.synchronize()?;
    let native_elapsed = start.elapsed();

    let reused_error = max_error(&input_host, &preallocated_output.to_vec1::<f32>()?, slope)?;
    let custom_error = max_error(&input_host, &custom_output.to_vec1::<f32>()?, slope)?;
    let candle_error = max_error(&input_host, &candle_output.to_vec1::<f32>()?, slope)?;
    let native_error = max_error(&input_host, &native_output.to_vec1::<f32>()?, slope)?;
    ensure!(
        reused_error <= f32::EPSILON,
        "reused output error: {reused_error}"
    );
    if custom_error > f32::EPSILON || candle_error > f32::EPSILON {
        bail!("maximum errors: cuda-oxide={custom_error}, Candle={candle_error}");
    }
    ensure!(
        native_error <= f32::EPSILON,
        "Candle native error: {native_error}"
    );

    let average_us =
        |elapsed: std::time::Duration| elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;

    println!(
        "cuda-oxide preallocated: elements={elements} average={:.2}us",
        average_us(preallocated_elapsed)
    );
    println!(
        "cuda-oxide allocated:    elements={elements} average={:.2}us max_error={custom_error:.2e}",
        average_us(allocated_elapsed)
    );
    println!(
        "Candle composed:         elements={elements} average={:.2}us max_error={candle_error:.2e}",
        average_us(candle_elapsed)
    );
    println!(
        "Candle native:           elements={elements} average={:.2}us max_error={native_error:.2e}",
        average_us(native_elapsed)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::max_error;

    #[test]
    fn checks_slope_and_rejects_incomplete_or_nonfinite_output() {
        assert_eq!(max_error(&[-1.0, 2.0], &[-0.1, 2.0], 0.1).unwrap(), 0.0);
        assert_eq!(max_error(&[-1.0], &[-0.01], 0.01).unwrap(), 0.0);
        assert!(max_error(&[1.0], &[], 0.1).is_err());
        assert!(max_error(&[1.0], &[f32::NAN], 0.1).is_err());
        assert!(max_error(&[1.0], &[f32::INFINITY], 0.1).is_err());
    }
}
