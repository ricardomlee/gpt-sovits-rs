use anyhow::{bail, Context, Result};
use candle_core::{
    cuda_backend::{
        cudarc::{
            driver::{DevicePtr, LaunchConfig, PushKernelArg},
            nvrtc::Ptx,
        },
        CudaStorageSlice,
    },
    Device, Storage, Tensor,
};
use std::{path::PathBuf, time::Instant};

const ELEMENTS: usize = 1 << 20;
const WARMUP: usize = 20;
const ITERATIONS: usize = 500;
const SLOPE: f32 = 0.1;

fn cuda_ptr(tensor: &Tensor) -> Result<u64> {
    let (storage, layout) = tensor.storage_and_layout();
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

fn candle_leaky_relu(input: &Tensor) -> Result<Tensor> {
    let zeros = Tensor::zeros_like(input)?;
    let positive = input.maximum(&zeros)?;
    let negative = input.minimum(&zeros)?;
    let slope = Tensor::full(SLOPE, input.dims(), input.device())?;
    Ok(positive.add(&negative.broadcast_mul(&slope)?)?)
}

fn max_error(input: &[f32], output: &[f32]) -> f32 {
    input
        .iter()
        .zip(output.iter())
        .map(|(&input, &actual)| {
            let expected = if input >= 0.0 { input } else { input * SLOPE };
            (expected - actual).abs()
        })
        .fold(0.0f32, f32::max)
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

    let device = Device::new_cuda(0).context("create Candle CUDA device")?;
    let Device::Cuda(cuda_device) = &device else {
        unreachable!();
    };
    let stream = cuda_device.cuda_stream();
    let module = stream
        .context()
        .load_module(Ptx::from_file(&ptx_path))
        .context("load cuda-oxide PTX into Candle's CUDA context")?;
    let function = module
        .load_function("fused_leaky_relu")
        .context("load fused_leaky_relu")?;

    let input_host: Vec<f32> = (0..ELEMENTS)
        .map(|index| index as f32 / ELEMENTS as f32 * 2.0 - 1.0)
        .collect();
    let input = Tensor::from_vec(input_host.clone(), ELEMENTS, &device)?;
    let input_ptr = cuda_ptr(&input)?;
    let length = ELEMENTS as u64;
    let config = LaunchConfig::for_num_elems(ELEMENTS as u32);

    let launch = |output: &Tensor| -> Result<()> {
        let output_ptr = cuda_ptr(output)?;
        let mut args = stream.launch_builder(&function);
        args.arg(&input_ptr)
            .arg(&length)
            .arg(&SLOPE)
            .arg(&output_ptr)
            .arg(&length);
        unsafe { args.launch(config) }.context("launch fused_leaky_relu")?;
        Ok(())
    };

    let preallocated_output = Tensor::zeros(ELEMENTS, candle_core::DType::F32, &device)?;
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

    let mut candle_output = candle_leaky_relu(&input)?;
    for _ in 1..WARMUP {
        candle_output = candle_leaky_relu(&input)?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        candle_output = candle_leaky_relu(&input)?;
    }
    stream.synchronize()?;
    let candle_elapsed = start.elapsed();

    let custom_error = max_error(&input_host, &custom_output.to_vec1::<f32>()?);
    let candle_error = max_error(&input_host, &candle_output.to_vec1::<f32>()?);
    if custom_error > f32::EPSILON || candle_error > f32::EPSILON {
        bail!("maximum errors: cuda-oxide={custom_error}, Candle={candle_error}");
    }

    let average_us =
        |elapsed: std::time::Duration| elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;

    println!(
        "cuda-oxide preallocated: elements={ELEMENTS} average={:.2}us",
        average_us(preallocated_elapsed)
    );
    println!(
        "cuda-oxide allocated:    elements={ELEMENTS} average={:.2}us max_error={custom_error:.2e}",
        average_us(allocated_elapsed)
    );
    println!(
        "Candle composed:         elements={ELEMENTS} average={:.2}us max_error={candle_error:.2e}",
        average_us(candle_elapsed)
    );
    Ok(())
}
