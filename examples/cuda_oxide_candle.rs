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
    let output = Tensor::zeros(ELEMENTS, candle_core::DType::F32, &device)?;
    let input_ptr = cuda_ptr(&input)?;
    let output_ptr = cuda_ptr(&output)?;
    let length = ELEMENTS as u64;
    let config = LaunchConfig::for_num_elems(ELEMENTS as u32);

    let launch = || -> Result<()> {
        let mut args = stream.launch_builder(&function);
        args.arg(&input_ptr)
            .arg(&length)
            .arg(&SLOPE)
            .arg(&output_ptr)
            .arg(&length);
        unsafe { args.launch(config) }.context("launch fused_leaky_relu")?;
        Ok(())
    };

    for _ in 0..WARMUP {
        launch()?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        launch()?;
    }
    stream.synchronize()?;
    let elapsed = start.elapsed();

    let output_host = output.to_vec1::<f32>()?;
    let max_error = input_host
        .iter()
        .zip(output_host.iter())
        .map(|(&input, &actual)| {
            let expected = if input >= 0.0 { input } else { input * SLOPE };
            (expected - actual).abs()
        })
        .fold(0.0f32, f32::max);
    if max_error > f32::EPSILON {
        bail!("maximum error {max_error} exceeds f32 epsilon");
    }

    let average_us = elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
    println!(
        "cuda-oxide through Candle: elements={ELEMENTS} average={average_us:.2}us max_error={max_error:.2e}"
    );
    Ok(())
}
