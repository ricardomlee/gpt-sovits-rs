/// Debug: Test full resblock0 step by step
use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::utils::{load_safetensors, StateDict};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let weights_map = load_safetensors("models/sovits-model.safetensors")?;
    let state_dict = StateDict::new(weights_map);

    let ups0 = load_tensor_file("debug_ups0.txt", &device)?;

    // Manual resblock0 computation matching Python exactly
    // For resblock 0: kernel=3 for all convs
    // convs1 dilations: [1, 3, 5], padding = dilation * (3-1)/2 = dilation
    // convs2 dilations: [1, 1, 1], padding = (3-1)/2 = 1
    let mut x_rb = ups0.clone();

    for ci in 0..3 {
        let dilation = [1, 3, 5][ci];
        let pad1 = dilation; // dilation * (3-1)/2
        let pad2 = 1;

        // Load convs1.{ci}
        let (w_c1, b_c1) = load_decomposed(
            &state_dict,
            &device,
            &format!("dec.resblocks.0.convs1.{}", ci),
        )?;
        let (w_c2, b_c2) = load_decomposed(
            &state_dict,
            &device,
            &format!("dec.resblocks.0.convs2.{}", ci),
        )?;

        // lrelu + convs1
        let xt = leaky_relu(&x_rb, 0.1)?;
        let xt = xt.conv1d(&w_c1, pad1, 1, dilation, 1)?;
        let xt = if let Some(b) = &b_c1 {
            xt.broadcast_add(&b.reshape((1, b.dims()[0], 1))?)?
        } else {
            xt
        };

        // lrelu + convs2
        let xt = leaky_relu(&xt, 0.1)?;
        let xt = xt.conv1d(&w_c2, pad2, 1, 1, 1)?;
        let xt = if let Some(b) = &b_c2 {
            xt.broadcast_add(&b.reshape((1, b.dims()[0], 1))?)?
        } else {
            xt
        };

        // Residual add
        x_rb = x_rb.add(&xt)?;

        println!(
            "After resadd_{}: mean={:.6}, std={:.6}",
            ci,
            tensor_mean(&x_rb)?,
            tensor_std(&x_rb)?
        );

        save_tensor(&format!("rust_rb0_after_resadd_{}", ci), &x_rb)?;
    }

    println!(
        "\nFinal resblock0: mean={:.6}, std={:.6}",
        tensor_mean(&x_rb)?,
        tensor_std(&x_rb)?
    );

    Ok(())
}

fn load_decomposed(
    state_dict: &StateDict,
    device: &Device,
    prefix: &str,
) -> Result<(Tensor, Option<Tensor>), Box<dyn std::error::Error>> {
    let wg = state_dict
        .get(&format!("{}.weight_g", prefix))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let wv = state_dict
        .get(&format!("{}.weight_v", prefix))?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    let bias = state_dict
        .get(&format!("{}.bias", prefix))
        .ok()
        .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
        .transpose()?;

    let v_squared = wv.sqr()?;
    let v_norm = v_squared
        .sum(candle_core::D::Minus1)?
        .sum(candle_core::D::Minus1)?
        .sqrt()?;
    let out_channels = wv.dims()[0];
    let v_norm_reshaped = v_norm.reshape((out_channels, 1, 1))?;
    let v_normalized = wv.broadcast_div(&v_norm_reshaped)?;
    let weight = v_normalized.broadcast_mul(&wg)?;

    Ok((weight, bias))
}

fn tensor_mean(t: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(flat.iter().sum::<f32>() / flat.len() as f32)
}

fn tensor_std(t: &Tensor) -> Result<f32, Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let var = flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    Ok(var.sqrt())
}

fn leaky_relu(x: &Tensor, slope: f32) -> Result<Tensor, Box<dyn std::error::Error>> {
    let zeros = Tensor::zeros_like(x)?;
    let positive = x.maximum(&zeros)?;
    let negative = x.minimum(&zeros)?;
    let slope_t = Tensor::full(slope, x.dims(), x.device())?;
    let scaled_negative = negative.broadcast_mul(&slope_t)?;
    Ok(positive.add(&scaled_negative)?)
}

fn save_tensor(name: &str, t: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let flat: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let dims = t.dims();
    let header = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let data = flat
        .iter()
        .map(|v| format!("{:.10}", v))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(format!("{}.txt", name), format!("{}\n{}\n", header, data))?;
    Ok(())
}

fn load_tensor_file(path: &str, device: &Device) -> Result<Tensor, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.trim().split('\n').collect();
    let dims: Vec<usize> = lines[0].split(',').map(|d| d.parse().unwrap()).collect();
    let data: Vec<f32> = lines[1..]
        .iter()
        .map(|s| s.trim().parse().unwrap())
        .collect();
    Ok(Tensor::from_vec(data, dims, device)?)
}
