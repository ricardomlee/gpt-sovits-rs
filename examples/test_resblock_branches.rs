/// Test: Compare individual resblock0 branches between Rust and Python
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};

fn relu(x: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    x.maximum(&zeros)
}

fn main() {
    let device = Device::Cpu;

    // Load debug_ups0 as input
    let content = std::fs::read_to_string("debug_ups0.txt").unwrap();
    let mut lines = content.lines();
    let shape_line = lines.next().unwrap();
    let shape: Vec<usize> = shape_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let data: Vec<f32> = lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    let x = Tensor::from_vec(data, &*shape, &device).unwrap();
    println!("Input shape: {:?}", x.dims());

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    let dilations = [1, 3, 5];
    let mut branch_sums = Vec::new();

    // Compute each branch separately and save
    for d_idx in 0..3 {
        let prefix1 = format!("dec.resblocks.0.convs1.{}", d_idx);
        let prefix2 = format!("dec.resblocks.0.convs2.{}", d_idx);

        let wg1 = state_dict.get(&format!("{}.weight_g", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let wv1 = state_dict.get(&format!("{}.weight_v", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let b1 = state_dict.get(&format!("{}.bias", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        
        let wv_shape = wv1.dims();
        let kernel_size = wv_shape[2];
        let dilation = dilations[d_idx];
        let padding = (kernel_size * dilation - dilation) / 2;

        let conv1 = Conv1dWeightNorm::new(wg1, wv1, Some(b1), 1, padding, dilation);

        let wg2 = state_dict.get(&format!("{}.weight_g", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let wv2 = state_dict.get(&format!("{}.weight_v", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        let b2 = state_dict.get(&format!("{}.bias", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
        
        let padding2 = (kernel_size - 1) / 2;
        let conv2 = Conv1dWeightNorm::new(wg2, wv2, Some(b2), 1, padding2, 1);

        // Branch computation: relu(x) → conv1 → relu → conv2
        let xt = relu(&x).unwrap();
        let xt = conv1.forward(&xt).unwrap();
        let xt = relu(&xt).unwrap();
        let xt = conv2.forward(&xt).unwrap();

        let xt_flat: Vec<f32> = xt.flatten_all().unwrap().to_vec1().unwrap();
        println!("Branch {} (dilation={}): first 5 = {:?}", d_idx, dilation, &xt_flat[..5]);

        // Save for Python comparison
        let header = format!("{}\n", xt.dims().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
        let data_str = xt_flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n");
        std::fs::write(format!("rust_resblock0_branch{}.txt", d_idx), format!("{}{}", header, data_str)).unwrap();
        
        branch_sums.push(xt);
    }

    // Sum of branches
    let mut sum = branch_sums[0].clone();
    for i in 1..3 {
        sum = sum.add(&branch_sums[i]).unwrap();
    }
    let sum_flat: Vec<f32> = sum.flatten_all().unwrap().to_vec1().unwrap();
    println!("\nSum of 3 branches first 5: {:?}", &sum_flat[..5]);

    // With residual: x + sum
    let x_plus_sum = x.add(&sum).unwrap();
    let xps_flat: Vec<f32> = x_plus_sum.flatten_all().unwrap().to_vec1().unwrap();
    println!("x + sum first 5: {:?}", &xps_flat[..5]);
}
