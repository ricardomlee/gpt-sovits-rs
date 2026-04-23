use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};

fn relu(x: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    x.maximum(&zeros)
}

fn main() {
    let device = Device::Cpu;
    
    let content = std::fs::read_to_string("debug_ups0.txt").unwrap();
    let mut lines = content.lines();
    let shape_line = lines.next().unwrap();
    let shape: Vec<usize> = shape_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let data: Vec<f32> = lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    let x = Tensor::from_vec(data, &*shape, &device).unwrap();

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    // Resblock1 convs1.0 (kernel=7, dilation=1)
    let prefix1 = "dec.resblocks.1.convs1.0";
    let wg1 = state_dict.get(&format!("{}.weight_g", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let wv1 = state_dict.get(&format!("{}.weight_v", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let b1 = state_dict.get(&format!("{}.bias", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    
    // kernel_size = 7, dilation = 1, padding = (7*1-1)/2 = 3
    let conv1 = Conv1dWeightNorm::new(wg1, wv1, Some(b1), 1, 3, 1);
    
    let prefix2 = "dec.resblocks.1.convs2.0";
    let wg2 = state_dict.get(&format!("{}.weight_g", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let wv2 = state_dict.get(&format!("{}.weight_v", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let b2 = state_dict.get(&format!("{}.bias", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    
    let conv2 = Conv1dWeightNorm::new(wg2, wv2, Some(b2), 1, 3, 1);

    let xt = relu(&x).unwrap();
    let xt = conv1.forward(&xt).unwrap();
    let xt = relu(&xt).unwrap();
    let xt = conv2.forward(&xt).unwrap();
    
    let xt_flat: Vec<f32> = xt.flatten_all().unwrap().to_vec1().unwrap();
    println!("Rust resblock1 branch 0 first 5: {:?}", &xt_flat[..5]);

    // Compare with Python
    let py_data: Vec<f32> = std::fs::read_to_string("py_resblock1_branch0.txt").unwrap()
        .lines().filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    
    let diff: Vec<f32> = xt_flat.iter().zip(py_data.iter()).map(|(a,b)| (a-b).abs()).collect();
    println!("Mean diff: {:.2e}, Max: {:.2e}",
        diff.iter().sum::<f32>() / diff.len() as f32,
        diff.iter().fold(0.0f32, |a, &b| a.max(b)));
}
