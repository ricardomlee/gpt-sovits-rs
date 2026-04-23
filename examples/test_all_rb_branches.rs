use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors, Conv1dWeightNorm};

fn relu(x: &Tensor) -> candle_core::Result<Tensor> {
    let zeros = Tensor::zeros_like(x)?;
    x.maximum(&zeros)
}

fn main() {
    let device = Device::Cpu;

    // Load ups0 output as input
    let content = std::fs::read_to_string("debug_ups0.txt").unwrap();
    let mut lines = content.lines();
    let shape_line = lines.next().unwrap();
    let shape: Vec<usize> = shape_line.split(',').map(|s| s.trim().parse::<usize>().unwrap()).collect();
    let data: Vec<f32> = lines.filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();
    let x = Tensor::from_vec(data, &*shape, &device).unwrap();

    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    let dilations = [1, 3, 5];
    let mut all_branch_diffs = Vec::new();

    for rb in 0..3 {
        for d_idx in 0..3 {
            let prefix1 = format!("dec.resblocks.{}.convs1.{}", rb, d_idx);
            let prefix2 = format!("dec.resblocks.{}.convs2.{}", rb, d_idx);

            let wg1 = state_dict.get(&format!("{}.weight_g", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
            let wv1 = state_dict.get(&format!("{}.weight_v", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
            let b1 = state_dict.get(&format!("{}.bias", prefix1)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
            let k = wv1.dims()[2];
            let d = dilations[d_idx];
            let pad1 = (k * d - d) / 2;
            let conv1 = Conv1dWeightNorm::new(wg1, wv1, Some(b1), 1, pad1, d);

            let wg2 = state_dict.get(&format!("{}.weight_g", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
            let wv2 = state_dict.get(&format!("{}.weight_v", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
            let b2 = state_dict.get(&format!("{}.bias", prefix2)).unwrap().to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
            let pad2 = (k - 1) / 2;
            let conv2 = Conv1dWeightNorm::new(wg2, wv2, Some(b2), 1, pad2, 1);

            let xt = relu(&x).unwrap();
            let xt = conv1.forward(&xt).unwrap();
            let xt = relu(&xt).unwrap();
            let xt = conv2.forward(&xt).unwrap();

            let xt_flat: Vec<f32> = xt.flatten_all().unwrap().to_vec1().unwrap();

            // Compare with Python
            let py_path = format!("py_rb{}_d{}.txt", rb, d_idx);
            let py_data: Vec<f32> = std::fs::read_to_string(&py_path).unwrap()
                .lines().filter(|l| !l.is_empty()).map(|l| l.trim().parse::<f32>().unwrap()).collect();

            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f32;
            for i in 0..xt_flat.len() {
                let d_val = (xt_flat[i] - py_data[i]).abs();
                max_diff = max_diff.max(d_val);
                sum_diff += d_val;
            }

            println!("rb{}_d{}: mean_diff={:.2e}, max_diff={:.2e}", rb, d_idx, sum_diff / xt_flat.len() as f32, max_diff);
            all_branch_diffs.push(max_diff);
        }
    }

    let overall_max = all_branch_diffs.iter().fold(0.0f32, |a, &b| a.max(b));
    println!("\nOverall max branch diff: {:.2e}", overall_max);
}
