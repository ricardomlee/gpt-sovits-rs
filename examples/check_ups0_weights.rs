/// Compare weight normalization between Rust and Python for ups0
use candle_core::{Device, DType, Tensor, D};
use gpt_sovits_rs::utils::{StateDict, load_safetensors};

fn main() {
    let device = Device::Cpu;
    let weights_map = load_safetensors("models/sovits-model.safetensors").unwrap();
    let state_dict = StateDict::new(weights_map);

    let weight_v = state_dict.get("dec.ups.0.weight_v").unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let weight_g = state_dict.get("dec.ups.0.weight_g").unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();
    let bias = state_dict.get("dec.ups.0.bias").unwrap()
        .to_device(&device).unwrap().to_dtype(DType::F32).unwrap();

    println!("weight_v shape: {:?}", weight_v.dims());
    println!("weight_g shape: {:?}", weight_g.dims());
    println!("bias shape: {:?}", bias.dims());

    let v_sq = weight_v.sqr().unwrap();
    let v_sum = v_sq.sum(D::Minus1).unwrap().sum(D::Minus1).unwrap();
    let v_norm = v_sum.sqrt().unwrap();
    println!("norm range: {:.10} - {:.10}", 
        v_norm.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        v_norm.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );

    let v_norm_r = v_norm.reshape((512, 1, 1)).unwrap();
    let v_normalized = weight_v.broadcast_div(&v_norm_r).unwrap();
    let weight = v_normalized.broadcast_mul(&weight_g).unwrap();

    let w_flat: Vec<f32> = weight.flatten_all().unwrap().to_vec1().unwrap();
    println!("normalized weight range: {:.10} - {:.10}",
        w_flat.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        w_flat.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("normalized weight first 5: {:?}", &w_flat[..5]);

    // Save norm stats and weight for comparison
    let n_flat: Vec<f32> = v_norm.flatten_all().unwrap().to_vec1().unwrap();
    std::fs::write("rust_ups0_norm_stats.txt", 
        n_flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n")).unwrap();
    std::fs::write("rust_ups0_weight_norm.txt",
        w_flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n")).unwrap();
    println!("Saved rust_ups0_norm_stats.txt and rust_ups0_weight_norm.txt");

    let v_flat: Vec<f32> = weight_v.flatten_all().unwrap().to_vec1().unwrap();
    let g_flat: Vec<f32> = weight_g.flatten_all().unwrap().to_vec1().unwrap();
    std::fs::write("rust_ups0_weight_v.txt",
        v_flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n")).unwrap();
    std::fs::write("rust_ups0_weight_g.txt",
        g_flat.iter().map(|v| format!("{:.10}", v)).collect::<Vec<_>>().join("\n")).unwrap();
    println!("Saved rust_ups0_weight_v.txt and rust_ups0_weight_g.txt");
}
