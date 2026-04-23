/// Test Candle conv_transpose1d at different scales to find where divergence occurs
use candle_core::{Device, DType, Tensor, D};
use std::collections::HashMap;
use gpt_sovits_rs::utils::load_safetensors;

fn main() {
    let device = Device::Cpu;
    
    // Load weights
    let weights = load_safetensors("models/sovits-model.safetensors").unwrap();
    let wv = weights["dec.ups.0.weight_v"].to_dtype(DType::F32).unwrap();
    let wg = weights["dec.ups.0.weight_g"].to_dtype(DType::F32).unwrap();
    let bias = weights["dec.ups.0.bias"].to_dtype(DType::F32).unwrap();
    
    // Apply weight norm
    let v_norm = wv.sqr().unwrap().sum(D::Minus1).unwrap().sum(D::Minus1).unwrap().sqrt().unwrap()
        .reshape((512, 1, 1)).unwrap();
    let weight = wv.broadcast_div(&v_norm).unwrap().broadcast_mul(&wg).unwrap();
    
    println!("Full weight shape: {:?}", weight.dims()); // [512, 256, 16]
    
    // Create random-ish input with known pattern (same in Rust and for Python comparison)
    // Using a deterministic pattern: sin/cos
    let test_sizes = [
        (4, 4, 10),     // tiny
        (16, 16, 20),   // small  
        (64, 64, 50),   // medium
        (128, 128, 100), // large
        (256, 256, 150), // xlarge
        (512, 256, 200), // full-ish
    ];
    
    for (in_c, out_c, t) in test_sizes {
        // Create input with varied values
        let mut inp_data = Vec::with_capacity(in_c * t);
        for i in 0..(in_c * t) {
            inp_data.push(((i % 1000) as f32 - 500.0) / 5000.0);
        }
        let inp = Tensor::from_vec(inp_data, (1, in_c, t), &device).unwrap();
        
        // Slice weight to match
        let w = weight.narrow(0, 0, in_c).unwrap().narrow(1, 0, out_c).unwrap();
        let b = bias.narrow(0, 0, out_c).unwrap();
        
        let out = inp.conv_transpose1d(&w, 3, 0, 10, 1, 1).unwrap();
        let b_r = b.reshape((1, out_c, 1)).unwrap();
        let out = out.broadcast_add(&b_r).unwrap();
        
        let out_flat: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        
        // Now compute y[0, 0, 0] manually
        // t_out=0: k = 3 - t_in*10, only t_in=0, k=3 is valid
        // y[c_out=0, t_out=0] = sum over c_in of x[c_in, 0] * w[c_in, 0, 3]
        let inp_flat: Vec<f32> = inp.flatten_all().unwrap().to_vec1().unwrap();
        let w_flat: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        
        let mut manual = 0.0f32;
        for c_in in 0..in_c {
            let x_val = inp_flat[c_in * t]; // x[c_in, t=0]
            let w_val = w_flat[c_in * out_c * 16 + 0 * 16 + 3]; // w[c_in, out=0, k=3]
            manual += x_val * w_val;
        }
        // Add bias
        let bias_val: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        manual += bias_val[0];
        
        let candle_val = out_flat[0]; // first element = y[0, c_out=0, t_out=0]
        let diff = (manual - candle_val).abs();
        
        println!("Size ({},{},{}): output shape={:?}, candle[0]={:.8}, manual={:.8}, diff={:.2e}", 
            in_c, out_c, t, out.dims(), candle_val, manual, diff);
    }
}
