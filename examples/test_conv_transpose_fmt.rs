/// Test if Candle expects different weight format for conv_transpose1d
use candle_core::{Device, DType, Tensor};

fn main() {
    let device = Device::Cpu;

    // PyTorch ConvTranspose1d expects weight [in_ch, out_ch, kernel]
    // Let's verify what Candle expects

    // Test: input [1, 2, 3], we want output [1, 4, 6]
    // With PyTorch: weight [2, 4, 2] (in=2, out=4, k=2)

    let x = Tensor::from_vec(vec![1.0f32; 6], (1, 2, 3), &device).unwrap();

    // Format A: [in_ch, out_ch, kernel] (PyTorch format)
    let w_a = Tensor::from_vec(vec![1.0f32; 16], (2, 4, 2), &device).unwrap();
    let out_a = x.conv_transpose1d(&w_a, 0, 0, 2, 1, 1);
    match out_a {
        Ok(out) => {
            let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            println!("Format A [in=2, out=4, k=2]: OK, shape={:?}, first 6: {:?}", out.dims(), &v[..6]);
        }
        Err(e) => println!("Format A [in=2, out=4, k=2]: ERROR: {}", e),
    }

    // Format B: [out_ch, in_ch, kernel] (transposed)
    let w_b = Tensor::from_vec(vec![1.0f32; 16], (4, 2, 2), &device).unwrap();
    let out_b = x.conv_transpose1d(&w_b, 0, 0, 2, 1, 1);
    match out_b {
        Ok(out) => {
            let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            println!("Format B [out=4, in=2, k=2]: OK, shape={:?}, first 6: {:?}", out.dims(), &v[..6]);
        }
        Err(e) => println!("Format B [out=4, in=2, k=2]: ERROR: {}", e),
    }

    // Now test with actual ups0 weights
    // Compare: does the output match PyTorch when using the same input and weight?
    println!("\n=== Testing with actual ups0 weights ===");

    // We need to load from safetensors
    // For now, just print the expected shapes
    println!("Expected: input [1, 512, 200], weight [512, 256, 16], output [1, 256, 2000]");
    println!("PyTorch weight format: [in_ch, out_ch, kernel] = [512, 256, 16]");
    println!("If Candle expects [out_ch, in_ch, kernel], weight would need to be [256, 512, 16]");
}
