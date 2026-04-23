/// Detailed ConvTranspose1d test: verify single output element
use candle_core::{Device, DType, Tensor};

fn main() {
    let device = Device::Cpu;

    // Create a known input and weight
    // x: [1, 2, 3], w: [2, 4, 2], padding=0, stride=2
    // output: [1, 4, 6]

    let x = Tensor::from_vec(vec![
        1.0f32, 2.0f32, 3.0f32,  // channel 0
        4.0f32, 5.0f32, 6.0f32,  // channel 1
    ], (1, 2, 3), &device).unwrap();

    let w = Tensor::from_vec(vec![
        // in_ch=0, out_ch=0: [1.0, 2.0]
        1.0f32, 2.0f32,
        // in_ch=0, out_ch=1: [3.0, 4.0]
        3.0f32, 4.0f32,
        // in_ch=0, out_ch=2: [5.0, 6.0]
        5.0f32, 6.0f32,
        // in_ch=0, out_ch=3: [7.0, 8.0]
        7.0f32, 8.0f32,
        // in_ch=1, out_ch=0: [9.0, 10.0]
        9.0f32, 10.0f32,
        // in_ch=1, out_ch=1: [11.0, 12.0]
        11.0f32, 12.0f32,
        // in_ch=1, out_ch=2: [13.0, 14.0]
        13.0f32, 14.0f32,
        // in_ch=1, out_ch=3: [15.0, 16.0]
        15.0f32, 16.0f32,
    ], (2, 4, 2), &device).unwrap();

    let out = x.conv_transpose1d(&w, 0, 0, 2, 1, 1).unwrap();
    let out_val: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    println!("Candle output shape: {:?}", out.dims());
    println!("Candle output: {:?}", out_val);

    // PyTorch expected:
    // output_time = (3-1)*2 - 0 + 2 = 6
    // For y[0, c_out, t_out]:
    //   y[c_out, t] = sum over c_in, k of x[c_in, t_in] * w[c_in, c_out, k]
    //   where t_out = t_in * stride - padding + k
    //
    // t_out=0: k=0,t_in=0; k=1,t_in=-1(no)
    //   y[c_out,0] = x[0,0]*w[0,c_out,0] + x[1,0]*w[1,c_out,0]
    //   = 1*1 + 4*9 = 37 for c_out=0
    //   = 1*3 + 4*11 = 47 for c_out=1
    //   = 1*5 + 4*13 = 57 for c_out=2
    //   = 1*7 + 4*15 = 67 for c_out=3
    println!("\nExpected (PyTorch) for t_out=0:");
    println!("  c_out=0: 1*1 + 4*9 = {}", 1*1 + 4*9);
    println!("  c_out=1: 1*3 + 4*11 = {}", 1*3 + 4*11);
    println!("  c_out=2: 1*5 + 4*13 = {}", 1*5 + 4*13);
    println!("  c_out=3: 1*7 + 4*15 = {}", 1*7 + 4*15);

    // t_out=1: k=1,t_in=0; k=0,t_in=0.5(no)
    //   y[c_out,1] = x[0,0]*w[0,c_out,1] + x[1,0]*w[1,c_out,1]
    //   = 1*2 + 4*10 = 42 for c_out=0
    println!("\nExpected (PyTorch) for t_out=1:");
    println!("  c_out=0: 1*2 + 4*10 = {}", 1*2 + 4*10);

    // t_out=2: k=0,t_in=1; k=1,t_in=0.5(no); k=2,t_in=0(out of range)
    //   y[c_out,2] = x[0,1]*w[0,c_out,0] + x[1,1]*w[1,c_out,0]
    //   = 2*1 + 5*9 = 47 for c_out=0
    println!("\nExpected (PyTorch) for t_out=2:");
    println!("  c_out=0: 2*1 + 5*9 = {}", 2*1 + 5*9);

    // t_out=3: k=1,t_in=1; k=0,t_in=1.5(no)
    //   = 2*2 + 5*10 = 54 for c_out=0
    println!("\nExpected (PyTorch) for t_out=3:");
    println!("  c_out=0: 2*2 + 5*10 = {}", 2*2 + 5*10);

    // t_out=4: k=0,t_in=2
    //   = 3*1 + 6*9 = 57 for c_out=0
    println!("\nExpected (PyTorch) for t_out=4:");
    println!("  c_out=0: 3*1 + 6*9 = {}", 3*1 + 6*9);

    // t_out=5: k=1,t_in=2
    //   = 3*2 + 6*10 = 66 for c_out=0
    println!("\nExpected (PyTorch) for t_out=5:");
    println!("  c_out=0: 3*2 + 6*10 = {}", 3*2 + 6*10);

    println!("\nExpected full: [37,47,57,67, 42,53,64,75, 47,57,67,77, 54,66,78,90, 57,67,77,87, 66,78,90,102]");
}
