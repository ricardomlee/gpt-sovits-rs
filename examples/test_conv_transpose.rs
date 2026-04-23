/// Test Candle ConvTranspose1d behavior
use candle_core::{Device, DType, Tensor};

fn main() {
    let device = Device::Cpu;

    // Simple test: single input, single output, kernel=1
    let x = Tensor::from_vec(vec![1.0f32], (1, 1, 1), &device).unwrap();
    let w = Tensor::from_vec(vec![2.0f32], (1, 1, 1), &device).unwrap();

    let out = x.conv_transpose1d(&w, 0, 0, 1, 1, 1).unwrap();
    let out_val: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    println!("Test 1: x=[1], w=[2], stride=1, pad=0 => {:?}", out_val);
    // PyTorch: conv_transpose1d([[1]], [[2]], padding=0, stride=1) = [[2]]

    // Test 2: stride=2
    let x2 = Tensor::from_vec(vec![1.0f32, 2.0f32], (1, 1, 2), &device).unwrap();
    let w2 = Tensor::from_vec(vec![1.0f32], (1, 1, 1), &device).unwrap();
    let out2 = x2.conv_transpose1d(&w2, 0, 0, 2, 1, 1).unwrap();
    let out2_val: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
    println!("Test 2: x=[1,2], w=[1], stride=2, pad=0 => {:?}", out2_val);
    // PyTorch: [1.0, 0.0, 2.0]

    // Test 3: match PyTorch test case
    let x3 = Tensor::ones((1, 2, 3), DType::F32, &device).unwrap();
    let w3 = Tensor::ones((2, 4, 2), DType::F32, &device).unwrap();
    let out3 = x3.conv_transpose1d(&w3, 0, 0, 2, 1, 1).unwrap();
    let out3_val: Vec<f32> = out3.flatten_all().unwrap().to_vec1().unwrap();
    println!("Test 3: x=ones(1,2,3), w=ones(2,4,2), stride=2, pad=0 => {:?}", out3_val);
    // PyTorch: all 2.0 values, shape (1,4,6) = 24 values of 2.0

    // Test 4: check output shape with padding
    let x4 = Tensor::ones((1, 2, 3), DType::F32, &device).unwrap();
    let out4 = x4.conv_transpose1d(&w3, 1, 0, 2, 1, 1).unwrap();
    println!("Test 4: with padding=1 => shape {:?}", out4.dims());
    // PyTorch: padding=1, stride=2, kernel=2 => (3-1)*2 - 2*1 + 2 = 4

    // Test 5: Real ups0 case (small portion)
    // Input: [1, 512, 200], Weight: [512, 256, 16], stride=10, padding=3
    let x5 = Tensor::ones((1, 4, 5), DType::F32, &device).unwrap();
    let w5 = Tensor::ones((4, 8, 3), DType::F32, &device).unwrap();
    let out5 = x5.conv_transpose1d(&w5, 0, 0, 2, 1, 1).unwrap();
    println!("Test 5: x=ones(1,4,5), w=ones(4,8,3), stride=2, pad=0 => shape {:?}", out5.dims());
    // PyTorch: (5-1)*2 - 0 + 3 = 11

    // Test 6: same with padding=1
    let out6 = x5.conv_transpose1d(&w5, 1, 0, 2, 1, 1).unwrap();
    println!("Test 6: same with padding=1 => shape {:?}", out6.dims());
    // PyTorch: (5-1)*2 - 2*1 + 3 = 9
}
