use super::*;

const Q: [f32; 16] = [
    0.8, 0.1, 0.0, -0.2, 0.0, 0.6, 0.2, 0.0, -0.1, 0.0, 1.0, 0.3, 0.2, -0.4, 0.0, 0.7,
];
const K: [f32; 16] = [
    0.5, 0.0, -0.3, 0.1, 0.2, 0.9, 0.0, -0.1, 0.0, -0.2, 0.4, 0.6, -0.4, 0.1, 0.5, 0.3,
];
const V: [f32; 16] = [
    1.0, 0.1, 0.2, 0.0, -0.2, 0.7, 0.0, 0.4, 0.3, 0.0, 0.8, -0.1, 0.0, -0.3, 0.2, 1.1,
];
const FC: [f32; 16] = [
    0.75, 0.1, 0.0, 0.0, 0.0, -0.5, 0.2, 0.0, 0.0, 0.0, 1.25, -0.1, 0.3, 0.0, 0.0, 0.6,
];
const FC_BIAS: [f32; 4] = [0.1, -0.2, 0.05, 0.3];
const Q_BIAS: [f32; 4] = [0.1, -0.2, 0.3, -0.4];
const K_BIAS: [f32; 4] = [-0.3, 0.1, 0.2, -0.1];
const V_BIAS: [f32; 4] = [0.2, 0.3, -0.1, 0.05];
const EMBEDDING: [f32; 8] = [0.8, -0.2, 0.3, 0.1, -0.4, 0.7, -0.1, 0.6];
const EMBEDDING_BIAS: [f32; 2] = [0.15, -0.25];

fn assert_close(actual: &Tensor, expected: &[f64]) -> Result<()> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(actual) - expected).abs() < 2e-6 + expected.abs() * 1e-6,
            "sample {index}: {actual} != {expected}"
        );
    }
    Ok(())
}

fn linear_scalar(x: &[f64], weight: &[f32], bias: &[f32]) -> Vec<f64> {
    weight
        .chunks_exact(x.len())
        .enumerate()
        .map(|(row, weights)| {
            weights
                .iter()
                .zip(x)
                .map(|(&w, &x)| f64::from(w) * x)
                .sum::<f64>()
                + bias.get(row).copied().map_or(0.0, f64::from)
        })
        .collect()
}

fn mish_scalar(x: f64) -> f64 {
    x * x.exp().ln_1p().tanh()
}

fn attention() -> Result<MultiHeadSelfAttention> {
    let linear = |w: &[f32], b: &[f32]| -> Result<LinearNorm> {
        Ok(LinearNorm {
            weight: Tensor::from_slice(w, (4, 4), &Device::Cpu)?,
            bias: Tensor::from_slice(b, 4, &Device::Cpu)?,
        })
    };
    Ok(MultiHeadSelfAttention {
        w_q: linear(&Q, &Q_BIAS)?,
        w_k: linear(&K, &K_BIAS)?,
        w_v: linear(&V, &V_BIAS)?,
        fc: linear(&FC, &FC_BIAS)?,
        n_heads: 2,
        head_dim: 2,
    })
}

// Independent scalar softmax and head assembly; no Candle/model helpers.
fn attention_scalar(input: &[[f64; 4]], blocked: &[bool]) -> Vec<Vec<f64>> {
    let q: Vec<_> = input
        .iter()
        .map(|x| linear_scalar(x, &Q, &Q_BIAS))
        .collect();
    let k: Vec<_> = input
        .iter()
        .map(|x| linear_scalar(x, &K, &K_BIAS))
        .collect();
    let v: Vec<_> = input
        .iter()
        .map(|x| linear_scalar(x, &V, &V_BIAS))
        .collect();
    q.iter()
        .zip(input)
        .map(|(query, residual)| {
            let mut context = [0.0; 4];
            for head in 0..2 {
                let offset = head * 2;
                let scores: Vec<_> = k
                    .iter()
                    .zip(blocked)
                    .map(|(key, &blocked)| {
                        if blocked {
                            f64::NEG_INFINITY
                        } else {
                            (query[offset] * key[offset] + query[offset + 1] * key[offset + 1])
                                / 4f64.sqrt()
                        }
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|score| (score - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for channel in offset..offset + 2 {
                    context[channel] = weights
                        .iter()
                        .zip(&v)
                        .map(|(w, v)| w * v[channel] / sum)
                        .sum();
                }
            }
            linear_scalar(&context, &FC, &FC_BIAS)
                .iter()
                .zip(residual)
                .map(|(a, b)| a + b)
                .collect()
        })
        .collect()
}

fn reference_encoder() -> Result<RefEnc> {
    let spectral = LinearNorm {
        weight: Tensor::eye(4, DType::F32, &Device::Cpu)?,
        bias: Tensor::zeros(4, DType::F32, &Device::Cpu)?,
    };
    let mut weights = vec![0f32; 8 * 4];
    for channel in 0..4 {
        weights[channel * 4 + channel] = 1.0;
    }
    let temporal = Conv1dGLU {
        conv_weight: Tensor::from_vec(weights, (8, 4, 1), &Device::Cpu)?,
        conv_bias: Tensor::zeros(8, DType::F32, &Device::Cpu)?,
        padding: 0,
    };
    Ok(RefEnc {
        spectral_0: spectral.clone(),
        spectral_3: spectral,
        temporal_0: temporal.clone(),
        temporal_1: temporal,
        self_attn: attention()?,
        fc: LinearNorm {
            weight: Tensor::from_slice(&EMBEDDING, (2, 4), &Device::Cpu)?,
            bias: Tensor::new(&EMBEDDING_BIAS, &Device::Cpu)?,
        },
    })
}

#[test]
fn mish_matches_scalar_values_on_both_sides_of_zero() -> Result<()> {
    let values = [-5f32, -1.0, 0.0, 0.75, 4.0];
    let expected: Vec<_> = values.iter().map(|&x| mish_scalar(f64::from(x))).collect();
    assert_close(&mish(&Tensor::new(&values, &Device::Cpu)?)?, &expected)
}

#[test]
fn convolution_glu_matches_scalar_padding_bias_and_channel_gate() -> Result<()> {
    let weights = [0.5f32, -0.75, 0.2, -0.3, 0.6, 0.4];
    let bias = [0.1f32, -0.2];
    let model = Conv1dGLU {
        conv_weight: Tensor::from_slice(&weights, (2, 1, 3), &Device::Cpu)?,
        conv_bias: Tensor::new(&bias, &Device::Cpu)?,
        padding: 1,
    };
    let input = [[0.2f32, -0.5, 1.2, 0.7], [-0.8, 0.6, 0.1, -0.3]];
    let mut expected = Vec::new();
    for samples in input {
        for time in 0..4 {
            let mut conv = bias.map(f64::from);
            for (channel, output) in conv.iter_mut().enumerate() {
                for tap in 0..3 {
                    let position = time as isize + tap as isize - 1;
                    if (0..4).contains(&position) {
                        *output += f64::from(weights[channel * 3 + tap])
                            * f64::from(samples[position as usize]);
                    }
                }
            }
            expected.push(f64::from(samples[time]) + conv[0] / (1.0 + (-conv[1]).exp()));
        }
    }
    let output = model.forward(&Tensor::new(&input, &Device::Cpu)?.reshape((2, 1, 4))?)?;
    assert_eq!(output.dims(), &[2, 1, 4]);
    assert_close(&output, &expected)
}

#[test]
fn attention_matches_scalar_heads_scaling_projection_and_key_mask() -> Result<()> {
    let input = [
        [
            [0.2f32, -0.5, 0.7, 1.1],
            [1.0, 0.3, -0.4, 0.6],
            [4.0, -3.0, 2.0, -1.0],
        ],
        [
            [0.8, -0.6, 0.4, -0.2],
            [-2.0, 1.0, 3.0, -4.0],
            [0.5, 0.1, -0.3, 0.9],
        ],
    ];
    let blocked = [[false, false, true], [false, true, false]];
    let model = attention()?;
    for masked in [false, true] {
        let mut expected = Vec::new();
        let mut mask = Vec::new();
        for batch in 0..2 {
            let scalar_input: Vec<_> = input[batch].iter().map(|row| row.map(f64::from)).collect();
            let keys = if masked { blocked[batch] } else { [false; 3] };
            expected.extend(attention_scalar(&scalar_input, &keys).concat());
            for _ in 0..3 {
                mask.extend(keys.map(|key| f32::from(u8::from(key))));
            }
        }
        let mask = Tensor::from_vec(mask, (2, 3, 3), &Device::Cpu)?;
        let output = model.forward(&Tensor::new(&input, &Device::Cpu)?, masked.then_some(&mask))?;
        assert_eq!(output.dims(), &[2, 3, 4]);
        assert_close(&output, &expected)?;
    }
    Ok(())
}

#[test]
fn temporal_pool_uses_each_samples_valid_count_and_ignores_padding() -> Result<()> {
    let model = reference_encoder()?;
    let input = Tensor::new(
        &[
            [[1f32, 10.0], [3.0, 30.0], [100.0, -100.0]],
            [[-2.0, 8.0], [200.0, -200.0], [400.0, 100.0]],
        ],
        &Device::Cpu,
    )?;
    assert_close(
        &model.temporal_avg_pool(&input, None)?,
        &[104.0 / 3.0, -20.0, 598.0 / 3.0, -92.0 / 3.0],
    )?;
    let mask = Tensor::new(&[[1f32, 1.0, 0.0], [1.0, 0.0, 0.0]], &Device::Cpu)?;
    for batch_size in [1, 2] {
        let output = model.temporal_avg_pool(
            &input.narrow(0, 0, batch_size)?,
            Some(&mask.narrow(0, 0, batch_size)?),
        )?;
        assert_eq!(output.dims(), &[batch_size, 2]);
        assert_close(&output, &[2.0, 20.0, -2.0, 8.0][..batch_size * 2])?;
    }
    Ok(())
}

#[test]
fn reference_encoder_matches_upstream_with_unequal_reference_lengths() -> Result<()> {
    let input = [
        [
            [0.2f32, -0.5, 0.7, 1.1],
            [1.0, 0.3, -0.4, 0.6],
            [4.0, -3.0, 2.0, -1.0],
        ],
        [
            [0.8, -0.6, 0.4, -0.2],
            [-2.0, 1.0, 3.0, -4.0],
            [0.5, 0.1, -0.3, 0.9],
        ],
    ];
    let valid = [[true, true, false], [true, false, false]];
    // MelStyleEncoder.eval() at upstream bf81cdb, with reference_encoder()'s weights.
    let expected = [2.39390087, 0.47957820, 2.51616693, -1.11406469];
    let model = reference_encoder()?;
    let input = Tensor::new(&input, &Device::Cpu)?
        .transpose(1, 2)?
        .contiguous()?;
    let mask: Vec<_> = valid
        .iter()
        .flatten()
        .map(|&valid| f32::from(u8::from(valid)))
        .collect();
    let mask = Tensor::from_vec(mask, (2, 1, 3), &Device::Cpu)?;
    for batch_size in [1, 2] {
        let output = model.forward(
            &input.narrow(0, 0, batch_size)?,
            &mask.narrow(0, 0, batch_size)?,
        )?;
        assert_eq!(output.dims(), &[batch_size, 2, 1]);
        assert_close(&output, &expected[..batch_size * 2])?;
    }
    Ok(())
}
