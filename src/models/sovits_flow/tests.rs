use super::*;
use std::collections::HashMap;

fn assert_close(actual: &Tensor, expected: &[f64]) -> Result<()> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(actual) - expected).abs() < 2e-6,
            "sample {index}: {actual} != {expected}"
        );
    }
    Ok(())
}

fn gate(a: f64, b: f64) -> f64 {
    a.tanh() / (1.0 + (-b).exp())
}

fn pointwise(weights: &[f32], bias: &[f32]) -> Result<Conv1dWeightNorm> {
    Conv1dWeightNorm::new_with_cached(
        Tensor::from_vec(
            weights.iter().map(|w| w.abs()).collect(),
            (weights.len(), 1, 1),
            &Device::Cpu,
        )?,
        Tensor::from_slice(weights, (weights.len(), 1, 1), &Device::Cpu)?,
        Some(Tensor::new(bias, &Device::Cpu)?),
        1,
        0,
        1,
    )
}

fn wave_net() -> Result<WN> {
    Ok(WN {
        in_layers: vec![
            pointwise(&[0.5, -0.75], &[0.1, -0.2])?,
            pointwise(&[-0.4, 0.6], &[-0.3, 0.25])?,
        ],
        res_skip_layers: vec![
            pointwise(&[0.8, -0.45], &[0.05, 0.07])?,
            pointwise(&[0.65], &[-0.02])?,
        ],
        cond_layer: Some(pointwise(
            &[0.3, -0.2, 0.7, -0.5],
            &[0.1, -0.05, -0.15, 0.2],
        )?),
        n_layers: 2,
    })
}

// Scalar oracle: two gated layers, one residual update and two skip outputs.
fn wave_net_scalar(x: f64, g: Option<f64>, valid: bool) -> f64 {
    if !valid {
        return 0.0;
    }
    let condition = |weight, bias| g.map_or(0.0, |g| weight * g + bias);
    let first = gate(
        0.5 * x + 0.1 + condition(0.3, 0.1),
        -0.75 * x - 0.2 + condition(-0.2, -0.05),
    );
    let residual = x + 0.8 * first + 0.05;
    let second = gate(
        -0.4 * residual - 0.3 + condition(0.7, -0.15),
        0.6 * residual + 0.25 + condition(-0.5, 0.2),
    );
    -0.45 * first + 0.07 + 0.65 * second - 0.02
}

fn coupling(mean_only: bool, offset: f32) -> Result<ResidualCouplingLayer> {
    Ok(ResidualCouplingLayer {
        pre: pointwise(&[0.7], &[-0.1])?,
        enc: wave_net()?,
        post: if mean_only {
            pointwise(&[0.4], &[offset])?
        } else {
            pointwise(&[0.4, -0.2], &[offset, 0.1])?
        },
        half_channels: 1,
        mean_only,
    })
}

fn insert_conv(
    weights: &mut HashMap<String, Tensor>,
    prefix: &str,
    conv: &Conv1dWeightNorm,
    normalized: bool,
) -> Result<()> {
    if normalized {
        weights.insert(format!("{prefix}.weight_g"), conv.weight_g.clone());
        weights.insert(format!("{prefix}.weight_v"), conv.weight_v.clone());
    } else {
        weights.insert(format!("{prefix}.weight"), conv.get_weight()?);
    }
    if let Some(bias) = &conv.bias {
        weights.insert(format!("{prefix}.bias"), bias.clone());
    }
    Ok(())
}

fn checkpoint(indices: &[usize], normalized: bool) -> Result<StateDict> {
    let mut weights = HashMap::new();
    for (position, index) in indices.iter().enumerate() {
        let prefix = format!("flow.flows.{index}");
        let layer = coupling(false, 0.1 - position as f32 * 0.4)?;
        insert_conv(
            &mut weights,
            &format!("{prefix}.pre"),
            &layer.pre,
            normalized,
        )?;
        insert_conv(
            &mut weights,
            &format!("{prefix}.post"),
            &layer.post,
            normalized,
        )?;
        for (group, layers) in [
            ("in_layers", &layer.enc.in_layers),
            ("res_skip_layers", &layer.enc.res_skip_layers),
        ] {
            for (i, conv) in layers.iter().enumerate() {
                insert_conv(
                    &mut weights,
                    &format!("{prefix}.enc.{group}.{i}"),
                    conv,
                    true,
                )?;
            }
        }
        insert_conv(
            &mut weights,
            &format!("{prefix}.enc.cond_layer"),
            layer.enc.cond_layer.as_ref().unwrap(),
            true,
        )?;
    }
    Ok(StateDict::new(weights))
}

#[test]
fn wave_net_loader_rejects_empty_layers() -> Result<()> {
    let error = WN::load(
        &StateDict::new(HashMap::new()),
        "flow.flows.0.enc",
        &Device::Cpu,
        DType::F32,
    )
    .expect_err("an empty WaveNet must not load as a zero-output encoder");
    assert!(matches!(error, crate::Error::ModelLoadError(_)));
    assert!(error.to_string().contains("flow.flows.0.enc"));
    Ok(())
}

#[test]
fn wave_net_loader_rejects_mismatched_input_and_residual_layers() -> Result<()> {
    for group in ["in_layers", "res_skip_layers"] {
        let mut weights = checkpoint(&[0], false)?;
        weights.remove(&format!("flow.flows.0.enc.{group}.1.weight_v"))?;
        let error = WN::load(&weights, "flow.flows.0.enc", &Device::Cpu, DType::F32)
            .expect_err("incomplete WaveNet layers must fail before inference");
        assert!(matches!(error, crate::Error::ModelLoadError(_)));
        assert!(error.to_string().contains("flow.flows.0.enc"));
    }
    Ok(())
}

#[test]
fn wave_net_loader_rejects_matching_gaps_in_both_layer_groups() -> Result<()> {
    let original = checkpoint(&[0], false)?;
    let weights = StateDict::new(
        original
            .keys()
            .map(|key| {
                let renamed = key
                    .replace(".enc.in_layers.1.", ".enc.in_layers.2.")
                    .replace(".enc.res_skip_layers.1.", ".enc.res_skip_layers.2.");
                Ok((renamed, original.get(key)?.clone()))
            })
            .collect::<Result<_>>()?,
    );
    let error = WN::load(&weights, "flow.flows.0.enc", &Device::Cpu, DType::F32)
        .expect_err("paired gaps must not silently truncate the WaveNet");
    assert!(matches!(error, crate::Error::ModelLoadError(_)));
    assert!(error.to_string().contains("flow.flows.0.enc"));
    Ok(())
}

#[test]
fn flow_loader_rejects_missing_and_extra_layers() -> Result<()> {
    for indices in [&[][..], &[0][..], &[2][..], &[0, 2, 4][..]] {
        let weights = checkpoint(indices, false)?;
        let error =
            ResidualCouplingBlock::load(&weights, "flow.flows", &Device::Cpu, 2, DType::F32)
                .expect_err("coupling layer count must match the requested architecture");
        assert!(matches!(error, crate::Error::ModelLoadError(_)));
        assert!(error.to_string().contains("flow.flows"));
    }
    Ok(())
}

#[test]
fn flow_loader_preserves_regular_and_normalized_projection_weights() -> Result<()> {
    let reference = ResidualCouplingBlock {
        layers: vec![coupling(false, 0.1)?, coupling(false, -0.3)?],
    };
    let input = Tensor::new(&[[[0.2f32, -0.6, 7.0], [0.8, -0.3, -9.0]]], &Device::Cpu)?;
    let mask = Tensor::new(&[[[1f32, 1.0, 0.0]]], &Device::Cpu)?;
    let g = Tensor::full(0.4f32, (1, 1, 1), &Device::Cpu)?;
    for indices in [[0, 2], [0, 8]] {
        for normalized in [false, true] {
            let weights = checkpoint(&indices, normalized)?;
            let loaded =
                ResidualCouplingBlock::load(&weights, "flow.flows", &Device::Cpu, 2, DType::F32)?;
            for reverse in [false, true] {
                let expected = reference
                    .forward(&input, &mask, Some(&g), reverse)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                assert_close(
                    &loaded.forward(&input, &mask, Some(&g), reverse)?,
                    &expected,
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn zero_initialized_regular_coupling_is_an_exact_identity() -> Result<()> {
    let original = checkpoint(&[0], false)?;
    let weights = StateDict::new(
        original
            .keys()
            .map(|key| {
                let value = original.get(key)?;
                let value = if matches!(
                    key.as_str(),
                    "flow.flows.0.post.weight" | "flow.flows.0.post.bias"
                ) {
                    Tensor::zeros_like(value)?
                } else {
                    value.clone()
                };
                Ok((key.clone(), value))
            })
            .collect::<Result<_>>()?,
    );
    let layer =
        ResidualCouplingLayer::load(&weights, "flow.flows.0", &Device::Cpu, false, DType::F32)?;
    let input = Tensor::new(&[[[0.2f32, -0.6, 7.0], [0.8, -0.3, -9.0]]], &Device::Cpu)?;
    let mask = Tensor::new(&[[[1f32, 1.0, 0.0]]], &Device::Cpu)?;
    let g = Tensor::full(0.4f32, (1, 1, 1), &Device::Cpu)?;
    for reverse in [false, true] {
        let output = layer.forward(&input, &mask, Some(&g), reverse)?;
        assert_eq!(output.to_vec3::<f32>()?, input.to_vec3::<f32>()?);
    }
    Ok(())
}

fn coupling_scalar(
    x: [f64; 2],
    valid: bool,
    g: Option<f64>,
    offset: f64,
    mean_only: bool,
    reverse: bool,
) -> [f64; 2] {
    let h = wave_net_scalar(0.7 * x[0] - 0.1, g, valid);
    let mean = if valid { 0.4 * h + offset } else { 0.0 };
    let logs = if valid && !mean_only {
        -0.2 * h + 0.1
    } else {
        0.0
    };
    let second = if reverse {
        (x[1] - mean) * (-logs).exp()
    } else {
        mean + x[1] * logs.exp()
    };
    [x[0], second]
}

#[test]
fn conditioned_gate_adds_before_nonlinearity_and_broadcasts_over_time() -> Result<()> {
    let a = Tensor::arange(0f32, 24f32, &Device::Cpu)?
        .affine(0.15, -1.0)?
        .reshape((2, 4, 3))?;
    let b = Tensor::new(
        &[
            [[0.4f32], [-0.2], [0.7], [-0.5]],
            [[-0.3], [0.6], [-0.4], [0.2]],
        ],
        &Device::Cpu,
    )?;
    let av = a.to_vec3::<f32>()?;
    let bv = b.to_vec3::<f32>()?;
    let mut expected = Vec::new();
    for batch in 0..2 {
        for channel in 0..2 {
            for (&tanh_input, &sigmoid_input) in
                av[batch][channel].iter().zip(&av[batch][channel + 2])
            {
                expected.push(gate(
                    f64::from(tanh_input) + f64::from(bv[batch][channel][0]),
                    f64::from(sigmoid_input) + f64::from(bv[batch][channel + 2][0]),
                ));
            }
        }
    }
    let output = fused_add_tanh_sigmoid_multiply(&a, &b, 2, true)?;
    assert_eq!(output.dims(), &[2, 2, 3]);
    assert_close(&output, &expected)
}

#[test]
fn wave_net_matches_scalar_residual_skip_conditioning_and_mask() -> Result<()> {
    let model = wave_net()?;
    let input = Tensor::new(
        &[[[0.2f32, -0.6, 40.0]], [[-0.4, 0.8, -30.0]]],
        &Device::Cpu,
    )?;
    let mask = Tensor::new(&[[[1f32, 1.0, 0.0]], [[1.0, 0.0, 1.0]]], &Device::Cpu)?;
    let inputs = input.flatten_all()?.to_vec1::<f32>()?;
    let masks = mask.flatten_all()?.to_vec1::<f32>()?;
    for conditioning in [None, Some(0.4f32)] {
        let g = conditioning
            .map(|g| Tensor::full(g, (2, 1, 1), &Device::Cpu))
            .transpose()?;
        let expected: Vec<_> = inputs
            .iter()
            .zip(&masks)
            .map(|(&x, &m)| wave_net_scalar(f64::from(x), conditioning.map(f64::from), m != 0.0))
            .collect();
        assert_close(&model.forward(&input, &mask, g.as_ref())?, &expected)?;
    }
    Ok(())
}

#[test]
fn coupling_forward_and_inverse_match_scalar_affine_and_mean_only_maps() -> Result<()> {
    let input = Tensor::new(&[[[0.2f32, -0.6, 7.0], [0.8, -0.3, -9.0]]], &Device::Cpu)?;
    let mask = Tensor::new(&[[[1f32, 1.0, 0.0]]], &Device::Cpu)?;
    let g = Tensor::full(0.4f32, (1, 1, 1), &Device::Cpu)?;
    let values = input.to_vec3::<f32>()?;
    for mean_only in [false, true] {
        let layer = coupling(mean_only, 0.1)?;
        for reverse in [false, true] {
            let mut expected = [vec![], vec![]];
            for (time, (&x0, &x1)) in values[0][0].iter().zip(&values[0][1]).enumerate() {
                let pair = coupling_scalar(
                    [f64::from(x0), f64::from(x1)],
                    time < 2,
                    Some(0.4),
                    0.1,
                    mean_only,
                    reverse,
                );
                expected[0].push(pair[0]);
                expected[1].push(pair[1]);
            }
            assert_close(
                &layer.forward(&input, &mask, Some(&g), reverse)?,
                &expected.concat(),
            )?;
        }
    }
    Ok(())
}

#[test]
fn flow_block_reverses_layer_order_and_channel_flips() -> Result<()> {
    let block = ResidualCouplingBlock {
        layers: vec![coupling(false, 0.1)?, coupling(false, -0.3)?],
    };
    let input = Tensor::new(&[[[0.2f32, -0.6, 7.0], [0.8, -0.3, -9.0]]], &Device::Cpu)?;
    let mask = Tensor::new(&[[[1f32, 1.0, 0.0]]], &Device::Cpu)?;
    let g = Tensor::full(0.4f32, (1, 1, 1), &Device::Cpu)?;
    let values = input.to_vec3::<f32>()?;
    for reverse in [false, true] {
        let mut expected = [vec![], vec![]];
        for (time, (&x0, &x1)) in values[0][0].iter().zip(&values[0][1]).enumerate() {
            let mut pair = [f64::from(x0), f64::from(x1)];
            let offsets = if reverse { [-0.3, 0.1] } else { [0.1, -0.3] };
            for offset in offsets {
                if reverse {
                    pair.swap(0, 1);
                }
                pair = coupling_scalar(pair, time < 2, Some(0.4), offset, false, reverse);
                if !reverse {
                    pair.swap(0, 1);
                }
            }
            expected[0].push(pair[0]);
            expected[1].push(pair[1]);
        }
        assert_close(
            &block.forward(&input, &mask, Some(&g), reverse)?,
            &expected.concat(),
        )?;
    }
    let encoded = block.forward(&input, &mask, Some(&g), false)?;
    assert_close(
        &block.forward(&encoded, &mask, Some(&g), true)?,
        &input
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect::<Vec<_>>(),
    )
}
