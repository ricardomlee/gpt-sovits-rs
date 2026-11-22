use super::*;
use std::collections::HashMap;

// Tiny conditioning/sampling fixture; component numerical tests cover attention and flow.
fn synthesis_model() -> Result<SoVITSModel> {
    let device = &Device::Cpu;
    let dtype = DType::F32;
    let mut weights = HashMap::new();
    for (name, shape, value) in [
        ("enc_p.ssl_proj.weight", &[2, 2, 1][..], 0.25),
        ("enc_p.text_embedding.weight", &[4, 2][..], 0.1),
        ("enc_p.proj.weight", &[4, 2, 1][..], 0.1),
        ("ref_enc.spectral.0.fc.weight", &[2, 2][..], 0.1),
        ("ref_enc.spectral.0.fc.bias", &[2][..], 0.0),
        ("ref_enc.spectral.3.fc.weight", &[2, 2][..], 0.2),
        ("ref_enc.spectral.3.fc.bias", &[2][..], 0.0),
        ("ref_enc.temporal.0.conv1.conv.weight", &[4, 2, 1][..], 0.1),
        ("ref_enc.temporal.0.conv1.conv.bias", &[4][..], 0.0),
        ("ref_enc.temporal.1.conv1.conv.weight", &[4, 2, 1][..], 0.1),
        ("ref_enc.temporal.1.conv1.conv.bias", &[4][..], 0.0),
        ("ref_enc.slf_attn.w_qs.weight", &[2, 2][..], 0.1),
        ("ref_enc.slf_attn.w_qs.bias", &[2][..], 0.0),
        ("ref_enc.slf_attn.w_ks.weight", &[2, 2][..], 0.1),
        ("ref_enc.slf_attn.w_ks.bias", &[2][..], 0.0),
        ("ref_enc.slf_attn.w_vs.weight", &[2, 2][..], 0.1),
        ("ref_enc.slf_attn.w_vs.bias", &[2][..], 0.0),
        ("ref_enc.slf_attn.fc.weight", &[2, 2][..], 0.1),
        ("ref_enc.slf_attn.fc.bias", &[2][..], 0.0),
        ("ref_enc.fc.fc.weight", &[2, 2][..], 0.3),
        ("ref_enc.fc.fc.bias", &[2][..], 0.1),
        ("dec.conv_pre.weight", &[2, 2, 1][..], 0.1),
        ("dec.conv_post.weight", &[1, 2, 1][..], 0.2),
        ("dec.cond.weight", &[2, 2, 1][..], 0.1),
    ] {
        weights.insert(name.into(), Tensor::full(value as f32, shape, device)?);
    }
    let weights = StateDict::new(weights);
    Ok(SoVITSModel {
        device: device.clone(),
        dtype,
        quantizer: Quantizer::new(
            2,
            3,
            Tensor::new(&[[0.1f32, 0.2], [-0.2, 0.3], [0.4, -0.1]], device)?,
        ),
        enc_p: EncP::load(&weights, device, 2, 0, 2, dtype)?,
        enc_q: None,
        flow: ResidualCouplingBlock::load(&weights, "flow.flows", device, 0, dtype)?,
        decoder: Decoder::load(&weights, device, dtype)?,
        ref_enc: RefEnc::load(&weights, device, dtype)?,
        ge_to512: Some(Linear::new(
            Tensor::new(&[[1f32, 0.], [0., 1.]], device)?,
            None,
        )),
        sv_emb: Some(Linear::new(
            Tensor::full(0.00001f32, (2, 20480), device)?,
            None,
        )),
        prelu_weight: Some(Tensor::full(0.2f32, 2, device)?),
        n_mels: 2,
        sampling_rate: 32000,
        gin_channels: 2,
        sampling_rng: SamplingRng::default(),
    })
}

fn trace_values(directory: &Path, name: &str) -> Vec<f32> {
    std::fs::read_to_string(directory.join(format!("{name}.txt")))
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| line.parse().unwrap())
        .collect()
}

#[test]
fn trace_preserves_conditioning_speed_and_random_state() -> Result<()> {
    let model = synthesis_model()?;
    let mel = Tensor::new(&[[[0.2f32, 0.3, 0.4], [0.1, -0.2, 0.5]]], &Device::Cpu)?;
    let sv = Tensor::ones((1, 20480), DType::F32, &Device::Cpu)?;
    let probe = Tensor::zeros((1, 2, 5), DType::F32, &Device::Cpu)?;
    for speed in [1.0, 1.5] {
        for embedding in [None, Some(&sv)] {
            for seed in [0, 42] {
                model.set_sampling_seed(Some(seed))?;
                let normal = model.synthesize_with_speed_and_sv(
                    &[0, 1, 2],
                    &[1, 2],
                    Some(&mel),
                    embedding,
                    0.5,
                    speed,
                )?;
                let next_noise = model.sample_noise(&probe)?.to_vec3::<f32>()?;
                model.set_sampling_seed(Some(seed))?;
                let directory = tempfile::tempdir().unwrap();
                let (input, traced) = model.synthesize_internal(
                    &[0, 1, 2],
                    &[1, 2],
                    Some(&mel),
                    embedding,
                    0.5,
                    speed,
                    Some(directory.path()),
                )?;
                assert_eq!(normal, traced);
                assert_eq!(next_noise, model.sample_noise(&probe)?.to_vec3::<f32>()?);
                let time = if speed == 1.0 { 6 } else { 5 };
                assert_eq!(input.dims(), &[1, 2, time]);
                assert_eq!(traced.len(), time);
                for (name, expected) in [
                    ("sovits_debug_ge", model.compute_ge(Some(&mel), embedding)?),
                    ("sovits_debug_dec_input", input),
                ] {
                    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                    let actual = trace_values(directory.path(), name);
                    assert_eq!(actual.len(), expected.len());
                    assert!(actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 1e-8));
                }
                let audio =
                    std::fs::read_to_string(directory.path().join("sovits_debug_audio.txt"))
                        .unwrap();
                let audio: Vec<f32> = audio.lines().map(|line| line.parse().unwrap()).collect();
                assert_eq!(audio.len(), traced.len());
                assert!(audio.iter().zip(traced).all(|(a, b)| (a - b).abs() < 1e-8));
            }
        }
    }
    Ok(())
}

#[test]
fn public_debug_synthesis_returns_the_actual_decoder_input() -> Result<()> {
    let model = synthesis_model()?;
    model.set_sampling_seed(Some(7))?;
    let expected = model.synthesize(&[0, 1, 2], &[1, 2], None, 0.5)?;
    model.set_sampling_seed(Some(7))?;
    let (input, audio) = model.synthesize_debug(&[0, 1, 2], &[1, 2], None, 0.5)?;
    assert_eq!(expected, audio);
    let ge = model.compute_ge(None, None)?;
    assert_eq!(model.decoder.forward(&input, Some(&ge))?, audio);
    Ok(())
}

#[test]
fn trace_output_errors_are_returned_instead_of_panicking() -> Result<()> {
    let model = synthesis_model()?;
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("sovits_debug_audio.txt")).unwrap();
    let error = model
        .synthesize_internal(&[0, 1], &[1], None, None, 0.5, 1.0, Some(directory.path()))
        .unwrap_err();
    assert!(matches!(error, Error::InferenceError(_)));
    assert!(error.to_string().contains("sovits_debug_audio"), "{error}");
    Ok(())
}

fn check_upsampling(device: &Device) -> Result<()> {
    for (batch, channels, time) in [
        (1, 1, 1),
        (1, 2, 3),
        (2, 3, 5),
        (1, 768, 63),
        (1, 768, 64),
        (2, 768, 65),
        (0, 2, 3),
        (1, 0, 3),
        (1, 2, 0),
    ] {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let source =
                Tensor::arange(0f32, ((batch + 1) * channels * time) as f32, &Device::Cpu)?
                    .affine(1.0 / 256.0, -1.0)?
                    .to_dtype(dtype)?
                    .reshape((batch + 1, time, channels))?;
            let values = source
                .narrow(0, 1, batch)?
                .transpose(1, 2)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected: Vec<f32> = values.iter().flat_map(|&value| [value, value]).collect();
            let input = source
                .to_device(device)?
                .narrow(0, 1, batch)?
                .transpose(1, 2)?;
            for contiguous in [false, true] {
                let input = if contiguous {
                    input.contiguous()?
                } else {
                    input.clone()
                };
                let output = nearest_upsample_2x(&input)?;
                assert_eq!(output.dims(), &[batch, channels, time * 2]);
                assert_eq!(output.dtype(), dtype);
                assert!(output.device().same_device(device));
                assert_eq!(
                    output
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected,
                    "shape={:?}, dtype={dtype:?}, contiguous={contiguous}",
                    input.dims()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn nearest_upsample_preserves_layout_values_and_dtype() -> Result<()> {
    check_upsampling(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn nearest_upsample_cuda_preserves_layout_values_and_dtype() -> Result<()> {
    check_upsampling(&Device::new_cuda(0)?)
}

#[test]
fn nearest_upsample_repeats_each_frame() -> Result<()> {
    let input = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.]]], &Device::Cpu)?;
    let output = nearest_upsample_2x(&input)?;
    assert_eq!(output.dims(), &[1, 2, 6]);
    assert_eq!(output.dtype(), DType::F32);
    assert_eq!(
        output.flatten_all()?.to_vec1::<f32>()?,
        [1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6.]
    );
    Ok(())
}

#[test]
fn quantizer_decodes_codes_without_changing_layout() -> Result<()> {
    let device = Device::Cpu;
    let codebook = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;
    let quantizer = Quantizer::new(2, 3, codebook);
    let codes = Tensor::new(&[[2i64, 0], [1, 2]], &device)?;
    let output = quantizer.decode(&codes)?;
    assert_eq!(output.dims(), &[2, 2, 2]);
    assert_eq!(
        output.flatten_all()?.to_vec1::<f32>()?,
        [5., 1., 6., 2., 3., 5., 4., 6.]
    );
    Ok(())
}
