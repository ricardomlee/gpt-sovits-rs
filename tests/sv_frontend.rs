use candle_core::{Device, Tensor};
use gpt_sovits_rs::models::sv::{features::fbank, validate_weights};

#[test]
fn kaldi_fbank_matches_upstream_f32_synthetic_fixture() -> anyhow::Result<()> {
    let tensors = candle_core::safetensors::load_buffer(
        include_bytes!("fixtures/sv/synthetic_fbank.safetensors"),
        &Device::Cpu,
    )?;
    let samples = tensors["samples"].to_vec1::<f32>()?;
    let expected = &tensors["fbank"];
    let actual = fbank(&samples, &Device::Cpu)?;
    assert_eq!(actual.dims(), expected.dims());
    let error = (&actual - expected)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    // F32 FFT/window reductions differ by a few ULPs; log amplifies tiny bin energies.
    assert!(error < 5e-4, "fbank max absolute error {error}");
    let relative_l2 = ((&actual - expected)?.sqr()?.sum_all()?.to_scalar::<f32>()?
        / expected.sqr()?.sum_all()?.to_scalar::<f32>()?)
    .sqrt();
    assert!(relative_l2 < 1e-5, "fbank relative L2 error {relative_l2}");
    Ok(())
}

#[test]
fn wrong_encoder_weights_fail_before_inference() {
    let weights = std::collections::HashMap::from([(
        "conv1.weight".to_string(),
        Tensor::zeros((1, 1, 1, 1), candle_core::DType::F32, &Device::Cpu).unwrap(),
    )]);
    assert!(validate_weights(weights).is_err());
}
