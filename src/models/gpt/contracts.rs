use super::sampling::DecoderSampler;
use super::*;
use std::collections::HashMap;

// Zero attention/FFN and EOS-dominant logits isolate decoder policy from learned behavior.
fn model() -> GPTModel {
    model_on(&Device::Cpu)
}

fn model_on(target_device: &Device) -> GPTModel {
    let device = &Device::Cpu;
    let mut weights = HashMap::new();
    for (key, rows, offset) in [
        ("model.ar_text_embedding.word_embeddings.weight", 8, 0.0),
        ("model.ar_audio_embedding.word_embeddings.weight", 7, 8.0),
    ] {
        let values: Vec<f32> = (0..rows * 16)
            .map(|i| {
                if i % 16 == 0 {
                    128.0 + offset
                } else {
                    (i / 16) as f32 / 8.0 + offset
                }
            })
            .collect();
        weights.insert(
            key.to_owned(),
            Tensor::from_vec(values, (rows, 16), device).unwrap(),
        );
    }
    for (key, rows, cols) in [
        ("model.h.layers.0.self_attn.in_proj_weight", 48, 16),
        ("model.h.layers.0.self_attn.out_proj.weight", 16, 16),
        ("model.h.layers.0.linear1.weight", 32, 16),
        ("model.h.layers.0.linear2.weight", 16, 32),
    ] {
        weights.insert(
            key.to_owned(),
            Tensor::zeros((rows, cols), DType::F32, device).unwrap(),
        );
    }
    for (key, len) in [
        ("model.h.layers.0.self_attn.in_proj_bias", 48),
        ("model.h.layers.0.self_attn.out_proj.bias", 16),
        ("model.h.layers.0.linear1.bias", 32),
        ("model.h.layers.0.linear2.bias", 16),
        ("model.h.layers.0.norm1.bias", 16),
        ("model.h.layers.0.norm2.bias", 16),
    ] {
        weights.insert(
            key.to_owned(),
            Tensor::zeros(len, DType::F32, device).unwrap(),
        );
    }
    for key in [
        "model.h.layers.0.norm1.weight",
        "model.h.layers.0.norm2.weight",
    ] {
        weights.insert(
            key.to_owned(),
            Tensor::ones(16, DType::F32, device).unwrap(),
        );
    }
    let mut output = vec![0.0f32; 7 * 16];
    // EOS is always the raw argmax; masking must keep the first eleven steps alive.
    output[6 * 16] = 100.0;
    weights.insert(
        "model.ar_predict_layer.weight".into(),
        Tensor::from_vec(output, (7, 16), device).unwrap(),
    );
    let mut projection = vec![0.0f32; 16 * 1024];
    for i in 0..16 {
        projection[i * 1024 + i] = 1.0;
    }
    weights.insert(
        "model.bert_proj.weight".into(),
        Tensor::from_vec(projection, (16, 1024), device).unwrap(),
    );
    weights.insert(
        "model.bert_proj.bias".into(),
        Tensor::from_vec((0..16).map(|i| i as f32 / 16.0).collect(), 16, device).unwrap(),
    );
    for (key, alpha) in [
        ("model.ar_text_position.alpha", 1.25f32),
        ("model.ar_audio_position.alpha", 0.75),
    ] {
        weights.insert(key.to_owned(), Tensor::new(&[alpha], device).unwrap());
    }
    let file = tempfile::NamedTempFile::new().unwrap();
    candle_core::safetensors::save(&weights, file.path()).unwrap();
    GPTModel::load_with_device(file.path().to_str().unwrap(), target_device, DType::F32).unwrap()
}

fn generate(model: &GPTModel, mode: &str, phones: &[usize], max_tokens: usize) -> Vec<usize> {
    let prompt = &[2, 4];
    let bert = Tensor::full(0.125f32, (1, phones.len(), 16), &Device::Cpu).unwrap();
    match mode {
        "plain" => model.generate_with_prompts_aligned_bert(
            phones,
            prompt,
            Some(&bert),
            0,
            1.0,
            0.8,
            1.0,
            max_tokens,
        ),
        "kv" => model.generate_with_prompts_aligned_bert_kv_cache(
            phones,
            prompt,
            Some(&bert),
            0,
            1.0,
            0.8,
            1.0,
            max_tokens,
        ),
        "static" => model.generate_with_static_kv(
            phones,
            prompt,
            Some(&bert),
            0,
            1.0,
            0.8,
            1.0,
            max_tokens,
            64,
        ),
        "graph-fallback" => model.generate_with_cuda_graph(
            phones,
            prompt,
            Some(&bert),
            0,
            1.0,
            0.8,
            1.0,
            max_tokens,
            64,
        ),
        _ => unreachable!(),
    }
    .unwrap()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn cuda_graph_param_cache_preserves_seeded_decode_across_capacities() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    let device = crate::Config::builder()
        .with_device("cuda")
        .build()
        .candle_device();
    assert!(device.is_cuda(), "CUDA test must not fall back to CPU");
    let mut model = model_on(&device);
    let mut output = vec![0f32; 7 * 16];
    output[6 * 16] = -100.0;
    model.ar_predict_layer = Tensor::from_vec(output, (7, 16), &device).unwrap();
    model.ar_predict_layer_t = model.ar_predict_layer.t().unwrap().contiguous().unwrap();
    let phones = &[1, 3, 2];
    let prompt = &[2, 4];
    let bert = Tensor::full(0.125f32, (1, phones.len(), 16), &device).unwrap();
    for (seed, capacity) in [(11, 64), (42, 80), (11, 64)] {
        model.set_sampling_seed(Some(seed)).unwrap();
        let expected = model
            .generate_with_prompts_aligned_bert_kv_cache(
                phones,
                prompt,
                Some(&bert),
                0,
                1.0,
                0.8,
                1.0,
                48,
            )
            .unwrap();
        model.set_sampling_seed(Some(seed)).unwrap();
        let actual = model
            .generate_with_cuda_graph(phones, prompt, Some(&bert), 0, 1.0, 0.8, 1.0, 48, capacity)
            .unwrap();
        assert_eq!(actual.len(), 48);
        assert_eq!(actual, expected, "seed={seed}, capacity={capacity}");
    }
}

#[test]
fn conditioned_decoders_preserve_seeded_sampling_eos_and_token_budgets() {
    let model = model();
    for seed in [0, 42] {
        for max_tokens in [1, 8, 11, 16] {
            model.set_sampling_seed(Some(seed)).unwrap();
            let expected = generate(&model, "plain", &[1, 3, 2], max_tokens);
            assert_eq!(expected.len(), max_tokens.min(11));
            assert!(expected.iter().all(|&token| token < 6));
            for mode in ["kv", "static", "graph-fallback"] {
                model.set_sampling_seed(Some(seed)).unwrap();
                assert_eq!(
                    generate(&model, mode, &[1, 3, 2], max_tokens),
                    expected,
                    "{mode}, seed={seed}, max_tokens={max_tokens}"
                );
            }
            if seed == 42 && max_tokens == 16 {
                assert_eq!(expected, vec![0, 3, 1, 3, 5, 3, 5, 2, 5, 0, 3]);
            }
        }
    }
}

#[test]
fn legacy_generator_keeps_its_distinct_eos_behavior() {
    let model = model();
    model.set_sampling_seed(Some(42)).unwrap();
    assert!(model
        .generate(&[1, 2], 1, 1.0, 0.8, 1.0, 16)
        .unwrap()
        .is_empty());
}

#[test]
fn hybrid_mask_keeps_text_bidirectional_and_audio_causal() {
    let mask = model()
        .create_hybrid_mask(2, 5)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(
        mask,
        vec![
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    );
}

#[test]
fn bert_projection_removes_special_tokens_and_preserves_word2ph() {
    let model = model();
    let values: Vec<f32> = [99.0, 10.0, 20.0, 88.0]
        .into_iter()
        .flat_map(|value| std::iter::repeat_n(value, 1024))
        .collect();
    let raw = Tensor::from_vec(values, (1, 4, 1024), &Device::Cpu).unwrap();
    for input in [&raw, &raw.transpose(1, 2).unwrap()] {
        for (word2ph, expected) in [
            (vec![1, 2], vec![10.0, 20.0, 20.0]),
            (vec![0, 3], vec![20.0, 20.0, 20.0]),
            (vec![], vec![10.0, 10.0, 20.0]),
        ] {
            let actual = model.project_and_align_bert(input, &word2ph, 3).unwrap();
            let expected: Vec<f32> = expected
                .into_iter()
                .flat_map(|value| (0..16).map(move |i| value + i as f32 / 16.0))
                .collect();
            assert_eq!(actual.dims(), &[1, 3, 16]);
            assert_eq!(
                actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected
            );
        }
    }
}

#[test]
fn embedding_vocabularies_and_positional_ranges_remain_distinct() {
    let model = model();
    let ids = Tensor::new(&[1i64, 2], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let text = model.lookup_text_tokens(&ids, 2).unwrap();
    let audio = model.lookup_audio_tokens(&ids, 2).unwrap();
    assert_eq!(text.dims(), &[1, 2, 16]);
    assert_eq!(audio.dims(), &[1, 2, 16]);
    assert_eq!(
        text.flatten_all().unwrap().to_vec1::<f32>().unwrap()[1],
        0.125
    );
    assert_eq!(
        audio.flatten_all().unwrap().to_vec1::<f32>().unwrap()[1],
        8.125
    );
    let zero = Tensor::zeros((1, 1, 16), DType::F32, &Device::Cpu).unwrap();
    let text = model
        .add_sine_positional_pub(&zero, "text")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let audio = model
        .add_sine_positional_pub(&zero, "audio")
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(text[1], 1.25);
    assert_eq!(audio[1], 0.75);
    assert!(model.lookup_audio_token(7).is_err());
    assert!(model.add_sine_positional_at(&zero, 2048).is_err());
}

#[test]
fn sampling_penalizes_each_history_token_once_and_reports_raw_argmax() {
    let model = model();
    let logits = Tensor::new(&[8.0f32, 3.0, -5.0], &Device::Cpu).unwrap();
    assert_eq!(
        model
            .sample_token(&logits, 1, 1.0, 1.0, 2.0, &[0, 0], &[0])
            .unwrap(),
        (0, 0)
    );
    let logits = Tensor::new(&[-1.0f32, -1.5, -5.0], &Device::Cpu).unwrap();
    assert_eq!(
        model
            .sample_token(&logits, 1, 1.0, 1.0, 2.0, &[0], &[])
            .unwrap(),
        (1, 0)
    );
}

#[test]
fn nucleus_filter_runs_before_temperature_scaling() {
    let model = model();
    model.set_sampling_seed(Some(42)).unwrap();
    let logits = Tensor::new(&[2.0f32, 1.0, 0.0], &Device::Cpu).unwrap();
    for _ in 0..32 {
        assert_eq!(
            model
                .sample_token(&logits, 0, 0.65, 50.0, 1.0, &[], &[])
                .unwrap(),
            (0, 0)
        );
    }
}

#[test]
fn shared_prefill_preserves_separate_embeddings_positions_and_layout() {
    let model = model();
    let phones = [1usize, 3, 2];
    let prompt = [2usize, 4];
    let text_ids = Tensor::new(&[1i64, 3, 2], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let audio_ids = Tensor::new(&[2i64, 4], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    for bert in [0.0f32, 0.125] {
        let condition = Tensor::full(bert, (1, 3, 16), &Device::Cpu).unwrap();
        let text = model
            .embed_phonemes(&phones)
            .unwrap()
            .broadcast_add(&condition)
            .unwrap();
        let actual = model.prepare_prefill(&text, &prompt).unwrap();
        let expected_text = model
            .lookup_text_tokens(&text_ids, 3)
            .unwrap()
            .broadcast_add(&condition)
            .unwrap();
        let expected_text = model
            .add_sine_positional_pub(&expected_text, "text")
            .unwrap();
        let expected_audio = model.lookup_audio_tokens(&audio_ids, 2).unwrap();
        let expected_audio = model
            .add_sine_positional_pub(&expected_audio, "audio")
            .unwrap();
        let expected = Tensor::cat(&[&expected_text, &expected_audio], 1).unwrap();
        assert_eq!(actual.dims(), &[1, 5, 16]);
        assert_eq!(
            actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}

#[test]
fn conditioned_sampler_stops_on_raw_argmax_eos_after_consuming_the_sample() {
    let model = model();
    let logits = Tensor::new(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], &Device::Cpu).unwrap();
    let history = vec![1usize; 11];
    model.set_sampling_seed(Some(42)).unwrap();
    let (sampled, argmax) = model
        .sample_token(&logits, 0, 1.0, 50.0, 1.0, &[2], &history)
        .unwrap();
    assert!(sampled < 6);
    assert_eq!(argmax, 6);
    let next_random = model.sampling_rng.uniform().unwrap();
    model.set_sampling_seed(Some(42)).unwrap();
    let mut sampler = DecoderSampler::new(&model, 0, 1.0, 50.0, 1.0);
    assert_eq!(sampler.sample_next(&logits, &[2], &history).unwrap(), None);
    assert_eq!(model.sampling_rng.uniform().unwrap(), next_random);
    for len in 0..11 {
        assert!(sampler
            .sample_next(&logits, &[2], &history[..len])
            .unwrap()
            .is_some());
    }
}

#[test]
fn zero_generation_budget_returns_no_tokens_in_every_conditioned_decoder() {
    let model = model();
    for mode in ["plain", "kv", "static", "graph-fallback"] {
        assert!(generate(&model, mode, &[1, 3, 2], 0).is_empty(), "{mode}");
    }
}

#[test]
fn empty_text_returns_no_tokens_in_every_conditioned_decoder() {
    let model = model();
    for mode in ["plain", "kv", "static", "graph-fallback"] {
        assert!(generate(&model, mode, &[], 16).is_empty(), "{mode}");
    }
}

#[test]
fn invalid_static_kv_capacity_returns_an_error_instead_of_panicking() {
    let model = model();
    for (max_tokens, capacity) in [(16, 3), (usize::MAX, 64)] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.generate_with_static_kv(
                &[1, 2],
                &[2],
                None,
                15,
                0.95,
                0.8,
                1.35,
                max_tokens,
                capacity,
            )
        }));
        assert!(
            result.is_ok(),
            "invalid capacity must be reported as Result::Err"
        );
        assert!(matches!(result.unwrap(), Err(Error::ConfigError(_))));
    }
}
