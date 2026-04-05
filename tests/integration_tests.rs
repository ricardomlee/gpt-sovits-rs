//! Integration tests for GPT-SoVITS Rust

#[cfg(test)]
mod tests {
    use gpt_sovits_rs::*;

    #[test]
    fn test_config_builder() {
        let config = Config::builder()
            .with_device("cpu")
            .with_half_precision(false)
            .build();

        assert_eq!(config.device, config::Device::Cpu);
        assert!(!config.half_precision);
    }

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str("zh"), Some(Language::Chinese));
        assert_eq!(Language::from_str("en"), Some(Language::English));
        assert_eq!(Language::from_str("ja"), Some(Language::Japanese));
        assert_eq!(Language::from_str("ko"), Some(Language::Korean));
        assert_eq!(Language::from_str("yue"), Some(Language::Cantonese));
        assert_eq!(Language::from_str("auto"), Some(Language::Auto));
        assert_eq!(Language::from_str("invalid"), None);
    }

    #[test]
    fn test_inference_options_builder() {
        let options = InferenceOptions::builder()
            .top_k(20)
            .top_p(0.9)
            .temperature(0.7)
            .speed(1.2)
            .language(Language::Chinese)
            .max_tokens(300)
            .build();

        assert_eq!(options.top_k, 20);
        assert!((options.top_p - 0.9).abs() < 0.001);
        assert!((options.temperature - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_audio_buffer_operations() {
        // Create buffer
        let samples = vec![0.5f32; 48000];
        let mut buffer = AudioBuffer::new(samples, 24000, 1);

        // Test duration
        assert!((buffer.duration() - 2.0).abs() < 0.1);

        // Test normalize
        buffer.normalize();
        assert!(buffer.samples.iter().all(|&s| s.abs() <= 1.0));

        // Test fade in/out
        buffer.fade_in(50);
        buffer.fade_out(50);

        // Test resample
        let resampled = buffer.resample(48000);
        assert_eq!(resampled.sample_rate, 48000);
        assert!(resampled.len() >= buffer.len());
    }

    #[test]
    fn test_audio_buffer_concat() {
        let buffer1 = AudioBuffer::new(vec![0.5f32; 24000], 24000, 1);
        let mut buffer2 = AudioBuffer::new(vec![0.5f32; 24000], 24000, 1);

        buffer2.concat(&buffer1).unwrap();
        assert_eq!(buffer2.len(), 48000);
    }

    #[test]
    fn test_pipeline_creation() {
        let config = Config::default();
        let pipeline = Pipeline::new(config);

        assert!(pipeline.is_ok());
        let pipeline = pipeline.unwrap();
        assert!(!pipeline.is_ready()); // No models loaded
    }
}

/// Test module for text frontend
#[cfg(test)]
mod text_frontend_tests {
    use gpt_sovits_rs::text_frontend::*;
    use gpt_sovits_rs::Language;

    #[test]
    fn test_text_normalizer() {
        let normalizer = TextNormalizer::new();
        let result = normalizer.normalize("hello    world").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_language_detector() {
        let detector = LanguageDetector::new().unwrap();

        let lang = detector.detect("你好世界");
        assert_eq!(lang.unwrap(), Language::Chinese);

        let lang = detector.detect("Hello World");
        assert_eq!(lang.unwrap(), Language::English);

        let lang = detector.detect("こんにちは");
        assert_eq!(lang.unwrap(), Language::Japanese);

        let lang = detector.detect("안녕하세요");
        assert_eq!(lang.unwrap(), Language::Korean);
    }

    #[test]
    fn test_symbol_table() {
        let table = SymbolTable::new();

        assert!(table.len() > 0);
        assert_eq!(table.pad_id(), 0);
        assert_eq!(table.bos_id(), 1);
        assert_eq!(table.eos_id(), 2);

        // Test encode/decode round trip
        let ids = table.encode("a o e").unwrap();
        assert!(ids.len() > 0);
        assert_eq!(ids[0], table.bos_id());
        assert_eq!(ids[ids.len() - 1], table.eos_id());
    }

    #[test]
    fn test_g2p_converter() {
        let converter = G2PConverter::new().expect("Failed to create G2PConverter");

        let result = converter.convert("测试", Language::Chinese);
        assert!(result.is_ok());

        let result = converter.convert("test", Language::English);
        assert!(result.is_ok());
    }
}

/// Test module for model weights
#[cfg(test)]
mod weights_tests {
    use gpt_sovits_rs::utils::weights::*;
    use candle_core::{Tensor, DType, Device};

    #[test]
    fn test_state_dict() {
        let device = Device::Cpu;
        let mut data = std::collections::HashMap::new();
        data.insert("layer.weight".to_string(), Tensor::ones((10, 5), DType::F32, &device).unwrap());
        data.insert("layer.bias".to_string(), Tensor::zeros(5, DType::F32, &device).unwrap());

        let sd = StateDict::new(data);
        assert!(sd.contains("layer.weight"));
        assert!(sd.contains("layer.bias"));
        assert!(!sd.contains("nonexistent"));
        assert_eq!(sd.len(), 2);
    }

    #[test]
    fn test_embedding() {
        let device = Device::Cpu;
        let weight = Tensor::ones((100, 32), DType::F32, &device).unwrap();
        let embedding = Embedding::new(weight);

        assert_eq!(embedding.num_embeddings(), 100);
        assert_eq!(embedding.embedding_dim(), 32);
    }

    #[test]
    fn test_linear() {
        let device = Device::Cpu;
        let weight = Tensor::ones((64, 32), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(64, DType::F32, &device).unwrap();
        let linear = Linear::new(weight, Some(bias));

        assert_eq!(linear.in_features(), 32);
        assert_eq!(linear.out_features(), 64);
    }

    #[test]
    fn test_layer_norm() {
        let device = Device::Cpu;
        let weight = Tensor::ones(32, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(32, DType::F32, &device).unwrap();
        let layer_norm = LayerNorm::new(weight, bias);

        let input = Tensor::randn(0.0, 1.0, (2, 10, 32), &device).unwrap();
        let output = layer_norm.forward(&input);
        if let Err(e) = &output {
            eprintln!("LayerNorm error: {:?}", e);
        }
        assert!(output.is_ok());
    }
}

/// Test module for transformer model
#[cfg(test)]
mod transformer_tests {
    use gpt_sovits_rs::models::transformer::*;

    #[test]
    fn test_causal_mask() {
        let mask = Transformer::create_causal_mask(4, &candle_core::Device::Cpu).unwrap();

        // Check mask shape
        assert_eq!(mask.dims(), &[4, 4]);

        // Check causal property using direct 2D indexing
        let mask_2d: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        assert_eq!(mask_2d[0][0], 1.0);  // (0,0) = 1
        assert_eq!(mask_2d[0][1], 0.0);  // (0,1) = 0
        assert_eq!(mask_2d[1][0], 1.0);  // (1,0) = 1
        assert_eq!(mask_2d[1][1], 1.0);  // (1,1) = 1
    }

    #[test]
    fn test_transformer_config() {
        let config = TransformerConfig::default();

        assert_eq!(config.vocab_size, 512);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_hidden_layers, 8);
        assert_eq!(config.num_attention_heads, 8);
    }
}
