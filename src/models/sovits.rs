//! SoVITS Model for audio synthesis

use candle_core::{Device, Tensor, DType};
use crate::{Result, Error};
use crate::utils::{StateDict, load_safetensors, Linear, LayerNorm};
use crate::utils::weights::Conv1d;

/// SoVITS Model for mel spectrogram generation
pub struct SoVITSModel {
    device: Device,
    dtype: DType,
    // Model components
    text_encoder: TextEncoder,
    duration_predictor: DurationPredictor,
    flow_decoder: FlowDecoder,
    speaker_embedding: SpeakerEmbedding,
    n_mels: usize,
    sampling_rate: u32,
}

/// Text encoder for phoneme features
pub struct TextEncoder {
    embedding: Tensor,
    #[allow(dead_code)]
    layers: Vec<EncoderLayer>,
    #[allow(dead_code)]
    hidden_size: usize,
}

struct EncoderLayer {
    conv1: Conv1d,
    conv2: Conv1d,
    norm: LayerNorm,
}

/// Duration predictor for timing
pub struct DurationPredictor {
    projection: Linear,
    device: Device,
}

/// Flow decoder for mel synthesis
pub struct FlowDecoder {
    #[allow(dead_code)]
    layers: Vec<FlowLayer>,
    output_projection: Linear,
}

struct FlowLayer {
    #[allow(dead_code)]
    coupling: CouplingLayer,
}

struct CouplingLayer {
    #[allow(dead_code)]
    net: Vec<Linear>,
}

/// Speaker embedding lookup
pub struct SpeakerEmbedding {
    embedding: Tensor,
}

impl SoVITSModel {
    /// Load model from safetensors file
    pub fn load(path: &str) -> Result<Self> {
        Self::load_with_device(path, &Device::Cpu)
    }

    /// Load model with specific device
    pub fn load_with_device(path: &str, device: &Device) -> Result<Self> {
        // Load weights from safetensors
        let weights_map = load_safetensors(path)?;
        let state_dict = StateDict::new(weights_map);

        // Infer configuration from weights
        let n_mels = 100; // Default for GPT-SoVITS
        let sampling_rate = 24000;

        // Create text encoder
        let text_encoder = TextEncoder::new(&state_dict, device)?;

        // Create duration predictor
        let duration_predictor = DurationPredictor::new(&state_dict, device)?;

        // Create flow decoder
        let flow_decoder = FlowDecoder::new(&state_dict, device)?;

        // Create speaker embedding
        let speaker_embedding = SpeakerEmbedding::new(&state_dict, device)?;

        Ok(Self {
            device: device.clone(),
            dtype: DType::F32,
            text_encoder,
            duration_predictor,
            flow_decoder,
            speaker_embedding,
            n_mels,
            sampling_rate,
        })
    }

    /// Synthesize mel spectrogram from semantic tokens
    ///
    /// # Arguments
    /// * `semantic_tokens` - Input semantic token sequence
    ///
    /// # Returns
    /// Mel spectrogram tensor [1, n_mels, time]
    pub fn synthesize(&self, semantic_tokens: &[usize]) -> Result<Tensor> {
        if semantic_tokens.is_empty() {
            return Err(Error::InferenceError("Empty semantic tokens".to_string()));
        }

        // Convert tokens to tensor [1, seq_len]
        let tokens: Vec<i64> = semantic_tokens.iter().map(|&x| x as i64).collect();
        let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Step 1: Encode text through text encoder
        let text_features = self.text_encoder.encode(&tokens_tensor)?;

        // Step 2: Predict durations
        let durations = self.duration_predictor.predict(&text_features)?;

        // Step 3: Expand features according to durations
        let expanded_features = self.expand_by_duration(&text_features, &durations)?;

        // Step 4: Get speaker embedding (using dummy speaker ID for now)
        let speaker_emb = self.speaker_embedding.get(0)?;

        // Step 5: Decode through flow
        let mel_spec = self.flow_decoder.decode(&expanded_features, &speaker_emb)?;

        Ok(mel_spec)
    }

    /// Expand encoded features by predicted durations
    fn expand_by_duration(&self, features: &Tensor, durations: &Tensor) -> Result<Tensor> {
        // Simple expansion: repeat each frame by its duration
        let dur_vec: Vec<i64> = durations.to_vec1()?;
        let mut expanded_frames = Vec::new();

        let feat_vec: Vec<Vec<f32>> = features.to_vec2()?;
        for (frame, &dur) in feat_vec.iter().zip(dur_vec.iter()) {
            let dur = dur.max(1) as usize; // At least 1 frame per token
            for _ in 0..dur {
                expanded_frames.push(frame.clone());
            }
        }

        // Create tensor from expanded frames [total_frames, hidden_dim]
        let flat: Vec<f32> = expanded_frames.into_iter().flatten().collect();
        let hidden_dim = feat_vec[0].len();
        let total_frames = flat.len() / hidden_dim;

        Ok(Tensor::from_vec(flat, (total_frames, hidden_dim), &self.device)?.unsqueeze(0)?)
    }

    /// Get model device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get model dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get number of mel bins
    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Get sampling rate
    pub fn sampling_rate(&self) -> u32 {
        self.sampling_rate
    }
}

impl TextEncoder {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        // Try to load embedding from GPT-SoVITS checkpoint
        // Key: enc_p.text_embedding.weight [vocab_size=322, hidden=192]
        let embedding = state_dict.get("enc_p.text_embedding.weight")
            .ok()
            .cloned()
            .unwrap_or_else(|| {
                // Create default embedding for 512 vocab, 256 dim
                Tensor::zeros((512, 256), DType::F32, device).unwrap()
            });

        let hidden_size = embedding.dims()[1];

        // Create encoder layers (simplified - just one layer for now)
        let layers = Vec::new();

        // Try to load conv layers from enc_p.encoder_text
        if state_dict.contains("enc_p.encoder_text.attn_layers.0.conv_q.weight") {
            // Note: The actual GPT-SoVITS model uses a different architecture
            // For now, we'll skip loading these complex layers
            // and use a simplified encode path
        }

        Ok(Self {
            embedding,
            layers,
            #[allow(dead_code)]
            hidden_size,
        })
    }

    pub fn encode(&self, input: &Tensor) -> Result<Tensor> {
        // Get embeddings: [1, seq_len] -> [1, seq_len, hidden_dim]
        // Use our custom embedding lookup that handles 2D inputs
        let mut x = self.embedding_lookup(input)?;

        // Apply encoder layers if available
        for layer in &self.layers {
            // Transpose for conv: [1, seq_len, hidden] -> [1, hidden, seq_len]
            let x_t = x.transpose(1, 2)?;
            let conv_out = layer.conv1.forward(&x_t)?;
            let conv_out = layer.conv2.forward(&conv_out)?;
            // Transpose back: [1, hidden, seq_len] -> [1, seq_len, hidden]
            x = conv_out.transpose(1, 2)?;

            // Layer norm
            x = layer.norm.forward(&x)?;
        }

        Ok(x)
    }

    /// Custom embedding lookup for 2D input
    fn embedding_lookup(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::UnexpectedShape {
                msg: "Expected 2D input for embedding".to_string(),
                expected: candle_core::Shape::from(&[1usize, 1]),
                got: candle_core::Shape::from(dims),
            }.into());
        }

        let (batch, seq_len) = (dims[0], dims[1]);

        // Flatten to 1D for processing
        let indices_flat: Vec<i64> = input.flatten_all()?.to_vec1()?;

        // Lookup each index and stack
        let mut embeddings = Vec::with_capacity(indices_flat.len());
        for &idx in &indices_flat {
            let emb = self.embedding.get(idx as usize)?;
            embeddings.push(emb);
        }

        // Stack: [batch*seq_len, hidden]
        let stacked = Tensor::stack(&embeddings, 0)?;

        // Reshape to [batch, seq_len, hidden]
        stacked.reshape((batch, seq_len, self.embedding.dims()[1]))
            .map_err(|e| e.into())
    }
}

impl DurationPredictor {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let projection = state_dict.get_linear("duration_predictor.projection")
            .ok()
            .unwrap_or_else(|| {
                // Create default projection
                Linear::new(
                    Tensor::zeros((1, 256), DType::F32, device).unwrap(),
                    Some(Tensor::zeros(1, DType::F32, device).unwrap()),
                )
            });

        Ok(Self { projection, device: device.clone() })
    }

    pub fn predict(&self, features: &Tensor) -> Result<Tensor> {
        // Simple projection to get duration predictions
        // features: [1, seq_len, hidden] -> durations: [seq_len]
        let _seq_len = features.dims()[1];

        // Flatten and project
        let flat = features.flatten_from(0)?;
        let output = self.projection.forward(&flat)?;

        // Convert to durations (abs + round)
        let output_vec: Vec<f32> = output.to_vec1()?;
        let durations: Vec<i64> = output_vec.iter()
            .map(|&x| x.abs() as i64 + 1) // At least 1 frame per token
            .collect();

        Tensor::new(durations.as_slice(), &self.device).map_err(|e| e.into())
    }
}

impl FlowDecoder {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let layers = Vec::new();

        let output_projection = state_dict.get_linear("flow_decoder.output")
            .ok()
            .unwrap_or_else(|| {
                Linear::new(
                    Tensor::zeros((100, 256), DType::F32, device).unwrap(),
                    Some(Tensor::zeros(100, DType::F32, device).unwrap()),
                )
            });

        Ok(Self { layers, output_projection })
    }

    pub fn decode(&self, features: &Tensor, _speaker_emb: &Tensor) -> Result<Tensor> {
        // Simple projection to mel space
        // features: [1, total_frames, hidden] -> mel: [1, n_mels, total_frames]

        let batch_size = features.dims()[0];
        let seq_len = features.dims()[1];

        // Flatten: [1, total_frames, hidden] -> [total_frames, hidden]
        let flat = features.flatten_from(0)?;

        // Project to mel: [total_frames, n_mels]
        let mel_flat = self.output_projection.forward(&flat)?;

        // Reshape and transpose: [1, n_mels, total_frames]
        Ok(mel_flat.reshape((batch_size, 100, seq_len))?)
    }
}

impl SpeakerEmbedding {
    pub fn new(state_dict: &StateDict, device: &Device) -> Result<Self> {
        let embedding = state_dict.get("speaker_embedding.weight")
            .ok()
            .cloned()
            .unwrap_or_else(|| {
                // Default: 4 speakers, 256 dim
                Tensor::zeros((4, 256), DType::F32, device).unwrap()
            });

        Ok(Self { embedding })
    }

    pub fn get(&self, speaker_id: usize) -> Result<Tensor> {
        let ids = Tensor::new(&[speaker_id as i64], &self.embedding.device())?;
        self.embedding.index_select(&ids, 0)
            .map_err(|e| e.into())
    }
}

impl crate::models::Model for SoVITSModel {
    fn load(path: &str) -> Result<Self> {
        Self::load(path)
    }

    fn device(&self) -> &str {
        match self.device {
            Device::Cpu => "cpu",
            Device::Cuda(_) => "cuda",
            Device::Metal(_) => "mps",
        }
    }

    fn to_device(&mut self, device: &str) -> Result<()> {
        let new_device = match device {
            "cuda" => Device::new_cuda(0),
            "mps" => Device::new_metal(0),
            _ => Ok(Device::Cpu),
        }
        .map_err(|e| Error::ModelLoadError(e.to_string()))?;

        self.device = new_device;
        Ok(())
    }
}
