//! Public numerical-inspection helpers kept outside the decoding implementations.

use super::GPTModel;
use crate::Result;
use candle_core::Tensor;

impl GPTModel {
    // ========================
    // Public helper methods for debugging
    // ========================

    /// Lookup text token embeddings for a tensor of token IDs.
    /// Public wrapper around the private lookup_tokens for text embeddings.
    pub fn lookup_text_tokens(&self, ids: &Tensor, seq: usize) -> Result<Tensor> {
        self.lookup_tokens(&self.text_embedding, ids, seq)
    }

    /// Lookup audio token embeddings for a tensor of token IDs.
    /// Public wrapper around the private lookup_tokens for audio embeddings.
    pub fn lookup_audio_tokens(&self, ids: &Tensor, seq: usize) -> Result<Tensor> {
        self.lookup_tokens(&self.audio_embedding, ids, seq)
    }

    /// Add sinusoidal positional encoding with learned alpha scaling.
    /// Public wrapper around the private add_sine_positional.
    pub fn add_sine_positional_pub(&self, x: &Tensor, kind: &str) -> Result<Tensor> {
        self.add_sine_positional(x, kind)
    }

    /// Get a reference to the BERT projection (weight, bias) if available.
    pub fn bert_proj_ref(&self) -> Option<&(Tensor, Tensor)> {
        self.bert_proj.as_ref()
    }

    /// Compute the output of the first transformer layer for debugging.
    ///
    /// This method reproduces the exact same computation as the beginning of
    /// `generate_with_prompts` up through the first transformer layer, allowing
    /// comparison with Python reference implementations.
    ///
    /// # Arguments
    /// * `phoneme_ids` - Input phoneme sequence
    /// * `bert_features` - Optional BERT features [batch, 1024, seq_len] or [batch, seq_len, 1024]
    /// * `prompt_tokens` - Prompt audio token IDs (from reference audio)
    ///
    /// # Returns
    /// Flattened hidden state after the first transformer layer as `Vec<f32>`
    pub fn debug_layer0_output(
        &self,
        phoneme_ids: &[usize],
        bert_features: Option<&Tensor>,
        prompt_tokens: &[usize],
    ) -> Result<Vec<f32>> {
        if phoneme_ids.is_empty() {
            return Err(crate::Error::InferenceError(
                "phoneme_ids cannot be empty".to_string(),
            ));
        }

        let text_seq = phoneme_ids.len();
        let prompt_seq = prompt_tokens.len();
        let x_emb = self.embed_phonemes(phoneme_ids)?;

        // Step 2: Project and add BERT features
        let x_emb = if let (Some(bert), Some((proj_w, proj_b))) = (bert_features, &self.bert_proj) {
            let bert_dims = bert.dims();
            let bert_reshaped = if bert_dims.len() == 3 && bert_dims[1] == 1024 {
                bert.transpose(1, 2)? // [batch, 1024, seq] -> [batch, seq, 1024]
            } else {
                bert.clone()
            };
            if bert_reshaped.dims().last().copied() == Some(1024) {
                let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
                let projected = bert_reshaped.matmul(&proj_w_3d)?;
                let projected =
                    projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
                // Align BERT features: strip CLS/SEP, expand via word2ph to phoneme seq length
                let text_seq = x_emb.dims()[1];
                let aligned = self.align_bert_to_phonemes(&projected, text_seq, &[])?;
                // Python: x = x + self.bert_proj(...) — no scaling
                x_emb.broadcast_add(&aligned)?
            } else {
                x_emb
            }
        } else {
            x_emb
        };

        let xy_pos = self.prepare_prefill(&x_emb, prompt_tokens)?;

        // Step 7: Create hybrid attention mask
        let total_seq = text_seq + prompt_seq;
        let mask = self.create_hybrid_mask(text_seq, total_seq)?;

        // Step 8: Run through the first transformer layer only
        let hidden = self.transformer.forward_first_layer(&xy_pos, &mask)?;

        // Flatten and return as Vec<f32>
        let flat = hidden.flatten_all()?;
        flat.to_vec1()
            .map_err(|e| crate::Error::InferenceError(e.to_string()))
    }

    /// Get a reference to the ar_predict_layer for debugging
    pub fn ar_predict_layer_ref(&self) -> Result<&Tensor> {
        Ok(&self.ar_predict_layer)
    }

    /// Run the full transformer (all 24 layers) on the input embeddings
    /// and return the hidden state. This is used for debugging layer-by-layer
    /// numerical accuracy. Uses causal mask internally.
    pub fn run_full_transformer(&self, xy_pos: &Tensor) -> Result<Tensor> {
        self.transformer.forward_all_layers(xy_pos)
    }

    /// Run all transformer layers with a provided attention mask.
    pub fn run_transformer_with_mask(&self, xy_pos: &Tensor, mask: &Tensor) -> Result<Tensor> {
        self.transformer.forward_all_layers_with_mask(xy_pos, mask)
    }

    /// Run first transformer layer and return debug intermediates:
    /// (attn_out, norm1_out, linear1_out, relu_out, final_out)
    pub fn debug_layer0_intermediates(
        &self,
        xy_pos: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.transformer.forward_first_layer_debug(xy_pos, mask)
    }
}
