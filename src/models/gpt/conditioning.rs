//! Text/audio embeddings, BERT alignment, positional offsets and attention masks.

use super::GPTModel;
use crate::{Error, Result};
use candle_core::Tensor;

impl GPTModel {
    /// Project and align raw BERT features to phone level.
    ///
    /// Takes raw BERT output [1, seq+2, 1024] (includes CLS/SEP), applies bert_proj,
    /// strips CLS/SEP, and expands to phone level via word2ph.
    /// Returns [1, n_phones, 512].
    ///
    /// Used by the inference pipeline to pre-align ref and target BERT features separately
    /// before concatenating them for joint GPT conditioning.
    pub fn project_and_align_bert(
        &self,
        bert_raw: &Tensor, // [1, seq_with_cls_sep, 1024]
        word2ph: &[usize],
        n_phones: usize,
    ) -> Result<Tensor> {
        let (proj_w, proj_b) = self
            .bert_proj
            .as_ref()
            .ok_or_else(|| Error::ModelLoadError("bert_proj not loaded".to_string()))?;

        let bert_raw = bert_raw.to_dtype(proj_w.dtype())?;
        let bert = if bert_raw.dims().len() == 3 && bert_raw.dims()[1] == 1024 {
            bert_raw.transpose(1, 2)?
        } else {
            bert_raw
        };
        let proj_w_3d = proj_w.t()?.unsqueeze(0)?;
        let projected = bert.matmul(&proj_w_3d)?;
        let projected = projected.broadcast_add(&proj_b.reshape((1, 1, proj_b.dims()[0]))?)?;
        self.align_bert_to_phonemes(&projected, n_phones, word2ph)
    }

    /// Align BERT features (including CLS/SEP) to phoneme sequence.
    ///
    /// Python's pipeline:
    ///   1. Extract BERT hidden states (char-level, includes CLS/SEP tokens)
    ///   2. Remove CLS (first) and SEP (last) tokens → bert_content[i] per char
    ///   3. Expand: repeat bert_content[i] word2ph[i] times
    ///
    /// When word2ph is provided: uses exact word2ph expansion (matches Python exactly).
    /// When word2ph is empty: falls back to nearest-neighbor interpolation.
    ///
    /// # Arguments
    /// * `projected` - BERT projected features [1, bert_seq, hidden] (including CLS/SEP)
    /// * `target_seq` - Target phoneme sequence length
    /// * `word2ph` - Per-character phoneme counts from G2P (empty → use nearest-neighbor)
    pub(super) fn align_bert_to_phonemes(
        &self,
        projected: &Tensor,
        target_seq: usize,
        word2ph: &[usize],
    ) -> Result<Tensor> {
        let bert_full_len = projected.dims()[1];

        // Strip CLS (index 0) and SEP (last index)
        let bert_content = if bert_full_len >= 2 {
            projected.narrow(1, 1, bert_full_len - 2)?
        } else {
            projected.clone()
        };
        let bert_len = bert_content.dims()[1];

        if bert_len == 0 {
            return Ok(projected.clone());
        }

        // Use word2ph expansion if provided and consistent with bert_content
        if !word2ph.is_empty() && word2ph.len() == bert_len {
            let total_phonemes: usize = word2ph.iter().sum();
            if total_phonemes == target_seq {
                let mut frames = Vec::with_capacity(target_seq);
                for (i, &count) in word2ph.iter().enumerate() {
                    for _ in 0..count {
                        frames.push(bert_content.narrow(1, i, 1)?);
                    }
                }
                return Tensor::cat(&frames, 1).map_err(|e| e.into());
            } else {
                tracing::debug!(
                    "word2ph sum={} != target_seq={}, falling back to nearest-neighbor",
                    total_phonemes,
                    target_seq
                );
            }
        }

        // Fallback: nearest-neighbor interpolation from bert_len to target_seq
        if bert_len == target_seq {
            Ok(bert_content)
        } else if bert_len > target_seq {
            bert_content.narrow(1, 0, target_seq).map_err(|e| e.into())
        } else {
            let mut frames = Vec::with_capacity(target_seq);
            for i in 0..target_seq {
                let bert_idx = (i * bert_len / target_seq).min(bert_len - 1);
                frames.push(bert_content.narrow(1, bert_idx, 1)?);
            }
            Tensor::cat(&frames, 1).map_err(|e| e.into())
        }
    }

    /// Create hybrid attention mask matching Python's infer_panel:
    /// Hybrid attention mask matching Python's infer_panel:
    /// - Text→Text: bidirectional (no masking)
    /// - Text→Audio: BLOCKED (True in Python x_attn_mask_pad)
    /// - Audio→Text: allowed (no masking)
    /// - Audio→Audio: causal (block future j > i)
    pub(super) fn create_hybrid_mask(&self, text_seq: usize, total_seq: usize) -> Result<Tensor> {
        let mut mask = vec![0.0f32; total_seq * total_seq]; // 0 = allow, 1 = block

        for i in 0..text_seq {
            // Text positions: block attending to audio positions
            for j in text_seq..total_seq {
                mask[i * total_seq + j] = 1.0;
            }
        }
        for i in text_seq..total_seq {
            // Audio positions: block future audio
            for j in (i + 1)..total_seq {
                mask[i * total_seq + j] = 1.0;
            }
        }

        Ok(Tensor::from_vec(
            mask,
            (total_seq, total_seq),
            &self.device,
        )?)
    }

    /// Lookup token embeddings for a tensor of token IDs.
    /// Stays entirely on GPU via Tensor::embedding (no D2H transfer).
    /// ids: [batch, seq] (i64) → output: [batch, seq, hidden]
    pub(super) fn lookup_tokens(
        &self,
        embedding: &Tensor,
        ids: &Tensor,
        _seq: usize,
    ) -> Result<Tensor> {
        let orig_dims = ids.dims().to_vec();
        let hidden = embedding.dims()[1];
        // index_select (used internally by Tensor::embedding) requires 1D ids
        let ids_flat = ids
            .flatten_all()
            .map_err(|e| crate::Error::InferenceError(e.to_string()))?
            .to_dtype(candle_core::DType::U32)
            .map_err(|e| crate::Error::InferenceError(e.to_string()))?;
        let flat_emb = embedding
            .embedding(&ids_flat)
            .map_err(|e| crate::Error::InferenceError(e.to_string()))?;
        // Restore original batch/seq dims: [batch*seq, hidden] → [batch, seq, hidden]
        let mut out_dims = orig_dims;
        out_dims.push(hidden);
        flat_emb
            .reshape(out_dims)
            .map_err(|e| crate::Error::InferenceError(e.to_string()))
    }

    pub(super) fn lookup_audio_token(&self, token: usize) -> Result<Tensor> {
        if token >= self.audio_embedding.dim(0)? {
            return Err(Error::InferenceError(format!(
                "audio token {token} exceeds embedding vocabulary {}",
                self.audio_embedding.dim(0)?
            )));
        }
        self.audio_embedding
            .narrow(0, token, 1)?
            .unsqueeze(0)
            .map_err(Into::into)
    }

    /// Add sinusoidal positional encoding with learned alpha scaling
    pub(super) fn add_sine_positional(&self, x: &Tensor, kind: &str) -> Result<Tensor> {
        self.add_sine_positional_with_alpha(x, 0, kind)
    }

    /// Add sinusoidal positional encoding starting at a specific position offset
    pub(super) fn add_sine_positional_at(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        // New tokens during generation are audio tokens
        self.add_sine_positional_with_alpha(x, start_pos, "audio")
    }

    /// Add sinusoidal positional encoding with explicit alpha
    pub(super) fn add_sine_positional_with_alpha(
        &self,
        x: &Tensor,
        start_pos: usize,
        kind: &str,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let seq = dims[1];
        let positions = match kind {
            "text" => &self.text_positional,
            _ => &self.audio_positional,
        };
        if start_pos + seq > positions.dim(1)? {
            return Err(Error::InferenceError(format!(
                "position range {}..{} exceeds model limit {}",
                start_pos,
                start_pos + seq,
                positions.dim(1)?
            )));
        }

        let positions = positions.narrow(1, start_pos, seq)?;
        x.broadcast_add(&positions).map_err(Into::into)
    }
}
