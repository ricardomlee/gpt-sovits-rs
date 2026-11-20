//! Sampling order, request-local RNG and reusable scratch buffers.

use super::GPTModel;
use crate::Result;
use candle_core::{DType, Tensor};

pub(super) const MIN_TOKENS_BEFORE_EOS: usize = 11;

/// Sampling state and EOS policy shared by reference-conditioned decoders.
pub(super) struct DecoderSampler<'a> {
    model: &'a GPTModel,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    repetition_penalty: f32,
    scratch: SamplingScratch,
}

impl<'a> DecoderSampler<'a> {
    pub(super) fn new(
        model: &'a GPTModel,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
    ) -> Self {
        Self {
            model,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            scratch: SamplingScratch::new(model.ar_predict_layer.dims()[0]),
        }
    }

    pub(super) fn sample_next(
        &mut self,
        logits: &Tensor,
        prompt_tokens: &[usize],
        generated_tokens: &[usize],
    ) -> Result<Option<usize>> {
        let eos = self.model.ar_predict_layer.dims()[0] - 1;
        let mask_eos = generated_tokens.len() < MIN_TOKENS_BEFORE_EOS;
        let logits = if mask_eos {
            logits.narrow(0, 0, eos)?
        } else {
            logits.clone()
        };
        // Sampling precedes stopping to preserve request-local RNG replay.
        let (token, argmax) = self.model.sample_token_with_scratch(
            &logits,
            self.top_k,
            self.top_p,
            self.temperature,
            self.repetition_penalty,
            prompt_tokens,
            generated_tokens,
            &mut self.scratch,
        )?;
        if token >= eos || (!mask_eos && argmax == eos) {
            return Ok(None);
        }
        Ok(Some(token))
    }
}

struct SamplingScratch {
    sorted_logits: Vec<f32>,
    idx_probs: Vec<(usize, f32)>,
    seen_tokens: Vec<u32>,
    seen_stamp: u32,
}

impl SamplingScratch {
    fn new(vocab_size: usize) -> Self {
        Self {
            sorted_logits: Vec::with_capacity(vocab_size),
            idx_probs: Vec::with_capacity(vocab_size),
            seen_tokens: vec![0; vocab_size],
            seen_stamp: 1,
        }
    }

    fn reset_seen(&mut self, vocab_size: usize) {
        if self.seen_tokens.len() < vocab_size {
            self.seen_tokens.resize(vocab_size, 0);
        }
        self.seen_stamp = self.seen_stamp.wrapping_add(1);
        if self.seen_stamp == 0 {
            self.seen_tokens.fill(0);
            self.seen_stamp = 1;
        }
    }

    fn mark_seen(&mut self, token: usize) -> bool {
        if token >= self.seen_tokens.len() {
            return false;
        }
        if self.seen_tokens[token] == self.seen_stamp {
            false
        } else {
            self.seen_tokens[token] = self.seen_stamp;
            true
        }
    }
}

impl GPTModel {
    /// Set the semantic sampling seed for this model instance.
    pub fn set_sampling_seed(&self, seed: Option<u64>) -> Result<()> {
        self.sampling_rng.set_seed(seed)
    }

    /// Sample next token from logits using top-k and top-p filtering
    /// Sample a token from logits, matching Python's `logits_to_probs` order exactly:
    ///   1. Repetition penalty (in logit domain)
    ///   2. top_k filter: set non-top-k logits to -inf
    ///   3. top_p filter: using softmax(logits) WITHOUT temperature, set overflow to -inf
    ///   4. Temperature + softmax → final probs
    ///   5. Multinomial sample
    ///
    /// Returns `(sampled_token, argmax_token)` — both computed from ONE D2H transfer.
    /// `prompt_tokens` = prompt audio tokens (for rep penalty matching Python's y which includes prompt)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample_token(
        &self,
        logits: &Tensor,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        prompt_tokens: &[usize],
        generated_tokens: &[usize],
    ) -> Result<(usize, usize)> {
        let mut scratch = SamplingScratch::new(logits.dims().last().copied().unwrap_or(0));
        self.sample_token_with_scratch(
            logits,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            prompt_tokens,
            generated_tokens,
            &mut scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_token_with_scratch(
        &self,
        logits: &Tensor,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        prompt_tokens: &[usize],
        generated_tokens: &[usize],
        scratch: &mut SamplingScratch,
    ) -> Result<(usize, usize)> {
        let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
        let n = logits_vec.len();

        // Compute argmax of raw logits (before any filtering) — Python: torch.argmax(logits)
        let argmax = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // 1. Repetition penalty — Python uses set(y.flatten()) so each unique token is penalized once
        if repetition_penalty != 1.0 {
            scratch.reset_seen(n);
            for &token in prompt_tokens.iter().chain(generated_tokens.iter()) {
                if token < n && scratch.mark_seen(token) {
                    if logits_vec[token] > 0.0 {
                        logits_vec[token] /= repetition_penalty;
                    } else {
                        logits_vec[token] *= repetition_penalty;
                    }
                }
            }
        }

        // 2. top_k: find the k-th largest logit value, set all below to -inf
        if top_k > 0 && top_k < n {
            scratch.sorted_logits.clear();
            scratch.sorted_logits.extend_from_slice(&logits_vec);
            scratch
                .sorted_logits
                .select_nth_unstable_by(top_k - 1, |a, b| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
            let pivot = scratch.sorted_logits[top_k - 1];
            for v in logits_vec.iter_mut() {
                if *v < pivot {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // 3. top_p: cumulative prob filter on UNTEMPERATURE-SCALED softmax (matches Python)
        if top_p < 1.0 {
            let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits_vec.iter().map(|&x| (x - max_l).exp()).sum();
            if exp_sum.is_finite() && exp_sum > 0.0 {
                scratch.idx_probs.clear();
                scratch.idx_probs.extend(
                    logits_vec
                        .iter()
                        .enumerate()
                        .map(|(idx, &x)| (idx, (x - max_l).exp() / exp_sum)),
                );
                scratch
                    .idx_probs
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut cumsum = 0.0f32;
                for &(idx, prob) in &scratch.idx_probs {
                    if cumsum > top_p {
                        logits_vec[idx] = f32::NEG_INFINITY;
                    }
                    cumsum += prob;
                }
            }
        }

        // 4. Temperature + softmax → final probs
        let t = if temperature > 0.0 { temperature } else { 1.0 };
        let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits_vec.iter().map(|&x| ((x - max_l) / t).exp()).sum();
        if !exp_sum.is_finite() || exp_sum <= 0.0 {
            return Ok((argmax, argmax));
        }

        // 5. Multinomial sample
        let rand_val = self.sampling_rng.uniform()?;
        let mut cumsum = 0.0f32;
        let mut best = 0usize;
        let mut best_prob = f32::NEG_INFINITY;
        for (i, &logit) in logits_vec.iter().enumerate() {
            let prob = ((logit - max_l) / t).exp() / exp_sum;
            if prob > best_prob {
                best = i;
                best_prob = prob;
            }
            cumsum += prob;
            if rand_val <= cumsum {
                return Ok((i, argmax));
            }
        }

        // Fallback: argmax of final probs
        Ok((best, argmax))
    }
}
