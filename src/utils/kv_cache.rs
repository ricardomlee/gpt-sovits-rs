//! KV Cache for Transformer Attention
//!
//! # KV Cache原理
//!
//! 在 Transformer 的自注意力机制中，对于每个 token，我们计算：
//! - Q (Query): 当前 token 的查询向量
//! - K (Key): 用于与其他 token 匹配的键向量
//! - V (Value): 实际传递的信息向量
//!
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! ## 问题：为什么需要 KV Cache？
//!
//! 在自回归生成中，每次生成新 token 时：
//! - 传统方法：重新计算所有 token 的 Q, K, V → O(n²) 复杂度
//! - KV Cache：只计算新 token 的 Q, K, V，复用之前的 K, V → O(n) 复杂度
//!
//! ## 示例
//!
//! 生成序列 "你好世界"：
//!
//! 步骤 1: 输入"你"
//!   - 计算 Q₁, K₁, V₁
//!   - 输出 P("好" | "你")
//!   - Cache: {K₁, V₁}
//!
//! 步骤 2: 输入"好"
//!   - 计算 Q₂, K₂, V₂ (只计算新 token!)
//!   - 拼接 Cache: K = [K₁, K₂], V = [V₁, V₂]
//!   - 输出 P("世" | "你好")
//!   - Cache: {K₁, V₁, K₂, V₂}
//!
//! 步骤 3: 输入"世"
//!   - 计算 Q₃, K₃, V₃
//!   - 拼接 Cache: K = [K₁, K₂, K₃], V = [V₁, V₂, V₃]
//!   - 输出 P("界" | "你好世")
//!   - Cache: {K₁, V₁, K₂, V₂, K₃, V₃}
//!
//! ## 性能提升
//!
//! 对于生成 500 个 token:
//! - 无 Cache: 500×501/2 = 125,250 次 KV 计算
//! - 有 Cache: 500 次 KV 计算
//! - 理论加速：250x (实际 5-10x，因为有内存开销)

use crate::Result;
use candle_core::{DType, Tensor};

/// KV Cache for a single transformer layer
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cached keys: [batch, num_heads, seq_len, head_dim]
    k_cache: Option<Tensor>,
    /// Cached values: [batch, num_heads, seq_len, head_dim]
    v_cache: Option<Tensor>,
    /// Number of tokens in cache
    len: usize,
}

impl KvCache {
    /// Create a new empty KV cache
    pub fn new() -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            len: 0,
        }
    }

    /// Get the current length of the cache
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Update cache with new K and V tensors
    ///
    /// # Arguments
    /// * `k` - New key tensor [batch, num_heads, 1, head_dim]
    /// * `v` - New value tensor [batch, num_heads, 1, head_dim]
    ///
    /// # Returns
    /// Updated full K and V tensors including cached and new values
    pub fn update(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let (k_out, v_out) = match &self.k_cache {
            None => {
                // First token, no cache
                self.k_cache = Some(k.clone());
                self.v_cache = Some(v.clone());
                self.len = 1;
                (k, v)
            }
            Some(prev_k) => {
                // Concatenate with existing cache
                let prev_k = prev_k.clone();
                let prev_v = self.v_cache.as_ref().unwrap().clone();

                // Concatenate along seq_len dimension (dim 2)
                let k_out = Tensor::cat(&[prev_k, k.clone()], 2)?;
                let v_out = Tensor::cat(&[prev_v, v.clone()], 2)?;

                self.k_cache = Some(k_out.clone());
                self.v_cache = Some(v_out.clone());
                self.len += 1;

                (k_out, v_out)
            }
        };

        Ok((k_out, v_out))
    }

    /// Consume the cache, returning the raw K and V tensors.
    pub fn into_tensors(self) -> (Option<Tensor>, Option<Tensor>) {
        (self.k_cache, self.v_cache)
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
        self.len = 0;
    }

    /// Get cached K tensor
    pub fn k(&self) -> Option<&Tensor> {
        self.k_cache.as_ref()
    }

    /// Get cached V tensor
    pub fn v(&self) -> Option<&Tensor> {
        self.v_cache.as_ref()
    }
}

/// KV Cache manager for all transformer layers
#[derive(Debug, Clone)]
pub struct KvCacheManager {
    /// One KV cache per transformer layer
    caches: Vec<KvCache>,
}

impl KvCacheManager {
    /// Create a new KV cache manager for `num_layers` transformer layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            caches: (0..num_layers).map(|_| KvCache::new()).collect(),
        }
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Get cache for a specific layer
    pub fn cache(&self, layer_idx: usize) -> Option<&KvCache> {
        self.caches.get(layer_idx)
    }

    /// Get mutable cache for a specific layer
    pub fn cache_mut(&mut self, layer_idx: usize) -> Option<&mut KvCache> {
        self.caches.get_mut(layer_idx)
    }

    /// Get or create cache for a layer
    pub fn get_or_create(&mut self, layer_idx: usize) -> &mut KvCache {
        if layer_idx >= self.caches.len() {
            // Extend the vector if needed
            while self.caches.len() <= layer_idx {
                self.caches.push(KvCache::new());
            }
        }
        &mut self.caches[layer_idx]
    }

    /// Reset all caches
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Get the current sequence length (assumes all layers have same length)
    pub fn len(&self) -> usize {
        self.caches.first().map(|c| c.len()).unwrap_or(0)
    }
}

impl Default for KvCacheManager {
    fn default() -> Self {
        Self::new(0)
    }
}

impl KvCacheManager {
    /// Consume the manager and return the raw per-layer caches.
    pub fn into_caches(self) -> Vec<KvCache> {
        self.caches
    }
}

// ---------------------------------------------------------------------------
// Static (pre-allocated) KV cache for fixed-shape decode
// ---------------------------------------------------------------------------

/// Pre-allocated KV cache for a single transformer layer.
/// Eliminates per-step tensor allocations during autoregressive decode.
/// K/V are stored in fixed-size buffers; new tokens are written via `scatter_set`.
pub struct StaticKvLayer {
    /// [1, n_heads, max_len, head_dim]
    k_buf: Tensor,
    /// [1, n_heads, max_len, head_dim]
    v_buf: Tensor,
    /// Number of filled positions (0..max_len)
    pub len: usize,
    pub max_len: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Pre-allocated position-index tensor [1, n_heads, 1, head_dim] (i64).
    /// During normal decode: filled with `len` by `append()`.
    /// During CUDA graph: filled externally via H2D copy before each graph replay,
    /// then `append_with_fixed_idx()` uses it directly (stable CUDA address).
    pub pos_idx: Tensor,
}

impl StaticKvLayer {
    /// Create from the dynamic KV tensors accumulated during prefill.
    /// Allocates buffers of size `max_len` and copies prefill data in.
    pub fn from_dynamic(k: Tensor, v: Tensor, max_len: usize) -> Result<Self> {
        let (batch, n_heads, cur_len, head_dim) = k.dims4()?;
        assert!(
            cur_len <= max_len,
            "static KV max_len ({max_len}) must be >= prefill length ({cur_len})"
        );
        let device = k.device().clone();
        let dtype = k.dtype();

        // Pre-allocate full-length zero buffers
        let k_zeros = Tensor::zeros((batch, n_heads, max_len, head_dim), dtype, &device)?;
        let v_zeros = Tensor::zeros((batch, n_heads, max_len, head_dim), dtype, &device)?;

        // Copy prefill K/V into the first cur_len positions (one-time init)
        let k_buf = k_zeros.slice_assign(&[0..batch, 0..n_heads, 0..cur_len, 0..head_dim], &k)?;
        let v_buf = v_zeros.slice_assign(&[0..batch, 0..n_heads, 0..cur_len, 0..head_dim], &v)?;

        // Pre-allocate position-index tensor (stable address for CUDA graph capture)
        let pos_idx = Tensor::full(cur_len as i64, (batch, n_heads, 1usize, head_dim), &device)?
            .to_dtype(DType::I64)?;

        Ok(Self {
            k_buf,
            v_buf,
            len: cur_len,
            max_len,
            n_heads,
            head_dim,
            pos_idx,
        })
    }

    /// Append a single decode token's K and V (in-place via `scatter_set`).
    /// Used in eager (non-graph) decode mode.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        if self.len >= self.max_len {
            return Err(candle_core::Error::Msg(format!(
                "StaticKvLayer: KV cache full ({}/{})",
                self.len, self.max_len
            ))
            .into());
        }
        // Overwrite pos_idx in-place so it contains the current `len` value.
        // (For graph mode this is done externally via H2D copy; for eager mode this allocates
        //  a tiny i64 tensor — the main savings still come from k_buf/v_buf not reallocating.)
        let idx = Tensor::full(self.len as i64, self.pos_idx.shape(), k_new.device())?
            .to_dtype(DType::I64)?;
        self.k_buf.scatter_set(&idx, k_new, 2)?;
        self.v_buf.scatter_set(&idx, v_new, 2)?;
        self.len += 1;
        Ok(())
    }

    /// Append using the pre-allocated `self.pos_idx` tensor (stable CUDA address).
    /// Caller MUST have updated `pos_idx` to the current position via an H2D copy
    /// before calling this — this is the path used inside a captured CUDA graph.
    pub fn append_with_fixed_idx(&self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        self.k_buf.scatter_set(&self.pos_idx, k_new, 2)?;
        self.v_buf.scatter_set(&self.pos_idx, v_new, 2)?;
        // NOTE: `self.len` is intentionally NOT incremented here.
        //        The caller increments `static_kv.len` (via StaticKvManager) after replay.
        Ok(())
    }

    /// Full K buffer [1, n_heads, max_len, head_dim] — includes padding zeros beyond `len`.
    pub fn k(&self) -> &Tensor {
        &self.k_buf
    }
    /// Full V buffer [1, n_heads, max_len, head_dim] — includes padding zeros beyond `len`.
    pub fn v(&self) -> &Tensor {
        &self.v_buf
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Build an attention mask: 0.0 for valid positions, -1e9 for padding.
    /// Shape: [1, 1, 1, max_len] — broadcast to [batch, n_heads, 1, max_len] in attention.
    pub fn make_mask(&self) -> Result<Tensor> {
        let device = self.k_buf.device();
        let dtype = self.k_buf.dtype();
        let mut vals = vec![0f32; self.max_len];
        // Positions from self.len onward are unused — mask them out
        for v in vals.iter_mut().take(self.max_len).skip(self.len) {
            *v = -1e9f32;
        }
        Ok(
            Tensor::from_slice(&vals, (1usize, 1usize, 1usize, self.max_len), device)?
                .to_dtype(dtype)?,
        )
    }
}

/// Pre-allocated KV cache manager for all transformer layers.
pub struct StaticKvManager {
    pub layers: Vec<StaticKvLayer>,
    pub max_len: usize,
}

impl StaticKvManager {
    /// Build from a dynamic `KvCacheManager` produced by prefill.
    pub fn from_dynamic(dynamic: KvCacheManager, max_len: usize) -> Result<Self> {
        let layers = dynamic
            .into_caches()
            .into_iter()
            .map(|cache| {
                let (k_opt, v_opt) = cache.into_tensors();
                let k = k_opt.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "StaticKvManager: missing K tensor in prefill cache".into(),
                    )
                })?;
                let v = v_opt.ok_or_else(|| {
                    candle_core::Error::Msg(
                        "StaticKvManager: missing V tensor in prefill cache".into(),
                    )
                })?;
                StaticKvLayer::from_dynamic(k, v, max_len)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, max_len })
    }

    /// Current valid sequence length (same for all layers after prefill).
    pub fn len(&self) -> usize {
        self.layers.first().map(|l| l.len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Increment each layer's `len` by 1 (called after a graph-replay decode step).
    pub fn step(&mut self) {
        for layer in &mut self.layers {
            layer.len += 1;
        }
    }

    /// Build a fresh mask for the current cache length.
    /// Returns `None` when all positions are valid (no padding yet — impossible since we always
    /// have padding beyond the prefill length).
    pub fn make_mask(&self) -> Result<Tensor> {
        self.layers
            .first()
            .ok_or_else(|| candle_core::Error::Msg("StaticKvManager: no layers".into()))?
            .make_mask()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_core::Device;

    #[test]
    fn test_kv_cache_update() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = KvCache::new();

        // First token: [batch=1, heads=8, seq=1, head_dim=64]
        let k1 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        let v1 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;

        let (k_out, v_out) = cache.update(k1, v1)?;
        assert_eq!(k_out.dims(), &[1, 8, 1, 64]);
        assert_eq!(v_out.dims(), &[1, 8, 1, 64]);
        assert_eq!(cache.len(), 1);

        // Second token
        let k2 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        let v2 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;

        let (k_out, v_out) = cache.update(k2, v2)?;
        assert_eq!(k_out.dims(), &[1, 8, 2, 64]);
        assert_eq!(v_out.dims(), &[1, 8, 2, 64]);
        assert_eq!(cache.len(), 2);

        // Third token
        let k3 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        let v3 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;

        let (k_out, v_out) = cache.update(k3, v3)?;
        assert_eq!(k_out.dims(), &[1, 8, 3, 64]);
        assert_eq!(v_out.dims(), &[1, 8, 3, 64]);
        assert_eq!(cache.len(), 3);

        Ok(())
    }

    #[test]
    fn test_kv_cache_manager() -> Result<()> {
        let device = Device::Cpu;
        let mut manager = KvCacheManager::new(4);

        assert_eq!(manager.num_layers(), 4);
        assert_eq!(manager.len(), 0);

        // Update cache for layer 0
        let cache = manager.get_or_create(0);
        let k = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        let v = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        cache.update(k, v)?;

        assert_eq!(manager.len(), 1);

        // Reset all
        manager.reset();
        assert_eq!(manager.len(), 0);

        Ok(())
    }
}
