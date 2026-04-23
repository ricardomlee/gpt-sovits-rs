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

use candle_core::Tensor;
use crate::Result;

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
        assert_eq!(cache.len(), 1);

        // Second token
        let k2 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        let v2 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;

        let (k_out, v_out) = cache.update(k2, v2)?;
        assert_eq!(k_out.dims(), &[1, 8, 2, 64]);
        assert_eq!(cache.len(), 2);

        // Third token
        let k3 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;
        let v3 = Tensor::ones((1, 8, 1, 64), DType::F32, &device)?;

        let (k_out, v_out) = cache.update(k3, v3)?;
        assert_eq!(k_out.dims(), &[1, 8, 3, 64]);
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
