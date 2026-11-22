use crate::{Error, Result};
use candle_core::Tensor;
use rand::SeedableRng;
use rand::{Rng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::sync::{Mutex, MutexGuard};

/// Owned by one model pipeline; never resets a shared device or thread RNG.
#[derive(Default)]
pub(crate) struct SamplingRng(Mutex<Option<StdRng>>);

impl SamplingRng {
    fn lock(&self) -> Result<MutexGuard<'_, Option<StdRng>>> {
        self.0
            .lock()
            .map_err(|_| Error::InferenceError("sampling RNG lock poisoned".into()))
    }

    pub(crate) fn set_seed(&self, seed: Option<u64>) -> Result<()> {
        *self.lock()? = seed.map(StdRng::seed_from_u64);
        Ok(())
    }

    pub(crate) fn uniform(&self) -> Result<f32> {
        Ok(match self.lock()?.as_mut() {
            Some(rng) => rng.random(),
            None => rand::random(),
        })
    }

    pub(crate) fn normal_like(&self, mean: &Tensor) -> Result<Tensor> {
        let mut state = self.lock()?;
        let noise = if let Some(rng) = state.as_mut() {
            let values: Vec<f32> = StandardNormal
                .sample_iter(rng)
                .take(mean.elem_count())
                .collect();
            Tensor::from_vec(values, mean.shape(), mean.device())?
        } else {
            Tensor::randn(0.0f32, 1.0, mean.shape(), mean.device())?
        };
        Ok(noise.to_dtype(mean.dtype())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn sequence(rng: &SamplingRng) -> Vec<f32> {
        (0..32).map(|_| rng.uniform().unwrap()).collect()
    }

    #[test]
    fn seeds_replay_and_reset_without_affecting_other_instances() {
        let a = SamplingRng::default();
        let b = SamplingRng::default();
        a.set_seed(Some(0)).unwrap();
        let expected = sequence(&a);
        a.set_seed(Some(0)).unwrap();
        b.set_seed(Some(u64::MAX)).unwrap();
        let other = sequence(&b);
        assert_ne!(expected, other);
        assert_eq!(expected, sequence(&a));
        a.set_seed(None).unwrap();
        assert!(a.lock().unwrap().is_none());
        a.set_seed(Some(0)).unwrap();
        assert_eq!(expected, sequence(&a));
    }

    #[test]
    fn seeded_normal_noise_is_replayable_on_cpu_without_device_seeding() {
        let rng = SamplingRng::default();
        let mean = Tensor::zeros((2, 4096), DType::F32, &Device::Cpu).unwrap();
        rng.set_seed(Some(42)).unwrap();
        let a = rng.normal_like(&mean).unwrap();
        rng.set_seed(Some(42)).unwrap();
        let b = rng.normal_like(&mean).unwrap();
        assert_eq!(a.dims(), mean.dims());
        let values = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, b.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        let average = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|v| (v - average).powi(2)).sum::<f32>() / values.len() as f32;
        assert!(average.abs() < 0.05, "mean {average}");
        assert!((variance - 1.0).abs() < 0.1, "variance {variance}");
    }
}
