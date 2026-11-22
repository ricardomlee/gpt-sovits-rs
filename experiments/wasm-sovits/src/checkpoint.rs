use candle_core::{Device, Tensor, safetensors::Load};
use gpt_sovits_rs::{Error, Result};
use safetensors::tensor::{Metadata, TensorInfo, TensorView};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};

const MAX_HEADER: u64 = 16 * 1024 * 1024;

fn invalid(error: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Invalid safetensors checkpoint: {error}"))
}

pub fn header_end(prefix: &[u8], size: u32) -> Result<u32> {
    let length = u64::from_le_bytes(
        prefix
            .try_into()
            .map_err(|_| invalid("missing header length"))?,
    );
    if length == 0 || length > MAX_HEADER || length + 8 > u64::from(size) {
        return Err(invalid(
            "header length is outside the file or exceeds 16 MiB",
        ));
    }
    Ok((length + 8) as u32)
}

#[derive(Deserialize)]
struct Header {
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

pub struct Checkpoint {
    payload_start: u32,
    pending: VecDeque<(String, TensorInfo)>,
    weights: HashMap<String, Tensor>,
}

impl Checkpoint {
    pub fn new(header: &[u8], size: u32) -> Result<Self> {
        let prefix = header
            .get(..8)
            .ok_or_else(|| invalid("missing header length"))?;
        let payload_start = header_end(prefix, size)?;
        if header.len() != payload_start as usize {
            return Err(invalid("incomplete header"));
        }
        let header: Header = serde_json::from_slice(&header[8..]).map_err(invalid)?;
        let mut tensors = header.tensors.into_iter().collect::<Vec<_>>();
        tensors.sort_unstable_by_key(|(_, tensor)| tensor.data_offsets);
        let metadata = Metadata::new(header.metadata, tensors.clone()).map_err(invalid)?;
        if tensors.is_empty() || metadata.data_len() != (size - payload_start) as usize {
            return Err(invalid("tensor data size does not match the file"));
        }
        Ok(Self {
            payload_start,
            weights: HashMap::with_capacity(tensors.len()),
            pending: tensors.into(),
        })
    }

    pub fn next_range(&self) -> Option<(u32, u32)> {
        let (start, end) = self.pending.front()?.1.data_offsets;
        Some((
            self.payload_start + start as u32,
            self.payload_start + end as u32,
        ))
    }

    pub fn push(&mut self, bytes: &[u8]) -> Result<()> {
        let (_, info) = self
            .pending
            .front()
            .ok_or_else(|| invalid("unexpected extra tensor"))?;
        let view = TensorView::new(info.dtype, info.shape.clone(), bytes).map_err(invalid)?;
        let weight = view.load(&Device::Cpu)?;
        let (name, _) = self.pending.pop_front().unwrap();
        self.weights.insert(name, weight);
        Ok(())
    }

    pub fn finish(self) -> Result<HashMap<String, Tensor>> {
        if !self.pending.is_empty() {
            return Err(invalid("incomplete tensor data"));
        }
        Ok(self.weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_and_validates_ranges_in_checkpoint_order() -> Result<()> {
        let first = [0, 0, 128, 63];
        let second = [0, 0, 0, 64];
        let view = |bytes| TensorView::new(safetensors::Dtype::F32, vec![1], bytes).unwrap();
        let weights = HashMap::from([("a", view(&first)), ("b", view(&second))]);
        let bytes = safetensors::serialize(weights, None).unwrap();
        let size = bytes.len() as u32;
        let end = header_end(&bytes[..8], size)? as usize;
        assert!(header_end(&u64::MAX.to_le_bytes(), size).is_err());
        assert!(Checkpoint::new(&bytes[..end], size - 1).is_err());
        assert!(Checkpoint::new(&bytes[..end], size)?.finish().is_err());
        let mut checkpoint = Checkpoint::new(&bytes[..end], size)?;
        assert!(checkpoint.push(&[0; 3]).is_err());
        while let Some((start, end)) = checkpoint.next_range() {
            checkpoint.push(&bytes[start as usize..end as usize])?;
        }
        assert!(checkpoint.push(&[]).is_err());
        let tensors = checkpoint.finish()?;
        assert_eq!(tensors["a"].to_vec1::<f32>()?, [1.0]);
        assert_eq!(tensors["b"].to_vec1::<f32>()?, [2.0]);
        Ok(())
    }
}
