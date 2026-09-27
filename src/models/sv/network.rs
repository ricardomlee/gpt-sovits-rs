//! ERes2NetV2 w24s4ep4 forward3 topology from GPT-SoVITS / 3D-Speaker.
//! Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
//! Apache-2.0; see licenses/3D-Speaker-Apache-2.0.txt.
//! Adapted to Rust/Candle; training/classifier omitted. See docs/SV.md.

use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT,
    VarBuilder,
};

struct ConvBn {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBn {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &VarBuilder<'_>,
        conv: &str,
        bn: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let config = Conv2dConfig {
            stride,
            padding: kernel / 2,
            ..Default::default()
        };
        let conv = if bias {
            conv2d(input, output, kernel, config, vb.pp(conv))?
        } else {
            conv2d_no_bias(input, output, kernel, config, vb.pp(conv))?
        };
        Ok(Self {
            conv,
            bn: batch_norm(output, 1e-5, vb.pp(bn))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.bn.forward_t(&self.conv.forward(x)?, false)
    }
}

struct Fusion {
    first: ConvBn,
    second: ConvBn,
}

impl Fusion {
    fn load(vb: VarBuilder<'_>, channels: usize) -> Result<Self> {
        let vb = vb.pp("local_att");
        Ok(Self {
            first: ConvBn::load(&vb, "0", "1", channels * 2, channels / 4, 1, 1, true)?,
            second: ConvBn::load(&vb, "3", "4", channels / 4, channels, 1, 1, true)?,
        })
    }
    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let joined = Tensor::cat(&[x, y], 1)?;
        let attention = self
            .second
            .forward(&candle_nn::ops::silu(&self.first.forward(&joined)?)?)?
            .tanh()?;
        (x * (&attention + 1.0)?)? + (y * (1.0 - &attention)?)?
    }
}

struct Block {
    first: ConvBn,
    splits: Vec<ConvBn>,
    fusions: Vec<Fusion>,
    last: ConvBn,
    shortcut: Option<ConvBn>,
    width: usize,
}

impl Block {
    fn load(
        vb: VarBuilder<'_>,
        input: usize,
        planes: usize,
        stride: usize,
        fused: bool,
    ) -> Result<Self> {
        let width = planes * 24 / 64;
        let output = planes * 4;
        Ok(Self {
            first: ConvBn::load(&vb, "conv1", "bn1", input, width * 4, 1, stride, false)?,
            splits: (0..4)
                .map(|i| {
                    ConvBn::load(
                        &vb,
                        &format!("convs.{i}"),
                        &format!("bns.{i}"),
                        width,
                        width,
                        3,
                        1,
                        false,
                    )
                })
                .collect::<Result<_>>()?,
            fusions: if fused {
                (0..3)
                    .map(|i| Fusion::load(vb.pp(format!("fuse_models.{i}")), width))
                    .collect::<Result<_>>()?
            } else {
                vec![]
            },
            last: ConvBn::load(&vb, "conv3", "bn3", width * 4, output, 1, 1, false)?,
            shortcut: if stride != 1 || input != output {
                Some(ConvBn::load(
                    &vb,
                    "shortcut.0",
                    "shortcut.1",
                    input,
                    output,
                    1,
                    stride,
                    false,
                )?)
            } else {
                None
            },
            width,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let split = self.first.forward(x)?.clamp(0f32, 20f32)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(4);
        for (i, conv) in self.splits.iter().enumerate() {
            let mut part = split.narrow(1, i * self.width, self.width)?.contiguous()?;
            if let Some(previous) = outputs.last() {
                part = if self.fusions.is_empty() {
                    (previous + &part)?
                } else {
                    self.fusions[i - 1].forward(previous, &part)?
                };
            }
            outputs.push(conv.forward(&part)?.clamp(0f32, 20f32)?);
        }
        let out = self.last.forward(&Tensor::cat(&outputs, 1)?)?;
        let residual = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        (out + residual)?.clamp(0f32, 20f32)
    }
}

pub(super) struct Network {
    stem: ConvBn,
    stages: Vec<Vec<Block>>,
    downsample: Conv2d,
    fusion: Fusion,
}

impl Network {
    pub(super) fn load(vb: VarBuilder<'_>) -> Result<Self> {
        let stem = ConvBn::load(&vb, "conv1", "bn1", 1, 64, 3, 1, false)?;
        let mut input = 64;
        let mut stages = Vec::new();
        for (stage, blocks) in [3, 4, 6, 3].into_iter().enumerate() {
            let planes = 64 << stage;
            let mut layer = Vec::new();
            for block in 0..blocks {
                let stride = if stage > 0 && block == 0 { 2 } else { 1 };
                layer.push(Block::load(
                    vb.pp(format!("layer{}.{block}", stage + 1)),
                    input,
                    planes,
                    stride,
                    stage >= 2,
                )?);
                input = planes * 4;
            }
            stages.push(layer);
        }
        Ok(Self {
            stem,
            stages,
            downsample: conv2d_no_bias(
                1024,
                2048,
                3,
                Conv2dConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
                vb.pp("layer3_ds"),
            )?,
            fusion: Fusion::load(vb.pp("fuse34"), 2048)?,
        })
    }
    pub(super) fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut x = self
            .stem
            .forward(&features.transpose(1, 2)?.unsqueeze(1)?.contiguous()?)?
            .relu()?;
        let mut third = None;
        for (stage, blocks) in self.stages.iter().enumerate() {
            for block in blocks {
                x = block.forward(&x)?;
            }
            if stage == 2 {
                third = Some(x.clone());
            }
        }
        let down = self
            .downsample
            .forward(&third.expect("four fixed stages"))?;
        self.fusion.forward(&x, &down)?.flatten(1, 2)?.mean(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn conv_bn_uses_running_statistics_in_eval_mode() -> Result<()> {
        let device = &Device::Cpu;
        let weights = HashMap::from([
            (
                "conv.weight".into(),
                Tensor::full(2f32, (1, 1, 1, 1), device)?,
            ),
            ("bn.weight".into(), Tensor::new(&[6f32], device)?),
            ("bn.bias".into(), Tensor::new(&[1f32], device)?),
            ("bn.running_mean".into(), Tensor::new(&[3f32], device)?),
            ("bn.running_var".into(), Tensor::new(&[4f32], device)?),
        ]);
        let vb = VarBuilder::from_tensors(weights, DType::F32, device);
        let layer = ConvBn::load(&vb, "conv", "bn", 1, 1, 1, 1, false)?;
        let out = layer.forward(&Tensor::full(5f32, (1, 1, 1, 1), device)?)?;
        let expected = (10f32 - 3.0) / (4f32 + 1e-5).sqrt() * 6.0 + 1.0;
        assert!((out.flatten_all()?.to_vec1::<f32>()?[0] - expected).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn fusion_uses_unnormalized_tanh_gates_in_the_correct_order() -> Result<()> {
        let device = &Device::Cpu;
        let mut weights = HashMap::new();
        for (conv, bn, input, output) in [("0", "1", 8, 1), ("3", "4", 1, 4)] {
            weights.insert(
                format!("local_att.{conv}.weight"),
                Tensor::zeros((output, input, 1, 1), DType::F32, device)?,
            );
            weights.insert(
                format!("local_att.{conv}.bias"),
                Tensor::zeros(output, DType::F32, device)?,
            );
            for field in ["weight", "running_var", "bias", "running_mean"] {
                let value = if matches!(field, "weight" | "running_var") {
                    1f32
                } else {
                    0f32
                };
                weights.insert(
                    format!("local_att.{bn}.{field}"),
                    Tensor::full(value, output, device)?,
                );
            }
        }
        let gates = [0.5f32, -0.5, 0.0, 2.0];
        weights.insert("local_att.4.bias".into(), Tensor::new(&gates, device)?);
        let fusion = Fusion::load(VarBuilder::from_tensors(weights, DType::F32, device), 4)?;
        let x = Tensor::full(2f32, (1, 4, 1, 1), device)?;
        let y = Tensor::full(5f32, (1, 4, 1, 1), device)?;
        let out = fusion.forward(&x, &y)?.flatten_all()?.to_vec1::<f32>()?;
        for (actual, gate) in out.iter().zip(gates) {
            let expected = 2.0 * (1.0 + gate.tanh()) + 5.0 * (1.0 - gate.tanh());
            assert!((actual - expected).abs() < 1e-5);
        }
        Ok(())
    }
}
