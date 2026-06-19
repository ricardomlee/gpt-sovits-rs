//! Module configuration

use serde::{Deserialize, Serialize};

/// Configuration for the GPT-SoVITS pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Device to use for inference
    pub device: Device,
    /// Use half precision (FP16)
    pub half_precision: bool,
    /// Model version
    pub model_version: ModelVersion,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cuda,
    Cpu,
    Mps,
}

impl Default for Device {
    fn default() -> Self {
        Device::Cuda
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cuda => write!(f, "cuda"),
            Device::Cpu => write!(f, "cpu"),
            Device::Mps => write!(f, "mps"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelVersion {
    V1,
    V2,
    V2Pro,
    V3,
    V4,
}

impl Default for ModelVersion {
    fn default() -> Self {
        ModelVersion::V2
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: Device::default(),
            half_precision: true,
            model_version: ModelVersion::default(),
        }
    }
}

impl Config {
    /// Create a new config builder
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// Get the Candle device type
    pub fn candle_device(&self) -> candle_core::Device {
        match self.device {
            Device::Cuda => candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu),
            Device::Cpu => candle_core::Device::Cpu,
            Device::Mps => candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu),
        }
    }
}

/// Builder for Config
#[derive(Default)]
pub struct ConfigBuilder {
    device: Option<Device>,
    half_precision: Option<bool>,
    model_version: Option<ModelVersion>,
}

impl ConfigBuilder {
    pub fn with_device(mut self, device: &str) -> Self {
        self.device = Some(match device.to_lowercase().as_str() {
            "cuda" => Device::Cuda,
            "cpu" => Device::Cpu,
            "mps" => Device::Mps,
            _ => Device::Cpu,
        });
        self
    }

    /// Auto-detect and use GPU if available (CUDA > Metal > CPU)
    pub fn with_auto_device(mut self) -> Self {
        // Try CUDA first, then Metal, fallback to CPU
        #[cfg(feature = "cuda")]
        {
            if candle_core::Device::new_cuda(0).is_ok() {
                self.device = Some(Device::Cuda);
                return self;
            }
        }
        // Try Metal
        if candle_core::Device::new_metal(0).is_ok() {
            self.device = Some(Device::Mps);
            return self;
        }
        // Fallback to CPU
        self.device = Some(Device::Cpu);
        self
    }

    pub fn with_half_precision(mut self, half: bool) -> Self {
        self.half_precision = Some(half);
        self
    }

    pub fn with_model_version(mut self, version: ModelVersion) -> Self {
        self.model_version = Some(version);
        self
    }

    pub fn build(self) -> Config {
        Config {
            device: self.device.unwrap_or_else(|| {
                // Auto-detect: prefer GPU if available
                #[cfg(feature = "cuda")]
                {
                    Device::Cuda
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Device::Cpu
                }
            }),
            half_precision: self.half_precision.unwrap_or(true),
            model_version: self.model_version.unwrap_or_default(),
        }
    }
}
