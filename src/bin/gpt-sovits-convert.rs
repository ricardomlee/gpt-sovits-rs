use anyhow::{bail, Context, Result};
use candle_core::{pickle::PthTensors, DType, Device, Tensor};
use clap::{Parser, Subcommand};
use safetensors::tensor::serialize_to_file;
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "gpt-sovits-convert")]
#[command(author = "GPT-SoVITS Rust Contributors")]
#[command(version)]
#[command(about = "Convert GPT-SoVITS PyTorch checkpoints to safetensors")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convert a GPT semantic model .ckpt to safetensors.
    Gpt {
        checkpoint: PathBuf,
        output: PathBuf,
    },
    /// Convert a SoVITS .pth checkpoint to safetensors.
    Sovits {
        checkpoint: PathBuf,
        output: PathBuf,
    },
    /// Convert Chinese RoBERTa pytorch_model.bin to safetensors.
    Bert {
        checkpoint: PathBuf,
        output: PathBuf,
    },
    /// Convert Chinese HuBERT pytorch_model.bin to safetensors.
    Hubert {
        checkpoint: PathBuf,
        output: PathBuf,
    },
    /// Convert a v2Pro speaker-verification embedding .pt to safetensors.
    Sv { embedding: PathBuf, output: PathBuf },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Gpt { checkpoint, output } => {
            convert_checkpoint(&checkpoint, &output, ModelKind::Gpt)
        }
        Command::Sovits { checkpoint, output } => {
            convert_checkpoint(&checkpoint, &output, ModelKind::Sovits)
        }
        Command::Bert { checkpoint, output } => convert_bert(&checkpoint, &output),
        Command::Hubert { checkpoint, output } => convert_hubert(&checkpoint, &output),
        Command::Sv { embedding, output } => convert_sv_embedding(&embedding, &output),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelKind {
    Gpt,
    Sovits,
}

fn convert_checkpoint(input: &Path, output: &Path, kind: ModelKind) -> Result<()> {
    let prepared = PreparedCheckpoint::open(input)?;
    let tensors = read_state_dict(prepared.path(), &["weight", "state_dict", ""])?;
    let required = match kind {
        ModelKind::Gpt => "model.ar_text_embedding.word_embeddings.weight",
        ModelKind::Sovits => "enc_p.text_embedding.weight",
    };
    if !tensors.iter().any(|(name, _)| name == required) {
        bail!("checkpoint is missing required tensor {required}");
    }

    let mut metadata = HashMap::new();
    match kind {
        ModelKind::Gpt => {
            metadata.insert("model_type".to_string(), "gpt".to_string());
        }
        ModelKind::Sovits => {
            metadata.insert("model_type".to_string(), "sovits".to_string());
            metadata.insert(
                "model_version".to_string(),
                sovits_version(input, &tensors).to_string(),
            );
        }
    }
    write_safetensors(output, tensors, Some(metadata))?;
    Ok(())
}

fn convert_sv_embedding(input: &Path, output: &Path) -> Result<()> {
    let prepared = PreparedCheckpoint::open(input)?;
    let tensor = match read_state_dict(prepared.path(), &["", "weight", "state_dict"]) {
        Ok(mut tensors) if tensors.len() == 1 => tensors.pop().expect("checked len").1,
        Ok(tensors) => bail!(
            "SV embedding checkpoint must contain exactly one tensor, found {}",
            tensors.len()
        ),
        Err(_) => read_bare_sv_tensor(prepared.path())?,
    };
    let tensor = tensor.to_dtype(DType::F32)?;
    let dims = tensor.dims();
    let tensor = match dims {
        [20480] => tensor.unsqueeze(0)?,
        [1, 20480] => tensor,
        other => bail!("SV embedding must have shape [20480] or [1, 20480], got {other:?}"),
    };
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "sv_embedding".to_string());
    metadata.insert("shape".to_string(), "1,20480".to_string());
    write_safetensors(
        output,
        vec![("sv_embedding".to_string(), tensor)],
        Some(metadata),
    )
}

fn convert_bert(input: &Path, output: &Path) -> Result<()> {
    let prepared = PreparedCheckpoint::open(input)?;
    let raw = read_state_dict(prepared.path(), &["", "state_dict", "model"])?;
    let mut tensors = Vec::new();
    for (name, tensor) in raw {
        if name == "bert.embeddings.position_ids" {
            continue;
        }
        if let Some(rest) = name.strip_prefix("bert.embeddings.") {
            tensors.push((format!("embeddings.{rest}"), tensor));
            continue;
        }
        if let Some(rest) = name.strip_prefix("bert.encoder.layer.") {
            let layer = rest
                .split('.')
                .next()
                .and_then(|part| part.parse::<usize>().ok());
            if layer.is_some_and(|layer| layer < 22) {
                tensors.push((format!("encoder.layer.{rest}"), tensor));
            }
        }
    }
    let required = "encoder.layer.21.output.LayerNorm.bias";
    if !tensors.iter().any(|(name, _)| name == required) {
        bail!("BERT checkpoint is missing {required}");
    }
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "bert".to_string());
    write_safetensors(output, tensors, Some(metadata))
}

fn convert_hubert(input: &Path, output: &Path) -> Result<()> {
    let prepared = PreparedCheckpoint::open(input)?;
    let raw = read_state_dict(prepared.path(), &["", "state_dict", "model"])?;
    let weight_g = find_tensor(&raw, "encoder.pos_conv_embed.conv.weight_g")?;
    let weight_v = find_tensor(&raw, "encoder.pos_conv_embed.conv.weight_v")?;
    let pos_conv_weight = normalize_weight_norm(weight_g, weight_v)?;

    let mut tensors = raw
        .into_iter()
        .filter(|(name, _)| {
            name != "masked_spec_embed" && !name.starts_with("encoder.pos_conv_embed.conv.weight_")
        })
        .collect::<Vec<_>>();
    tensors.push((
        "encoder.pos_conv_embed.conv.weight".to_string(),
        pos_conv_weight,
    ));

    let required = "encoder.layers.11.final_layer_norm.bias";
    if !tensors.iter().any(|(name, _)| name == required) {
        bail!("HuBERT checkpoint is missing {required}");
    }
    let mut metadata = HashMap::new();
    metadata.insert("model_type".to_string(), "hubert".to_string());
    write_safetensors(output, tensors, Some(metadata))
}

fn find_tensor<'a>(tensors: &'a [(String, Tensor)], name: &str) -> Result<&'a Tensor> {
    tensors
        .iter()
        .find(|(tensor_name, _)| tensor_name == name)
        .map(|(_, tensor)| tensor)
        .ok_or_else(|| anyhow::anyhow!("checkpoint is missing {name}"))
}

fn normalize_weight_norm(weight_g: &Tensor, weight_v: &Tensor) -> Result<Tensor> {
    let weight_g = weight_g.to_dtype(DType::F32)?;
    let weight_v = weight_v.to_dtype(DType::F32)?;
    let norm = weight_v
        .sqr()?
        .sum_keepdim(0)?
        .sum_keepdim(1)?
        .sqrt()?
        .clamp(1e-12f32, f32::INFINITY)?;
    Ok(weight_v.broadcast_div(&norm)?.broadcast_mul(&weight_g)?)
}

fn read_bare_sv_tensor(path: &Path) -> Result<Tensor> {
    let file = fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut zip = zip::ZipArchive::new(reader)?;
    let names: Vec<String> = zip.file_names().map(str::to_string).collect();
    let data_names: Vec<_> = names
        .iter()
        .filter(|name| {
            name.rsplit_once('/')
                .is_some_and(|(_, file)| file == "0" && name.contains("/data/"))
        })
        .cloned()
        .collect();
    if data_names.len() != 1 {
        bail!(
            "bare SV .pt must contain exactly one data/0 tensor storage, found {}",
            data_names.len()
        );
    }
    let data_name = &data_names[0];
    let size = zip.by_name(data_name)?.size() as usize;
    let dtype = match size {
        40960 => DType::F16,
        81920 => DType::F32,
        _ => bail!("bare SV tensor storage has unexpected byte length {size}"),
    };
    let mut reader = zip.by_name(data_name)?;
    let mut bytes = Vec::with_capacity(size);
    reader.read_to_end(&mut bytes)?;
    match dtype {
        DType::F16 => {
            let values = bytes
                .chunks_exact(2)
                .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(values, (1, 20480), &Device::Cpu)?)
        }
        DType::F32 => {
            let values = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>();
            Ok(Tensor::from_vec(values, (1, 20480), &Device::Cpu)?)
        }
        _ => unreachable!("only f16/f32 are matched above"),
    }
}

fn read_state_dict(path: &Path, keys: &[&str]) -> Result<Vec<(String, Tensor)>> {
    let mut last_error = None;
    for key in keys {
        let key = if key.is_empty() { None } else { Some(*key) };
        match read_state_dict_with_key(path, key) {
            Ok(tensors) if !tensors.is_empty() => return Ok(tensors),
            Ok(_) => {
                last_error = Some(format!("no tensors found with key {key:?}"));
            }
            Err(err) => {
                last_error = Some(err.to_string());
            }
        }
    }
    bail!(
        "failed to read tensors from {}: {}",
        path.display(),
        last_error.unwrap_or_else(|| "no compatible state_dict found".to_string())
    )
}

fn read_state_dict_with_key(path: &Path, key: Option<&str>) -> Result<Vec<(String, Tensor)>> {
    let pth = PthTensors::new(path, key)?;
    let mut names: Vec<_> = pth.tensor_infos().keys().cloned().collect();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len());
    for name in names {
        if let Some(tensor) = pth.get(&name)? {
            tensors.push((name, tensor.to_dtype(DType::F32)?));
        }
    }
    Ok(tensors)
}

fn write_safetensors(
    output: &Path,
    tensors: Vec<(String, Tensor)>,
    metadata: Option<HashMap<String, String>>,
) -> Result<()> {
    if tensors.is_empty() {
        bail!("no tensors to write");
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = output.with_extension(format!(
        "{}.tmp",
        output
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("safetensors")
    ));
    serialize_to_file(tensors, metadata, &temporary)?;
    fs::rename(&temporary, output)?;
    let size_mb = output.metadata()?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "Saved {} tensors to {} ({size_mb:.1} MiB)",
        safetensor_count(output)?,
        output.display()
    );
    Ok(())
}

fn safetensor_count(path: &Path) -> Result<usize> {
    let data = fs::read(path)?;
    let st = safetensors::SafeTensors::deserialize(&data)?;
    Ok(st.names().len())
}

fn sovits_version(input: &Path, tensors: &[(String, Tensor)]) -> &'static str {
    match fs::read(input)
        .ok()
        .and_then(|bytes| bytes.get(0..2).map(|b| b.to_vec()))
    {
        Some(header) if header.as_slice() == b"00" => return "v1",
        Some(header) if header.as_slice() == b"01" => return "v2",
        Some(header) if header.as_slice() == b"02" => return "v3",
        Some(header) if header.as_slice() == b"03" => return "v3",
        Some(header) if header.as_slice() == b"04" => return "v4",
        Some(header) if header.as_slice() == b"05" => return "v2Pro",
        Some(header) if header.as_slice() == b"06" => return "v2ProPlus",
        _ => {}
    }
    if tensors.iter().any(|(name, _)| name == "ge_to512.weight")
        && tensors.iter().any(|(name, _)| name == "sv_emb.weight")
    {
        return "v2Pro";
    }
    if tensors
        .iter()
        .find(|(name, _)| name == "enc_p.text_embedding.weight")
        .is_some_and(|(_, tensor)| tensor.dims().first() == Some(&322))
    {
        return "v1";
    }
    "v2"
}

struct PreparedCheckpoint {
    path: PathBuf,
    temporary: Option<PathBuf>,
}

impl PreparedCheckpoint {
    fn open(input: &Path) -> Result<Self> {
        let bytes =
            fs::read(input).with_context(|| format!("failed to read {}", input.display()))?;
        if bytes.get(0..2) == Some(b"PK") {
            return Ok(Self {
                path: input.to_path_buf(),
                temporary: None,
            });
        }
        if !matches!(
            bytes.get(0..2),
            Some(b"00" | b"01" | b"02" | b"03" | b"04" | b"05" | b"06")
        ) {
            return Ok(Self {
                path: input.to_path_buf(),
                temporary: None,
            });
        }

        let mut patched = bytes;
        patched[0] = b'P';
        patched[1] = b'K';
        let temporary = std::env::temp_dir().join(format!(
            "gpt-sovits-convert-{}-{}.pth",
            std::process::id(),
            input
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("checkpoint")
        ));
        fs::write(&temporary, patched)
            .with_context(|| format!("failed to write temporary {}", temporary.display()))?;
        Ok(Self {
            path: temporary.clone(),
            temporary: Some(temporary),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PreparedCheckpoint {
    fn drop(&mut self) {
        if let Some(path) = self.temporary.as_ref() {
            let _ = fs::remove_file(path);
        }
    }
}
