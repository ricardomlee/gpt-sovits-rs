use super::{ModelKind, convert_bert, convert_checkpoint, convert_hubert};
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;

const DOWNLOADS: &str = include_str!("../../../examples/first-run/models.toml");
const VOICE: &str = include_str!("../../../examples/first-run/voice.json");
const MARKER: &str = ".gpt-sovits-demo";
const DEMO: &str = "gpt-sovits-public-v2-demo";
const RECEIPT: &str = "download-sources.toml";
const OUTPUT_FILES: [&str; 7] = [
    "models/gpt-model.safetensors",
    "models/sovits-model.safetensors",
    "models/bert/bert.safetensors",
    "models/bert/tokenizer.json",
    "models/hubert/hubert.safetensors",
    "voices/demo/ref.wav",
    "voices/demo/voice.json",
];

#[derive(Args, Debug)]
pub(super) struct Options {
    /// Directory to create with the converted models and demo voice.
    #[arg(long, default_value = "demo-data")]
    output_dir: PathBuf,
    /// Read the original files from this directory without downloading.
    #[arg(long, conflicts_with = "cache_dir")]
    source_dir: Option<PathBuf>,
    /// Store downloaded original files in this directory.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Override the source URLs using a TOML file with the pinned demo hashes.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Try ModelScope before the primary URL for the upstream models.
    #[arg(long)]
    modelscope: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    files: Vec<Download>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Download {
    path: String,
    sha256: String,
    url: String,
    modelscope: Option<String>,
}

impl Manifest {
    fn same_inputs(&self, other: &Self) -> bool {
        self.files.len() == other.files.len()
            && self.files.iter().all(|file| {
                other.files.iter().any(|other| {
                    other.path == file.path && other.sha256.eq_ignore_ascii_case(&file.sha256)
                })
            })
    }
}

pub(super) fn prepare(options: Options) -> Result<()> {
    let output = &options.output_dir;
    let bundled = std::env::current_exe()?.with_file_name("models.toml");
    let manifest_path = match options.manifest {
        Some(path) => Some(path),
        None => bundled.try_exists()?.then_some(bundled),
    };
    let contents = match manifest_path {
        Some(path) => {
            fs::read_to_string(&path).with_context(|| format!("Read {}", path.display()))?
        }
        None => DOWNLOADS.to_owned(),
    };
    let pinned: Manifest = toml::from_str(DOWNLOADS).context("Parse the pinned demo manifest")?;
    let manifest: Manifest = toml::from_str(&contents).context("Parse the demo manifest")?;
    ensure!(
        pinned.same_inputs(&manifest),
        "The manifest must keep the public demo file paths and pinned SHA-256 hashes."
    );
    for file in &manifest.files {
        for url in std::iter::once(&file.url).chain(file.modelscope.iter()) {
            ensure!(
                url.starts_with("https://") && !url.contains(char::is_whitespace),
                "Expected an HTTPS URL for {}.",
                file.path
            );
        }
    }
    if output.try_exists()? {
        ensure!(
            is_prepared(output, &pinned),
            "The output directory already exists: {}. Choose a new --output-dir.",
            output.display()
        );
        println!(
            "The public v2 demo is already prepared in {}.",
            output.display()
        );
        return Ok(());
    }

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let cache = options
        .cache_dir
        .unwrap_or_else(|| parent.join(".gpt-sovits-demo-source"));
    let source_dir = options.source_dir.as_deref().unwrap_or(&cache);
    for file in &manifest.files {
        let mirror = file.modelscope.as_deref().map(|url| ("ModelScope", url));
        let mut urls = [Some(("the primary source", file.url.as_str())), mirror];
        if options.modelscope && mirror.is_some() {
            urls.swap(0, 1);
        }
        prepare_source(
            &source_dir.join(&file.path),
            &file.sha256,
            urls.into_iter().flatten(),
            options.source_dir.is_none(),
        )?;
    }

    let staging = tempfile::Builder::new()
        .prefix(".prepare-demo-")
        .tempdir_in(parent)?;
    convert(source_dir, staging.path())?;
    fs::write(staging.path().join(RECEIPT), contents)?;
    fs::write(staging.path().join(MARKER), DEMO)?;
    if let Err(error) = fs::rename(staging.path(), output) {
        if !is_prepared(output, &pinned) {
            return Err(error)
                .with_context(|| format!("Install the demo into {}", output.display()));
        }
    } else {
        let _ = staging.keep();
    }
    println!("Prepared the public v2 demo in {}.", output.display());
    Ok(())
}

fn is_prepared(directory: &Path, expected: &Manifest) -> bool {
    fs::read_to_string(directory.join(MARKER)).is_ok_and(|marker| marker.trim() == DEMO)
        && fs::read_to_string(directory.join(RECEIPT))
            .ok()
            .and_then(|contents| toml::from_str(&contents).ok())
            .is_some_and(|receipt| expected.same_inputs(&receipt))
        && OUTPUT_FILES
            .iter()
            .all(|path| directory.join(path).is_file())
}

fn prepare_source<'a>(
    path: &Path,
    digest: &str,
    urls: impl Iterator<Item = (&'a str, &'a str)>,
    download: bool,
) -> Result<()> {
    if path.try_exists()? {
        verify(path, digest)?;
        println!("Verified {}", path.display());
        return Ok(());
    }
    ensure!(download, "Missing source file: {}", path.display());
    let parent = path.parent().context("Missing demo cache directory")?;
    fs::create_dir_all(parent)?;
    let partial = NamedTempFile::new_in(parent)?.into_temp_path();
    for (source, url) in urls {
        println!("Downloading {} from {source}", path.display());
        let status = Command::new(if cfg!(windows) { "curl.exe" } else { "curl" })
            .args([
                "--fail",
                "--location",
                "--retry",
                "2",
                "--connect-timeout",
                "15",
                "--max-time",
                "1800",
                "--output",
            ])
            .arg(partial.as_os_str())
            .arg("--")
            .arg(url)
            .status()
            .context("Start curl. Install curl or supply verified files with --source-dir.")?;
        if !status.success() {
            eprintln!("Download failed with {status}.");
            continue;
        }
        if let Err(error) = verify(&partial, digest) {
            eprintln!("{error:#}");
            continue;
        }
        match partial.persist_noclobber(path) {
            Ok(_) => return Ok(()),
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
                return verify(path, digest);
            }
            Err(error) => return Err(error.into()),
        }
    }
    bail!(
        "Could not obtain a verified copy of {}. See the model guide for --source-dir.",
        path.display()
    )
}

fn verify(path: &Path, expected: &str) -> Result<()> {
    let mut file = File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let mut sha = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        sha.update(&buffer[..count]);
    }
    let actual: String = sha
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    ensure!(
        actual.eq_ignore_ascii_case(expected),
        "SHA-256 mismatch for {}. Check the source file before retrying.",
        path.display()
    );
    Ok(())
}

fn convert(source: &Path, stage: &Path) -> Result<()> {
    convert_checkpoint(
        &source.join("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        &stage.join("models/gpt-model.safetensors"),
        ModelKind::Gpt,
    )?;
    convert_checkpoint(
        &source.join("gsv-v2final-pretrained/s2G2333k.pth"),
        &stage.join("models/sovits-model.safetensors"),
        ModelKind::Sovits,
    )?;
    convert_bert(
        &source.join("chinese-roberta-wwm-ext-large/pytorch_model.bin"),
        &stage.join("models/bert/bert.safetensors"),
    )?;
    convert_hubert(
        &source.join("chinese-hubert-base/pytorch_model.bin"),
        &stage.join("models/hubert/hubert.safetensors"),
    )?;
    fs::copy(
        source.join("chinese-roberta-wwm-ext-large/tokenizer.json"),
        stage.join("models/bert/tokenizer.json"),
    )?;
    fs::create_dir_all(stage.join("voices/demo"))?;
    fs::copy(
        source.join("reference.wav"),
        stage.join("voices/demo/ref.wav"),
    )?;
    fs::write(stage.join("voices/demo/voice.json"), VOICE)?;
    Ok(())
}
