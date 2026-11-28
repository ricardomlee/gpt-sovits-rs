use super::{ModelKind, convert_bert, convert_checkpoint, convert_hubert};
use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;
use std::process::Command;
use tempfile::NamedTempFile;

const DOWNLOADS: &str = include_str!("../../../examples/first-run/downloads.txt");
const VOICE: &str = include_str!("../../../examples/first-run/voice.json");
const MARKER: &str = ".gpt-sovits-demo";
const DEMO: &str = "gpt-sovits-public-v2-demo";
const OUTPUT_FILES: [&str; 7] = [
    "models/gpt-model.safetensors",
    "models/sovits-model.safetensors",
    "models/bert/bert.safetensors",
    "models/bert/tokenizer.json",
    "models/hubert/hubert.safetensors",
    "voices/demo/ref.wav",
    "voices/demo/voice.json",
];

pub(super) fn prepare(output: &Path, source: Option<&Path>, modelscope: bool) -> Result<()> {
    if output.try_exists()? {
        ensure!(
            is_prepared(output),
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
    let cache = parent.join(".gpt-sovits-demo-source");
    let source_dir = source.unwrap_or(&cache);
    for line in DOWNLOADS.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let digest = fields
            .next()
            .context("Missing SHA-256 in the demo manifest")?;
        let relative = fields
            .next()
            .context("Missing file path in the demo manifest")?;
        let primary = fields.next().context("Missing URL in the demo manifest")?;
        let mirror = fields.next();
        let mut urls = [Some(primary), mirror];
        if modelscope && mirror.is_some() {
            urls.swap(0, 1);
        }
        prepare_source(
            &source_dir.join(relative),
            digest,
            urls.into_iter().flatten(),
            source.is_none(),
        )?;
    }

    let staging = tempfile::Builder::new()
        .prefix(".prepare-demo-")
        .tempdir_in(parent)?;
    convert(source_dir, staging.path())?;
    fs::write(staging.path().join(MARKER), DEMO)?;
    if let Err(error) = fs::rename(staging.path(), output) {
        if !is_prepared(output) {
            return Err(error)
                .with_context(|| format!("Install the demo into {}", output.display()));
        }
    } else {
        let _ = staging.keep();
    }
    println!("Prepared the public v2 demo in {}.", output.display());
    Ok(())
}

fn is_prepared(directory: &Path) -> bool {
    fs::read_to_string(directory.join(MARKER)).is_ok_and(|marker| marker.trim() == DEMO)
        && OUTPUT_FILES
            .iter()
            .all(|path| directory.join(path).is_file())
}

fn prepare_source<'a>(
    path: &Path,
    digest: &str,
    urls: impl Iterator<Item = &'a str>,
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
    for url in urls {
        println!("Downloading {} from {url}", path.display());
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
        actual == expected,
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
    fs::write(stage.join("download-sources.txt"), DOWNLOADS)?;
    Ok(())
}
