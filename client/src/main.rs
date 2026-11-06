use std::io::{self, IsTerminal, Read, Seek};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use tempfile::NamedTempFile;
use url::Url;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Synthesize speech through a running GPT-SoVITS HTTP service"
)]
struct Args {
    /// Service base URL (without /tts or /v1)
    #[arg(long, default_value = "http://127.0.0.1:9880", value_parser = parse_url)]
    url: Url,

    /// Voice name from GET /voices; inference settings come from the server
    #[arg(long)]
    voice: String,

    /// Text to synthesize; reads UTF-8 from stdin when omitted
    #[arg(long)]
    text: Option<String>,

    /// Output WAV file; writes WAV bytes to stdout when omitted
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// HTTP timeout in seconds, including queueing and audio download
    #[arg(long, default_value_t = 120, value_parser = clap::value_parser!(u64).range(1..))]
    timeout: u64,
}

fn parse_url(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|e| e.to_string())?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("expected an http:// or https:// service URL".into());
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err("service URL must not contain a query or fragment".into());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("service URL must not contain credentials".into());
    }
    Ok(url)
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    ensure!(!args.voice.trim().is_empty(), "--voice must not be empty");
    let text = match args.text {
        Some(text) => text,
        None => {
            ensure!(
                !io::stdin().is_terminal(),
                "provide --text or pipe UTF-8 text to stdin"
            );
            let mut text = String::new();
            io::stdin()
                .read_to_string(&mut text)
                .context("read UTF-8 text from stdin")?;
            text
        }
    };
    ensure!(!text.trim().is_empty(), "text must not be empty");
    ensure!(
        args.output.is_some() || !io::stdout().is_terminal(),
        "provide --output or redirect WAV output to a file or pipe"
    );

    let endpoint = format!("{}/tts", args.url.as_str().trim_end_matches('/'));
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(args.timeout))
        .redirects(0)
        .build();
    let response = match agent
        .post(&endpoint)
        .set("Accept", "audio/wav")
        .send_json(serde_json::json!({"voice": args.voice, "text": text.trim()}))
    {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            let mut detail = Vec::new();
            response
                .into_reader()
                .take(8192)
                .read_to_end(&mut detail)
                .ok();
            bail!(
                "TTS returned HTTP {status}: {}",
                String::from_utf8_lossy(&detail).trim()
            );
        }
        Err(error) => return Err(error).context("TTS request failed"),
    };
    ensure!(
        response.status() == 200,
        "TTS returned HTTP {}",
        response.status()
    );
    let mime = response
        .header("Content-Type")
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();
    ensure!(
        mime.eq_ignore_ascii_case("audio/wav") || mime.eq_ignore_ascii_case("audio/x-wav"),
        "expected audio/wav, received Content-Type: {mime:?}"
    );

    // Keep incomplete downloads separate from the destination, even when replacing a file.
    let mut wav = match args.output.as_ref() {
        Some(path) => NamedTempFile::new_in(
            path.parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        ),
        None => NamedTempFile::new(),
    }
    .context("create temporary WAV file")?;
    io::copy(&mut response.into_reader(), &mut wav).context("download WAV")?;
    wav.rewind()?;
    validate_wav(&mut wav)?;

    match args.output {
        Some(path) => {
            wav.persist(&path)
                .with_context(|| format!("save WAV to {}", path.display()))?;
        }
        None => {
            wav.rewind()?;
            io::copy(&mut wav, &mut io::stdout().lock()).context("write WAV to stdout")?;
        }
    }
    Ok(())
}

fn validate_wav(wav: &mut NamedTempFile) -> Result<()> {
    let mut reader =
        hound::WavReader::new(io::BufReader::new(wav)).context("invalid WAV response")?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.sample_rate == 32_000
            && spec.bits_per_sample == 16
            && spec.sample_format == hound::SampleFormat::Int,
        "expected 32 kHz mono PCM16 WAV"
    );
    ensure!(reader.duration() > 0, "empty WAV response");
    for sample in reader.samples::<i16>() {
        sample.context("incomplete WAV response")?;
    }
    Ok(())
}
