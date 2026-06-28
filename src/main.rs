//! GPT-SoVITS CLI - Command line interface for TTS inference

use clap::Parser;
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::voice::{
    list_voice_profiles, load_optional_voice_profile, InferenceOptionOverrides, LoadedVoiceProfile,
    VoiceDefaults,
};
use gpt_sovits_rs::{split_sentences, AudioBuffer, Config, InferenceOptions, Language, Pipeline};
use std::path::PathBuf;
use tracing::{error, info};

#[cfg(feature = "http-api")]
mod server;

#[derive(Parser, Debug)]
#[command(name = "gpt-sovits")]
#[command(author = "GPT-SoVITS Rust Contributors")]
#[command(version = "0.1.0")]
#[command(about = "GPT-SoVITS TTS Inference Engine", long_about = None)]
struct Args {
    /// Input text for synthesis
    #[arg(short, long)]
    text: Option<String>,

    /// Voice profile name under --voices-dir, e.g. voices/mao/voice.json
    #[arg(long)]
    voice: Option<String>,

    /// Directory containing voice profiles
    #[arg(long, default_value = "voices")]
    voices_dir: PathBuf,

    /// List available voice profiles and exit
    #[arg(long)]
    list_voices: bool,

    /// Inspect model file
    #[arg(long)]
    inspect: Option<PathBuf>,

    /// Path to GPT model file
    #[arg(long)]
    gpt_model: Option<PathBuf>,

    /// Directory searched for models not passed explicitly
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,

    /// Path to SoVITS model file
    #[arg(long)]
    sovits_model: Option<PathBuf>,

    /// Path to BigVGAN model file (experimental; not used by the main SoVITS decoder path yet)
    #[arg(long)]
    bigvgan_model: Option<PathBuf>,

    /// Path to BERT safetensors model file (optional, improves quality)
    #[arg(long)]
    bert_model: Option<PathBuf>,

    /// Path to HuBERT/Wav2Vec2 safetensors model file (optional, improves quality)
    #[arg(long)]
    hubert_model: Option<PathBuf>,

    /// Reference audio path
    #[arg(long)]
    reference_audio: Option<PathBuf>,

    /// Reference audio text
    #[arg(long)]
    reference_text: Option<String>,

    /// Language of reference audio
    #[arg(long)]
    language: Option<String>,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Top-k sampling
    #[arg(long)]
    top_k: Option<usize>,

    /// Top-p sampling
    #[arg(long)]
    top_p: Option<f32>,

    /// Sampling temperature
    #[arg(long)]
    temperature: Option<f32>,

    /// Speed multiplier
    #[arg(long)]
    speed: Option<f32>,

    /// Maximum semantic tokens to generate. Use higher values for long sentences.
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Repetition penalty applied during GPT sampling.
    #[arg(long)]
    repetition_penalty: Option<f32>,

    /// Inference mode
    #[arg(long, value_parser = ["plain", "kv", "cuda-graph"])]
    mode: Option<String>,

    /// Split long text by sentence and concatenate audio chunks.
    #[arg(long)]
    split_sentences: bool,

    /// Minimum characters per sentence chunk when --split-sentences is enabled.
    #[arg(long)]
    min_sentence_chars: Option<usize>,

    /// Silence inserted between sentence chunks.
    #[arg(long)]
    sentence_gap_ms: Option<u32>,

    /// Fade in/out each sentence chunk before concatenation.
    #[arg(long)]
    sentence_fade_ms: Option<u32>,

    /// Request half precision (SoVITS currently falls back to F32 for audio quality)
    #[arg(long)]
    half: bool,

    /// Device to use
    #[arg(long, default_value = "auto", value_parser = ["auto", "cuda", "cpu", "mps"])]
    device: String,

    /// Start HTTP server mode
    #[arg(long)]
    http: bool,

    /// HTTP server port
    #[arg(long, default_value = "9880")]
    port: u16,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    // Inspect mode
    if let Some(ref model_path) = args.inspect {
        inspect_model(model_path);
        return;
    }

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    std::env::set_var("RUST_LOG", log_level);
    tracing_subscriber::fmt::init();

    info!("Starting GPT-SoVITS TTS Engine");

    if args.list_voices {
        match list_voice_profiles(&args.voices_dir) {
            Ok(voices) if voices.is_empty() => {
                println!("No voices found in {}", args.voices_dir.display())
            }
            Ok(voices) => {
                for voice in voices {
                    println!("{voice}");
                }
            }
            Err(e) => {
                error!("{}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    let voice_profile = match load_optional_voice_profile(args.voice.as_deref(), &args.voices_dir) {
        Ok(profile) => profile,
        Err(e) => {
            error!("{}", e);
            std::process::exit(1);
        }
    };
    if let Some(voice) = voice_profile.as_ref() {
        info!(
            "Loaded voice profile '{}' from {:?}",
            voice.name,
            voice.dir.join("voice.json")
        );
    }

    if !args.http && args.text.is_none() {
        eprintln!("Error: --text is required in CLI mode");
        eprintln!("Usage: gpt-sovits --voice <VOICE> --text <TEXT> [OPTIONS]");
        eprintln!("       gpt-sovits --http [OPTIONS]");
        std::process::exit(1);
    }
    if let Some(language) = args.language.as_deref() {
        if Language::from_str(language).is_none() {
            error!(
                "Unsupported language '{}'; expected zh, en, ja, ko, yue, or auto",
                language
            );
            std::process::exit(1);
        }
    }

    let model_paths = match ModelPaths::discover(
        &args.models_dir,
        ModelPathOverrides {
            gpt: args.gpt_model.clone(),
            sovits: args.sovits_model.clone(),
            bert: args.bert_model.clone(),
            hubert: args.hubert_model.clone(),
        },
    ) {
        Ok(paths) => paths,
        Err(e) => {
            error!("{}", e);
            std::process::exit(1);
        }
    };
    if model_paths.bert.is_none() {
        tracing::warn!("BERT model not found; speech quality will be reduced");
    }
    if model_paths.hubert.is_none() {
        tracing::warn!("HuBERT model not found; voice similarity will be reduced");
    }

    // HTTP mode
    if args.http {
        #[cfg(feature = "http-api")]
        {
            if let Err(e) = server::run(
                args.port,
                &args.device,
                args.half,
                Some(&model_paths.gpt),
                Some(&model_paths.sovits),
                args.bigvgan_model.as_deref(),
                model_paths.bert.as_deref(),
                model_paths.hubert.as_deref(),
                &args.voices_dir,
            ) {
                error!("HTTP server error: {}", e);
                std::process::exit(1);
            }
        }
        #[cfg(not(feature = "http-api"))]
        {
            error!("HTTP API feature is not enabled. Build with --features http-api");
        }
        return;
    }

    // CLI mode - validate required arguments
    let text = args.text.clone().expect("text was validated above");

    let gpt_model = model_paths.gpt;
    let sovits_model = model_paths.sovits;

    let reference_audio = match resolve_reference_audio(&args, voice_profile.as_ref()) {
        Some(a) => a,
        None => {
            eprintln!("Error: --reference-audio is required in CLI mode unless --voice provides reference_audio");
            std::process::exit(1);
        }
    };

    let reference_text = match resolve_reference_text(&args, voice_profile.as_ref()) {
        Some(t) => t,
        None => {
            eprintln!("Error: --reference-text is required in CLI mode unless --voice provides reference_text");
            std::process::exit(1);
        }
    };

    let output = args.output;

    info!("Loading models...");
    info!("  GPT model: {:?}", gpt_model);
    info!("  SoVITS model: {:?}", sovits_model);

    // Initialize configuration
    let config = Config::builder()
        .with_half_precision(args.half)
        .with_device(&args.device)
        .build();

    // Create pipeline
    let mut pipeline = match Pipeline::new(config) {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to initialize pipeline: {}", e);
            std::process::exit(1);
        }
    };

    // Load models
    info!("Loading GPT model...");
    if let Err(e) = pipeline.load_gpt(&gpt_model) {
        error!("Failed to load GPT model: {}", e);
        std::process::exit(1);
    }

    info!("Loading SoVITS model...");
    if let Err(e) = pipeline.load_sovits(&sovits_model) {
        error!("Failed to load SoVITS model: {}", e);
        std::process::exit(1);
    }

    // BigVGAN loading is experimental. The current SoVITS synthesis path still uses
    // the decoder embedded in the SoVITS weights.
    if let Some(ref bigvgan_path) = args.bigvgan_model {
        info!("Loading BigVGAN model (experimental; not used by main synthesis path yet)...");
        if let Err(e) = pipeline.load_bigvgan(bigvgan_path) {
            error!("Failed to load BigVGAN model: {}", e);
            std::process::exit(1);
        }
    } else {
        info!("BigVGAN model not specified; using SoVITS decoder");
    }

    // Load BERT model (optional, significantly improves quality)
    if let Some(ref bert_path) = model_paths.bert {
        info!("Loading BERT model...");
        if let Err(e) = pipeline.load_bert(bert_path) {
            error!("Failed to load BERT model: {}", e);
        }
    } else {
        info!("BERT model not specified, skipping (quality may be reduced)");
    }

    // Load Hubert model (optional, needed for semantic token extraction)
    if let Some(ref hubert_path) = model_paths.hubert {
        info!("Loading Hubert model...");
        if let Err(e) = pipeline.load_hubert(hubert_path) {
            error!("Failed to load Hubert model: {}", e);
        }
    } else {
        info!("Hubert model not specified, skipping (quality may be reduced)");
    }

    // Load semantic tokenizer (optional, uses SoVITS weights for prompt token extraction)
    if model_paths.hubert.is_some() {
        info!("Loading semantic tokenizer from SoVITS weights...");
        if let Err(e) = pipeline.load_semantic_tokenizer(&sovits_model) {
            error!("Failed to load semantic tokenizer: {}", e);
        }
    }

    // Parse language
    let voice_defaults = VoiceDefaults::from_profile(voice_profile.as_ref().map(|v| &v.profile));
    let language_text = args.language.as_deref().unwrap_or(&voice_defaults.language);
    let language = match Language::from_str(language_text) {
        Some(language) => language,
        None => {
            error!(
                "Unsupported language '{}'; expected zh, en, ja, ko, yue, or auto",
                language_text
            );
            std::process::exit(1);
        }
    };
    let mode = args
        .mode
        .clone()
        .unwrap_or_else(|| voice_defaults.mode.clone());
    let split_sentences = args.split_sentences || voice_defaults.split_sentences;
    let min_sentence_chars = args
        .min_sentence_chars
        .unwrap_or(voice_defaults.min_sentence_chars);
    let sentence_gap_ms = args
        .sentence_gap_ms
        .unwrap_or(voice_defaults.sentence_gap_ms);
    let sentence_fade_ms = args
        .sentence_fade_ms
        .unwrap_or(voice_defaults.sentence_fade_ms);

    // Create inference options
    let options = voice_defaults.to_inference_options(
        language,
        InferenceOptionOverrides {
            top_k: args.top_k,
            top_p: args.top_p,
            temperature: args.temperature,
            speed: args.speed,
            max_tokens: args.max_tokens,
            repetition_penalty: args.repetition_penalty,
        },
    );

    // Run inference
    info!("Running inference...");
    info!("  Text: {}", text);
    info!("  Reference: {:?}", reference_audio);
    info!("  Language: {:?}", language);

    let result = if split_sentences {
        run_split_inference(
            &mut pipeline,
            &text,
            &reference_audio,
            &reference_text,
            &options,
            &mode,
            min_sentence_chars,
            sentence_gap_ms,
            sentence_fade_ms,
        )
    } else {
        run_inference(
            &mut pipeline,
            &text,
            &reference_audio,
            &reference_text,
            &options,
            &mode,
        )
    };

    match result {
        Ok(audio) => {
            info!("Saving output to {:?}", output);
            if let Err(e) = audio.save(&output) {
                error!("Failed to save audio: {}", e);
                std::process::exit(1);
            }
            info!("Done! Output saved to {:?}", output);
        }
        Err(e) => {
            error!("Inference failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_inference(
    pipeline: &mut Pipeline,
    text: &str,
    reference_audio: &PathBuf,
    reference_text: &str,
    options: &InferenceOptions,
    mode: &str,
) -> gpt_sovits_rs::Result<AudioBuffer> {
    pipeline.inference_with_mode(mode, text, reference_audio, reference_text, options)
}

#[allow(clippy::too_many_arguments)]
fn run_split_inference(
    pipeline: &mut Pipeline,
    text: &str,
    reference_audio: &PathBuf,
    reference_text: &str,
    options: &InferenceOptions,
    mode: &str,
    min_sentence_chars: usize,
    gap_ms: u32,
    fade_ms: u32,
) -> gpt_sovits_rs::Result<AudioBuffer> {
    let chunks = split_sentences(text, min_sentence_chars);
    info!(
        "Split text into {} sentence chunk(s), mode={}, gap={}ms, fade={}ms",
        chunks.len(),
        mode,
        gap_ms,
        fade_ms
    );
    pipeline.preload_speaker(reference_audio, reference_text, options.language)?;

    let mut output: Option<AudioBuffer> = None;
    for (idx, chunk) in chunks.iter().enumerate() {
        info!("Synthesizing chunk {}/{}: {}", idx + 1, chunks.len(), chunk);
        let mut audio = run_inference(
            pipeline,
            chunk,
            reference_audio,
            reference_text,
            options,
            mode,
        )?;
        if fade_ms > 0 {
            audio.fade_in(fade_ms);
            audio.fade_out(fade_ms);
        }

        if let Some(out) = output.as_mut() {
            if gap_ms > 0 {
                let gap_samples = (gap_ms as f32 * out.sample_rate as f32 / 1000.0) as usize
                    * out.channels as usize;
                out.concat(&AudioBuffer::new(
                    vec![0.0; gap_samples],
                    out.sample_rate,
                    out.channels,
                ))?;
            }
            out.concat(&audio)?;
        } else {
            output = Some(audio);
        }
    }

    output.ok_or_else(|| {
        gpt_sovits_rs::Error::InferenceError("No sentence chunks generated".to_string())
    })
}

fn resolve_reference_audio(args: &Args, voice: Option<&LoadedVoiceProfile>) -> Option<PathBuf> {
    args.reference_audio
        .clone()
        .or_else(|| voice.and_then(|v| v.reference_audio_path()))
}

fn resolve_reference_text(args: &Args, voice: Option<&LoadedVoiceProfile>) -> Option<String> {
    args.reference_text
        .clone()
        .or_else(|| voice.and_then(|v| v.reference_text().map(str::to_string)))
}

/// Inspect model file
fn inspect_model(path: &PathBuf) {
    use safetensors::SafeTensors;
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();

    let st = SafeTensors::deserialize(&buffer).unwrap();
    let name = path.file_name().unwrap().to_str().unwrap();

    println!("{name} keys ({} total):", st.names().len());
    for name in st.names() {
        let tensor = st.tensor(name).unwrap();
        println!("  {name:60} {:?}", tensor.shape());
    }
}
