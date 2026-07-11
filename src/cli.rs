//! GPT-SoVITS CLI - Command line interface for TTS inference

use clap::Parser;
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::voice::{
    list_voice_profiles, load_optional_voice_profile, InferenceOptionOverrides, LoadedVoiceProfile,
    VoiceDefaults,
};
use gpt_sovits_rs::{
    split_cut5_for_language, split_sentences_for_language, AudioBuffer, Config, InferenceOptions,
    Language, Pipeline, SplitMethod,
};
use std::path::PathBuf;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "gpt-sovits")]
#[command(author = "GPT-SoVITS Rust Contributors")]
#[command(version)]
#[command(about = "GPT-SoVITS TTS Inference Engine", long_about = None)]
pub(crate) struct Args {
    /// Input text for synthesis
    #[arg(short, long)]
    pub(crate) text: Option<String>,

    /// Voice profile name under --voices-dir, e.g. voices/mao/voice.json
    #[arg(long)]
    pub(crate) voice: Option<String>,

    /// Directory containing voice profiles
    #[arg(long, default_value = "voices")]
    pub(crate) voices_dir: PathBuf,

    /// List available voice profiles and exit
    #[arg(long)]
    pub(crate) list_voices: bool,

    /// Check models, voice profile, device, and common setup issues without running inference
    #[arg(long)]
    pub(crate) doctor: bool,

    /// Inspect model file
    #[arg(long)]
    pub(crate) inspect: Option<PathBuf>,

    /// Path to GPT model file
    #[arg(long)]
    pub(crate) gpt_model: Option<PathBuf>,

    /// Directory searched for models not passed explicitly
    #[arg(long, default_value = "models")]
    pub(crate) models_dir: PathBuf,

    /// Path to SoVITS model file
    #[arg(long)]
    pub(crate) sovits_model: Option<PathBuf>,

    /// Path to BigVGAN model file (experimental; not used by the main SoVITS decoder path yet)
    #[arg(long)]
    pub(crate) bigvgan_model: Option<PathBuf>,

    /// Path to BERT safetensors model file (optional, improves quality)
    #[arg(long)]
    pub(crate) bert_model: Option<PathBuf>,

    /// Path to HuBERT/Wav2Vec2 safetensors model file (optional, improves quality)
    #[arg(long)]
    pub(crate) hubert_model: Option<PathBuf>,

    /// Reference audio path
    #[arg(long)]
    pub(crate) reference_audio: Option<PathBuf>,

    /// Reference audio text
    #[arg(long)]
    pub(crate) reference_text: Option<String>,

    /// v2Pro speaker-verification embedding safetensors file
    #[arg(long)]
    pub(crate) sv_embedding: Option<PathBuf>,

    /// Language of reference audio
    #[arg(long)]
    pub(crate) language: Option<String>,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    pub(crate) output: PathBuf,

    /// Top-k sampling
    #[arg(long)]
    pub(crate) top_k: Option<usize>,

    /// Top-p sampling
    #[arg(long)]
    pub(crate) top_p: Option<f32>,

    /// Sampling temperature
    #[arg(long)]
    pub(crate) temperature: Option<f32>,

    /// Speed multiplier
    #[arg(long)]
    pub(crate) speed: Option<f32>,

    /// Maximum semantic tokens to generate. Use higher values for long sentences.
    #[arg(long)]
    pub(crate) max_tokens: Option<usize>,

    /// Repetition penalty applied during GPT sampling.
    #[arg(long)]
    pub(crate) repetition_penalty: Option<f32>,

    /// Inference mode (auto uses CUDA Graph on supported CUDA F32 models, otherwise KV)
    #[arg(long, value_parser = ["auto", "plain", "kv", "cuda-graph"])]
    pub(crate) mode: Option<String>,

    /// Split long text by sentence and concatenate audio chunks.
    #[arg(long, conflicts_with = "no_split_sentences")]
    pub(crate) split_sentences: bool,

    /// Disable the default Python-compatible punctuation splitting.
    #[arg(long)]
    pub(crate) no_split_sentences: bool,

    /// Text splitting policy: sentence is smoother; cut5 matches Python punctuation splitting.
    #[arg(long, value_parser = ["sentence", "cut5"])]
    pub(crate) split_method: Option<String>,

    /// Minimum characters per sentence chunk when --split-sentences is enabled.
    #[arg(long)]
    pub(crate) min_sentence_chars: Option<usize>,

    /// Silence inserted between sentence chunks.
    #[arg(long)]
    pub(crate) sentence_gap_ms: Option<u32>,

    /// Fade in/out each sentence chunk before concatenation.
    #[arg(long)]
    pub(crate) sentence_fade_ms: Option<u32>,

    /// Request half precision (SoVITS currently falls back to F32 for audio quality)
    #[arg(long)]
    pub(crate) half: bool,

    /// Device to use
    #[arg(long, default_value = "auto", value_parser = ["auto", "cuda", "cpu", "mps"])]
    pub(crate) device: String,

    /// Start HTTP server mode
    #[arg(long)]
    pub(crate) http: bool,

    /// HTTP server port
    #[arg(long, default_value = "9880")]
    pub(crate) port: u16,

    /// Maximum number of GPT/SoVITS model pipelines kept in memory by the HTTP server
    #[arg(long, default_value_t = 2, value_parser = parse_positive_usize)]
    pub(crate) max_cached_pipelines: usize,

    /// Allow HTTP requests to reference audio/SV files outside --voices-dir
    #[arg(long)]
    pub(crate) allow_external_reference_paths: bool,

    /// Maximum Unicode characters accepted by one HTTP synthesis item
    #[arg(long, default_value_t = 10_000, value_parser = parse_positive_usize)]
    pub(crate) max_text_chars: usize,

    /// Maximum number of items accepted by one HTTP batch request
    #[arg(long, default_value_t = 64, value_parser = parse_positive_usize)]
    pub(crate) max_batch_items: usize,

    /// Maximum seconds an HTTP request may wait for the serialized inference slot
    #[arg(long, default_value_t = 120, value_parser = parse_positive_usize)]
    pub(crate) queue_timeout_secs: usize,

    /// Comma-separated voice profiles to warm before the HTTP server becomes ready
    #[arg(long, value_delimiter = ',')]
    pub(crate) preload_voices: Vec<String>,

    /// Verbose output
    #[arg(short, long)]
    pub(crate) verbose: bool,
}

pub(crate) fn run() {
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

    if args.doctor {
        let ok = crate::doctor::run_doctor(&args);
        std::process::exit(if ok { 0 } else { 1 });
    }

    info!("Starting GPT-SoVITS TTS Engine");

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
        if Language::parse(language).is_none() {
            error!(
                "Unsupported language '{}'; expected zh, en, ja, ko, yue, or auto",
                language
            );
            std::process::exit(1);
        }
    }

    let voice_model_paths = voice_profile
        .as_ref()
        .map(|voice| voice.model_paths(&args.models_dir))
        .unwrap_or_default();
    let bigvgan_model = args
        .bigvgan_model
        .clone()
        .or(voice_model_paths.bigvgan.clone());
    if let Some(path) = bigvgan_model.as_ref() {
        if !path.is_file() {
            error!("BigVGAN model file does not exist: {}", path.display());
            std::process::exit(1);
        }
    }

    let model_paths = match ModelPaths::discover(
        &args.models_dir,
        ModelPathOverrides {
            gpt: args.gpt_model.clone().or(voice_model_paths.gpt),
            sovits: args.sovits_model.clone().or(voice_model_paths.sovits),
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
            if let Err(e) = crate::server::run(
                args.port,
                &args.device,
                args.half,
                Some(&model_paths.gpt),
                Some(&model_paths.sovits),
                bigvgan_model.as_deref(),
                model_paths.bert.as_deref(),
                model_paths.hubert.as_deref(),
                args.max_cached_pipelines,
                args.allow_external_reference_paths,
                args.max_text_chars,
                args.max_batch_items,
                args.queue_timeout_secs,
                &args.preload_voices,
                &args.models_dir,
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
    let sv_embedding = resolve_sv_embedding(&args, voice_profile.as_ref());
    if let Some(path) = sv_embedding.as_ref() {
        if !path.is_file() {
            eprintln!(
                "Error: --sv-embedding file does not exist: {}",
                path.display()
            );
            std::process::exit(1);
        }
    }

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
    if let Some(ref bigvgan_path) = bigvgan_model {
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

    // Parse language
    let voice_defaults = VoiceDefaults::from_profile(voice_profile.as_ref().map(|v| &v.profile));
    let language_text = args.language.as_deref().unwrap_or(&voice_defaults.language);
    let language = match Language::parse(language_text) {
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
    let split_sentences = if args.no_split_sentences {
        false
    } else {
        args.split_sentences || voice_defaults.split_sentences
    };
    let min_sentence_chars = args
        .min_sentence_chars
        .unwrap_or(voice_defaults.min_sentence_chars);
    let sentence_gap_ms = args
        .sentence_gap_ms
        .unwrap_or(voice_defaults.sentence_gap_ms);
    let sentence_fade_ms = args
        .sentence_fade_ms
        .unwrap_or(voice_defaults.sentence_fade_ms);
    let split_method = args
        .split_method
        .as_deref()
        .and_then(SplitMethod::parse)
        .unwrap_or(voice_defaults.split_method);

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
    let options = if let Some(path) = sv_embedding {
        InferenceOptions {
            sv_embedding: Some(path),
            ..options
        }
    } else {
        options
    };

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
            split_method,
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
    split_method: SplitMethod,
) -> gpt_sovits_rs::Result<AudioBuffer> {
    let chunks = match split_method {
        SplitMethod::Sentence => {
            split_sentences_for_language(text, min_sentence_chars, options.language)
        }
        SplitMethod::Cut5 => split_cut5_for_language(text, min_sentence_chars, options.language),
    };
    info!(
        "Split text into {} chunk(s), method={:?}, mode={}, gap={}ms, fade={}ms",
        chunks.len(),
        split_method,
        mode,
        gap_ms,
        fade_ms
    );
    pipeline.inference_split_with_method(
        text,
        reference_audio,
        reference_text,
        options,
        mode,
        min_sentence_chars,
        gap_ms,
        fade_ms,
        split_method,
    )
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

fn resolve_sv_embedding(args: &Args, voice: Option<&LoadedVoiceProfile>) -> Option<PathBuf> {
    args.sv_embedding
        .clone()
        .or_else(|| voice.and_then(|v| v.sv_embedding_path()))
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got '{value}'"))?;
    if value == 0 {
        Err("value must be at least 1".to_string())
    } else {
        Ok(value)
    }
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
