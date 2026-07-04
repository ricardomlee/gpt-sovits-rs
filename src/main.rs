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
use std::path::{Path, PathBuf};
use tracing::{error, info};

#[cfg(feature = "http-api")]
mod server;

#[derive(Parser, Debug)]
#[command(name = "gpt-sovits")]
#[command(author = "GPT-SoVITS Rust Contributors")]
#[command(version)]
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

    /// Check models, voice profile, device, and common setup issues without running inference
    #[arg(long)]
    doctor: bool,

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

    /// Inference mode (auto uses CUDA Graph on supported CUDA F32 models, otherwise KV)
    #[arg(long, value_parser = ["auto", "plain", "kv", "cuda-graph"])]
    mode: Option<String>,

    /// Split long text by sentence and concatenate audio chunks.
    #[arg(long, conflicts_with = "no_split_sentences")]
    split_sentences: bool,

    /// Disable the default Python-compatible punctuation splitting.
    #[arg(long)]
    no_split_sentences: bool,

    /// Text splitting policy: sentence is smoother; cut5 matches Python punctuation splitting.
    #[arg(long, value_parser = ["sentence", "cut5"])]
    split_method: Option<String>,

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
        let ok = run_doctor(&args);
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

fn run_doctor(args: &Args) -> bool {
    let mut report = DoctorReport::default();
    println!("GPT-SoVITS-RS doctor\n");

    check_device(&mut report, &args.device);
    check_models(&mut report, args);
    check_voices(&mut report, args);

    println!();
    if report.errors == 0 {
        println!(
            "Doctor finished: {} check(s), {} warning(s), no blocking errors.",
            report.checks, report.warnings
        );
        true
    } else {
        println!(
            "Doctor found {} blocking error(s) and {} warning(s) across {} check(s).",
            report.errors, report.warnings, report.checks
        );
        println!("Fix the errors above, then run `gpt-sovits --doctor` again.");
        false
    }
}

#[derive(Default)]
struct DoctorReport {
    checks: usize,
    warnings: usize,
    errors: usize,
}

impl DoctorReport {
    fn ok(&mut self, message: impl AsRef<str>) {
        self.checks += 1;
        println!("[ok] {}", message.as_ref());
    }

    fn warn(&mut self, message: impl AsRef<str>) {
        self.checks += 1;
        self.warnings += 1;
        println!("[warn] {}", message.as_ref());
    }

    fn error(&mut self, message: impl AsRef<str>) {
        self.checks += 1;
        self.errors += 1;
        println!("[error] {}", message.as_ref());
    }
}

fn check_device(report: &mut DoctorReport, requested: &str) {
    let config = Config::builder().with_device(requested).build();
    match requested {
        "cuda" => {
            #[cfg(feature = "cuda")]
            match candle_core::Device::new_cuda(0) {
                Ok(_) => report.ok("CUDA device is available"),
                Err(e) => report.error(format!(
                    "CUDA was requested, but Candle could not open CUDA device 0: {e}"
                )),
            }
            #[cfg(not(feature = "cuda"))]
            report.error("CUDA was requested, but this binary was not built with --features cuda");
        }
        "mps" => match candle_core::Device::new_metal(0) {
            Ok(_) => report.ok("Metal/MPS device is available"),
            Err(e) => report.error(format!(
                "MPS was requested, but Candle could not open Metal device 0: {e}"
            )),
        },
        "cpu" => report.ok("CPU device selected"),
        "auto" => report.ok(format!("auto device resolved to {}", config.device)),
        other => report.error(format!(
            "Unsupported device '{other}'; expected auto, cuda, cpu, or mps"
        )),
    }
}

fn check_models(report: &mut DoctorReport, args: &Args) {
    if args.models_dir.is_dir() {
        report.ok(format!(
            "models directory exists: {}",
            args.models_dir.display()
        ));
    } else {
        report.warn(format!(
            "models directory does not exist yet: {}",
            args.models_dir.display()
        ));
    }

    match ModelPaths::discover(
        &args.models_dir,
        ModelPathOverrides {
            gpt: args.gpt_model.clone(),
            sovits: args.sovits_model.clone(),
            bert: args.bert_model.clone(),
            hubert: args.hubert_model.clone(),
        },
    ) {
        Ok(paths) => {
            check_safetensors(report, "GPT", &paths.gpt, true);
            check_safetensors(report, "SoVITS", &paths.sovits, true);
            match paths.bert.as_ref() {
                Some(path) => {
                    check_safetensors(report, "BERT", path, false);
                    check_bert_tokenizer(report, path);
                }
                None => report.warn("BERT model not found; speech quality will be reduced"),
            }
            match paths.hubert.as_ref() {
                Some(path) => check_safetensors(report, "HuBERT", path, false),
                None => report.warn("HuBERT model not found; voice similarity will be reduced"),
            }
        }
        Err(e) => report.error(e),
    }
}

fn check_safetensors(report: &mut DoctorReport, label: &str, path: &Path, required: bool) {
    if !path.is_file() {
        let message = format!("{label} model file does not exist: {}", path.display());
        if required {
            report.error(message);
        } else {
            report.warn(message);
        }
        return;
    }

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("safetensors") => {}
        Some("ckpt" | "pth") => {
            report.error(format!(
                "{label} points to a raw PyTorch checkpoint, not a runtime model: {}. Convert it to safetensors first.",
                path.display()
            ));
            return;
        }
        Some(other) => {
            report.warn(format!(
                "{label} model extension is .{other}; expected .safetensors: {}",
                path.display()
            ));
        }
        None => report.warn(format!(
            "{label} model has no file extension; expected .safetensors: {}",
            path.display()
        )),
    }

    match read_safetensors_header_count(path) {
        Ok(count) => report.ok(format!(
            "{label} safetensors header is readable: {} ({count} tensor entries)",
            path.display()
        )),
        Err(e) => report.error(format!(
            "{label} safetensors header is not readable: {} ({e})",
            path.display()
        )),
    }
}

fn read_safetensors_header_count(path: &Path) -> Result<usize, String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| e.to_string())?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(format!("invalid safetensors header length: {header_len}"));
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    let object = value
        .as_object()
        .ok_or_else(|| "safetensors header is not a JSON object".to_string())?;
    Ok(object
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .count())
}

fn check_bert_tokenizer(report: &mut DoctorReport, bert_path: &Path) {
    let sibling = bert_path.with_file_name("tokenizer.json");
    let fallback = PathBuf::from("models/bert/tokenizer.json");
    if sibling.is_file() {
        report.ok(format!("BERT tokenizer found: {}", sibling.display()));
    } else if fallback.is_file() {
        report.ok(format!(
            "BERT tokenizer found through fallback: {}",
            fallback.display()
        ));
    } else {
        report.error(format!(
            "BERT tokenizer not found. Put tokenizer.json next to {} or at {}",
            bert_path.display(),
            fallback.display()
        ));
    }
}

fn check_voices(report: &mut DoctorReport, args: &Args) {
    match list_voice_profiles(&args.voices_dir) {
        Ok(voices) if voices.is_empty() => report.warn(format!(
            "no voice profiles found in {}; create voices/<name>/voice.json or pass --reference-audio and --reference-text",
            args.voices_dir.display()
        )),
        Ok(voices) => report.ok(format!(
            "found {} voice profile(s) in {}: {}",
            voices.len(),
            args.voices_dir.display(),
            voices.join(", ")
        )),
        Err(e) => report.error(e),
    }

    if let Some(name) = args.voice.as_deref() {
        match LoadedVoiceProfile::load(name, &args.voices_dir) {
            Ok(profile) => check_loaded_voice(report, &profile),
            Err(e) => report.error(e),
        }
    } else {
        if let Some(audio) = args.reference_audio.as_ref() {
            check_reference_audio(report, audio);
        }
        if let Some(text) = args.reference_text.as_deref() {
            check_reference_text(report, text);
        }
        if args.reference_audio.is_none() && args.reference_text.is_none() {
            report.warn("no --voice or inline reference audio/text selected for a test run");
        }
    }
}

fn check_loaded_voice(report: &mut DoctorReport, voice: &LoadedVoiceProfile) {
    report.ok(format!(
        "voice profile '{}' parsed: {}",
        voice.name,
        voice.dir.join("voice.json").display()
    ));

    match voice.reference_audio_path() {
        Some(path) => check_reference_audio(report, &path),
        None => report.error(format!(
            "voice '{}' does not set reference_audio",
            voice.name
        )),
    }
    match voice.reference_text() {
        Some(text) => check_reference_text(report, text),
        None => report.error(format!(
            "voice '{}' does not set reference_text",
            voice.name
        )),
    }
}

fn check_reference_audio(report: &mut DoctorReport, path: &Path) {
    if !path.is_file() {
        report.error(format!(
            "reference audio does not exist: {}",
            path.display()
        ));
        return;
    }
    if path.extension().and_then(|ext| ext.to_str()) != Some("wav") {
        report.warn(format!(
            "reference audio is not a .wav file; current CLI path expects WAV: {}",
            path.display()
        ));
    }
    match hound::WavReader::open(path) {
        Ok(reader) => {
            let spec = reader.spec();
            let duration = reader.duration() as f32 / spec.sample_rate as f32;
            let level = if (3.0..=10.0).contains(&duration) {
                "ok"
            } else {
                "warn"
            };
            let message = format!(
                "reference audio: {} ({:.2}s, {} Hz, {} channel(s))",
                path.display(),
                duration,
                spec.sample_rate,
                spec.channels
            );
            if level == "ok" {
                report.ok(message);
            } else {
                report.warn(format!("{message}; recommended reference length is 3-10s"));
            }
        }
        Err(e) => report.error(format!(
            "reference audio is not readable as WAV: {} ({e})",
            path.display()
        )),
    }
}

fn check_reference_text(report: &mut DoctorReport, text: &str) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        report.error("reference_text is empty");
    } else {
        report.ok(format!(
            "reference_text is present ({} chars)",
            trimmed.chars().count()
        ));
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
