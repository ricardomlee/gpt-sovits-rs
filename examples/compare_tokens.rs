/// Compare Rust semantic tokens against Python-generated tokens for test_zh.wav
///
/// This verifies VQ prompt token extraction matches Python exactly.
/// Python always appends 0.3s silence (9600 samples at 32kHz, equivalent to 0.6s at 16kHz)
/// before HuBERT processing — our Rust code must match this behavior.
use gpt_sovits_rs::models::{HubertModel, SemanticTokenizer};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Python first 20 tokens for test_zh.wav (with silence padding):
    let py_tokens: Vec<usize> = vec![
        54, 234, 234, 405, 320, 807, 552, 369, 0, 21, 306, 434, 504, 320, 320, 59, 422, 173, 805,
        190,
    ];

    // Load HuBERT
    let mut hubert = HubertModel::load_with_device(
        "models/hubert/hubert.safetensors",
        &candle_core::Device::Cpu,
    )?;
    println!("[OK] HuBERT loaded");

    // Load semantic tokenizer
    let tokenizer = SemanticTokenizer::load_with_device(
        "models/sovits-model.safetensors",
        &candle_core::Device::Cpu,
    )?;
    println!("[OK] Tokenizer loaded");

    // Test 1: test_zh.wav (32kHz — gets resampled + silence padded)
    let ref_audio_32k = "/home/ric/gpt-sovits/test_zh.wav";
    if std::path::Path::new(ref_audio_32k).exists() {
        let hubert_feats = hubert.extract(ref_audio_32k)?;
        println!("\n[32kHz] HuBERT features: {:?}", hubert_feats.dims());
        let hubert_t = hubert_feats.transpose(1, 2)?;
        let tokens = tokenizer.extract(&hubert_t)?;
        println!("[32kHz] Tokens: {} total", tokens.len());
        println!("[32kHz] First 20: {:?}", &tokens[..tokens.len().min(20)]);
        let matches = tokens
            .iter()
            .zip(py_tokens.iter())
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "[32kHz] Match vs Python: {}/{}",
            matches,
            20.min(tokens.len())
        );
    }

    // Test 2: test_zh_py_wav16k.wav (already 16kHz — must also get silence padded)
    let ref_audio_16k = "/home/ric/gpt-sovits-rs/test_zh_py_wav16k.wav";
    if std::path::Path::new(ref_audio_16k).exists() {
        let hubert_feats = hubert.extract(ref_audio_16k)?;
        println!("\n[16kHz] HuBERT features: {:?}", hubert_feats.dims());
        let hubert_t = hubert_feats.transpose(1, 2)?;
        let tokens = tokenizer.extract(&hubert_t)?;
        println!("[16kHz] Tokens: {} total", tokens.len());
        println!("[16kHz] First 20: {:?}", &tokens[..tokens.len().min(20)]);
        let matches = tokens
            .iter()
            .zip(py_tokens.iter())
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "[16kHz] Match vs Python: {}/{}",
            matches,
            20.min(tokens.len())
        );
    }

    println!("\nPython first 20: {:?}", py_tokens);
    Ok(())
}
