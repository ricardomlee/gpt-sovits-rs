use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::models::{BertModel, GPTModel, HubertModel, SemanticTokenizer};
use gpt_sovits_rs::text_frontend::TextFrontend;
use gpt_sovits_rs::Language;
use std::path::Path;
use std::time::{Duration, Instant};

const REF_AUDIO: &str = "mao.wav";
const REF_TEXT: &str = "会战兵力是八十万对六十万，优势在我。";
const TARGET_TEXT: &str = "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。";
const MAX_TOKENS: usize = 300;

struct Inputs {
    phoneme_ids: Vec<usize>,
    prompt_tokens: Vec<usize>,
    ref_word2ph: Vec<usize>,
    target_word2ph: Vec<usize>,
    ref_phone_count: usize,
    target_phone_count: usize,
    ref_bert: Tensor,
    target_bert: Tensor,
}

fn first_difference(left: &[usize], right: &[usize]) -> Option<usize> {
    let common = left.len().min(right.len());
    (0..common)
        .find(|&index| left[index] != right[index])
        .or_else(|| (left.len() != right.len()).then_some(common))
}

fn run_gpt(
    path: &Path,
    device: &Device,
    dtype: DType,
    inputs: &Inputs,
) -> Result<(Vec<usize>, Duration), Box<dyn std::error::Error>> {
    let gpt = GPTModel::load_with_device(path.to_str().unwrap(), device, dtype)?;
    let ref_aligned = gpt.project_and_align_bert(
        &inputs.ref_bert,
        &inputs.ref_word2ph,
        inputs.ref_phone_count,
    )?;
    let target_aligned = gpt.project_and_align_bert(
        &inputs.target_bert,
        &inputs.target_word2ph,
        inputs.target_phone_count,
    )?;
    let bert = Tensor::cat(&[&ref_aligned, &target_aligned], 1)?;
    let mut tokens = Vec::new();
    let mut elapsed = Duration::ZERO;
    for iteration in 0..4 {
        device.synchronize()?;
        let start = Instant::now();
        let current = gpt.generate_with_prompts_aligned_bert_kv_cache(
            &inputs.phoneme_ids,
            &inputs.prompt_tokens,
            Some(&bert),
            1,
            1.0,
            1.0,
            1.35,
            MAX_TOKENS,
        )?;
        device.synchronize()?;
        if iteration > 0 {
            elapsed += start.elapsed();
            tokens = current;
        }
    }
    Ok((tokens, elapsed / 3))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("gpt_sovits_rs=info")
        .init();
    let device = Device::new_cuda_with_stream(0)?;
    if let Device::Cuda(cuda) = &device {
        unsafe { cuda.disable_event_tracking() };
    }
    let paths = ModelPaths::discover(Path::new("models"), ModelPathOverrides::default())?;
    let hubert_path = paths.hubert.as_ref().ok_or("HuBERT model not found")?;
    let bert_path = paths.bert.as_ref().ok_or("BERT model not found")?;
    let mut hubert = HubertModel::load_with_device(hubert_path.to_str().unwrap(), &device)?;
    let mut bert = BertModel::load_with_device(bert_path.to_str().unwrap(), &device)?;
    let tokenizer = SemanticTokenizer::load_with_device(paths.sovits.to_str().unwrap(), &device)?;

    let features = hubert.extract(REF_AUDIO)?;
    let prompt_tokens = tokenizer.extract(&features.transpose(1, 2)?)?;
    let frontend = TextFrontend::new()?;
    let (mut ref_ids, ref_word2ph) = frontend.process_with_word2ph(REF_TEXT, Language::Chinese)?;
    let ref_phone_count = ref_ids.len();
    let (target_ids, target_word2ph) =
        frontend.process_with_word2ph(TARGET_TEXT, Language::Chinese)?;
    let target_phone_count = target_ids.len();
    ref_ids.extend(target_ids);
    let inputs = Inputs {
        phoneme_ids: ref_ids,
        prompt_tokens,
        ref_word2ph,
        target_word2ph,
        ref_phone_count,
        target_phone_count,
        ref_bert: bert.extract(REF_TEXT)?,
        target_bert: bert.extract(TARGET_TEXT)?,
    };

    let (f32_tokens, f32_time) = run_gpt(&paths.gpt, &device, DType::F32, &inputs)?;
    let (bf16_tokens, bf16_time) = run_gpt(&paths.gpt, &device, DType::BF16, &inputs)?;
    let (f16_tokens, f16_time) = run_gpt(&paths.gpt, &device, DType::F16, &inputs)?;
    println!(
        "F32:  {:>4} tokens, {:.3}s\nBF16: {:>4} tokens, {:.3}s, speedup {:.2}x\nF16:  {:>4} tokens, {:.3}s, speedup {:.2}x",
        f32_tokens.len(),
        f32_time.as_secs_f64(),
        bf16_tokens.len(),
        bf16_time.as_secs_f64(),
        f32_time.as_secs_f64() / bf16_time.as_secs_f64(),
        f16_tokens.len(),
        f16_time.as_secs_f64(),
        f32_time.as_secs_f64() / f16_time.as_secs_f64(),
    );
    match first_difference(&f32_tokens, &bf16_tokens) {
        None => println!("BF16 tokens: exact match"),
        Some(index) => println!(
            "BF16 tokens: first difference at {index}, F32={:?}, BF16={:?}",
            f32_tokens.get(index),
            bf16_tokens.get(index)
        ),
    }
    match first_difference(&f32_tokens, &f16_tokens) {
        None => println!("F16 tokens: exact match"),
        Some(index) => println!(
            "F16 tokens: first difference at {index}, F32={:?}, F16={:?}",
            f32_tokens.get(index),
            f16_tokens.get(index)
        ),
    }
    Ok(())
}
