use candle_core::{DType, Device, Tensor};
use gpt_sovits_rs::model_paths::{ModelPathOverrides, ModelPaths};
use gpt_sovits_rs::models::{BertModel, GPTModel, HubertModel, SemanticTokenizer};
use gpt_sovits_rs::text_frontend::TextFrontend;
use gpt_sovits_rs::Language;
use std::path::Path;

const REF_AUDIO: &str = "mao.wav";
const REF_TEXT: &str = "会战兵力是八十万对六十万，优势在我。";
const TARGET_TEXT: &str = "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。";
const MAX_TOKENS: usize = 300;

fn first_difference(left: &[usize], right: &[usize]) -> Option<usize> {
    let common = left.len().min(right.len());
    (0..common)
        .find(|&index| left[index] != right[index])
        .or_else(|| (left.len() != right.len()).then_some(common))
}

fn print_comparison(name: &str, expected: &[usize], actual: &[usize]) -> bool {
    let Some(index) = first_difference(expected, actual) else {
        println!("{name}: exact match ({} tokens)", actual.len());
        return true;
    };

    let start = index.saturating_sub(4);
    let expected_end = (index + 5).min(expected.len());
    let actual_end = (index + 5).min(actual.len());
    println!(
        "{name}: first difference at token {index}, expected_len={}, actual_len={}",
        expected.len(),
        actual.len()
    );
    println!(
        "  expected[{start}..{expected_end}]={:?}",
        &expected[start..expected_end]
    );
    println!(
        "  actual  [{start}..{actual_end}]={:?}",
        &actual[start..actual_end]
    );
    false
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

    println!("Loading GPT and prompt models...");
    let gpt = GPTModel::load_with_device(paths.gpt.to_str().unwrap(), &device, DType::F32)?;
    let mut hubert = HubertModel::load_with_device(hubert_path.to_str().unwrap(), &device)?;
    let mut bert = BertModel::load_with_device(bert_path.to_str().unwrap(), &device)?;
    let tokenizer = SemanticTokenizer::load_with_device(paths.sovits.to_str().unwrap(), &device)?;
    let max_tokens = std::env::var("GPT_SOVITS_TEST_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(MAX_TOKENS);

    let features = hubert.extract(REF_AUDIO)?;
    let prompt_tokens = tokenizer.extract(&features.transpose(1, 2)?)?;
    let frontend = TextFrontend::new()?;
    let (ref_ids, ref_word2ph) = frontend.process_with_word2ph(REF_TEXT, Language::Chinese)?;
    let (target_ids, target_word2ph) =
        frontend.process_with_word2ph(TARGET_TEXT, Language::Chinese)?;
    let ref_bert = bert.extract(REF_TEXT)?;
    let ref_bert = gpt.project_and_align_bert(&ref_bert, &ref_word2ph, ref_ids.len())?;
    let target_bert = bert.extract(TARGET_TEXT)?;
    let target_bert =
        gpt.project_and_align_bert(&target_bert, &target_word2ph, target_ids.len())?;
    let combined_bert = Tensor::cat(&[&ref_bert, &target_bert], 1)?;
    let mut phoneme_ids = ref_ids;
    phoneme_ids.extend(target_ids);
    let max_kv_len = phoneme_ids.len() + prompt_tokens.len() + max_tokens + 32;

    println!(
        "phones={} prompt={} max_kv_len={max_kv_len}",
        phoneme_ids.len(),
        prompt_tokens.len()
    );

    let dynamic = gpt.generate_with_prompts_aligned_bert_kv_cache(
        &phoneme_ids,
        &prompt_tokens,
        Some(&combined_bert),
        1,
        1.0,
        1.0,
        1.35,
        max_tokens,
    )?;
    let bounded = gpt.generate_with_static_kv(
        &phoneme_ids,
        &prompt_tokens,
        Some(&combined_bert),
        1,
        1.0,
        1.0,
        1.35,
        max_tokens,
        max_kv_len,
    )?;
    let graph = gpt.generate_with_cuda_graph(
        &phoneme_ids,
        &prompt_tokens,
        Some(&combined_bert),
        1,
        1.0,
        1.0,
        1.35,
        max_tokens,
        max_kv_len,
    )?;

    let bounded_matches = print_comparison("bounded KV", &dynamic, &bounded);
    let graph_matches = print_comparison("CUDA Graph", &dynamic, &graph);
    if !bounded_matches || !graph_matches {
        std::process::exit(1);
    }
    Ok(())
}
