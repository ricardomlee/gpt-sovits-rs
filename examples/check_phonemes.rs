use gpt_sovits_rs::text_frontend::TextFrontend;
use gpt_sovits_rs::Language;

fn main() {
    let frontend = TextFrontend::new().unwrap();
    
    let text = "你好世界";
    let ids = frontend.process(text, Language::Chinese).unwrap();
    let phonemes = frontend.get_phonemes(text, Language::Chinese).unwrap();
    
    println!("Input: {}", text);
    println!("Phonemes: {}", phonemes);
    println!("Phoneme IDs: {:?}", ids);
    println!("Phoneme count: {}", ids.len());
}
