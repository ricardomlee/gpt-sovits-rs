/// Test embedding lookup stack+reshape
use candle_core::{Device, DType, Tensor};
use gpt_sovits_rs::utils::load_safetensors;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let state_dict = load_safetensors("models/gpt-model.safetensors")?;

    let text_emb = state_dict.get("model.ar_text_embedding.word_embeddings.weight")
        .ok_or("not found")?.to_device(&device)?.to_dtype(DType::F32)?;

    // Phoneme IDs
    let phoneme_ids: Vec<usize> = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].to_vec();

    // Method 1: lookup_tokens style (stack + reshape)
    let mut embeddings = Vec::new();
    for &id in &phoneme_ids {
        embeddings.push(text_emb.get(id)?);
    }
    let batch = 1;
    let seq = phoneme_ids.len();
    let hidden = text_emb.dims()[1];
    let stacked = Tensor::stack(&embeddings, 0)?;
    println!("Stacked shape: {:?}", stacked.dims());
    let reshaped = stacked.reshape((batch, seq, hidden))?;
    println!("Reshaped shape: {:?}", reshaped.dims());
    let flat: Vec<f32> = reshaped.flatten_all()?.to_vec1()?;
    println!("Reshaped[0, 0, :5] = {:?}", &flat[..5]);

    // Method 2: Python equivalent (index each row, stack along dim 0)
    // In Python: text_emb[phoneme_ids] -> [seq, hidden]
    // Then unsqueeze to [1, seq, hidden]
    let rows: Vec<Tensor> = phoneme_ids.iter()
        .map(|&id| text_emb.narrow(0, id, 1))
        .collect::<Result<Vec<_>, _>>()?;
    let cat_rows = Tensor::cat(&rows, 0)?;
    println!("\nCat narrow shape: {:?}", cat_rows.dims());
    let unsqueezed = cat_rows.unsqueeze(0)?;
    println!("Unsqueezed shape: {:?}", unsqueezed.dims());
    let flat2: Vec<f32> = unsqueezed.flatten_all()?.to_vec1()?;
    println!("Unsqueezed[0, 0, :5] = {:?}", &flat2[..5]);

    // Compare
    let diff = (reshaped.clone() - unsqueezed.clone())?;
    let flat_diff: Vec<f32> = diff.flatten_all()?.to_vec1()?;
    let max_diff = flat_diff.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("\nMax diff between methods: {:.2e}", max_diff);

    // Python expected for phoneme_id 10
    println!("\nPython expected for text_emb[10]: [0.2464599609375, -0.60302734375, 0.5, -0.8837890625, 0.289794921875]");
    println!("Reshaped method: {:?}", &flat[..5]);
    println!("Cat+narrow method: {:?}", &flat2[..5]);

    // Also test with Tensor::from_vec of IDs (Candle's embedding_lookup style)
    let id_tensor = Tensor::new(phoneme_ids.iter().map(|&x| x as u32).collect::<Vec<_>>().as_slice(), &device)?;
    let emb_via_gather = text_emb.index_select(&id_tensor, 0)?;
    println!("\nindex_select shape: {:?}", emb_via_gather.dims());
    let flat3: Vec<f32> = emb_via_gather.flatten_all()?.to_vec1()?;
    println!("index_select[0, :5] = {:?}", &flat3[..5]);
    let diff2 = (reshaped.clone() - emb_via_gather.unsqueeze(0)?)?;
    let flat_diff2: Vec<f32> = diff2.flatten_all()?.to_vec1()?;
    let max_diff2 = flat_diff2.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("Max diff reshaped vs index_select: {:.2e}", max_diff2);

    Ok(())
}
