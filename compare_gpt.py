"""Compare GPT model forward pass between Python and Rust."""
import torch
import sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0, '/home/ric/gpt-sovits/GPT_SoVITS')
from AR.models.t2s_lightning_module import Text2SemanticLightningModule

torch.set_grad_enabled(False)

print("=== GPT Comparison: Python vs Rust ===\n")

# Load weights
tensors = {}
with safe_open('models/gpt-model.safetensors', framework='pt', device='cpu') as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key).float()

print(f"Loaded {len(tensors)} weights")
print(f"Text embedding: {tensors['model.ar_text_embedding.word_embeddings.weight'].shape}")
print(f"Audio embedding: {tensors['model.ar_audio_embedding.word_embeddings.weight'].shape}")

# Create config as dict (Python model expects dict access)
config = {
    "model": {
        "embedding_dim": 512,
        "hidden_dim": 512,
        "filter_channels": 2048,
        "head": 8,
        "n_heads": 8,
        "n_layer": 24,
        "n_layers": 24,
        "kernel_size": 3,
        "p_dropout": 0.0,
        "dropout": 0.0,
        "phoneme_vocab_size": 512,
        "vocab_size": 1025,
        "EOS": 1024,
    }
}

# Load model
print("\nLoading Python GPT model...")
model = Text2SemanticLightningModule(config, "****", is_train=False)

# Rename keys if needed - check what Python expects vs what we have
python_keys = set()
for name, param in model.named_parameters():
    python_keys.add(name)

missing_in_safetensors = python_keys - set(tensors.keys())
extra_in_safetensors = set(tensors.keys()) - python_keys

if missing_in_safetensors:
    print(f"Missing keys in safetensors ({len(missing_in_safetensors)}):")
    for k in sorted(missing_in_safetensors)[:5]:
        print(f"  {k}")
    print("  ...")

if extra_in_safetensors:
    print(f"Extra keys in safetensors ({len(extra_in_safetensors)}):")
    for k in sorted(extra_in_safetensors)[:5]:
        print(f"  {k}")
    print("  ...")

# Try loading with strict=False
model.load_state_dict(tensors, strict=False)
model.eval()
print("Model loaded successfully")

# Prepare test input
print("\nPreparing test input...")
phoneme_ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
x = torch.tensor([phoneme_ids], dtype=torch.long)
x_lens = torch.tensor([len(phoneme_ids)], dtype=torch.long)

# Dummy BERT features [1, 1024, seq_len]
bert_feature = torch.randn(1, 1024, len(phoneme_ids))

# Dummy prompt audio tokens
prompts = torch.tensor([[100, 200, 300, 400, 500]], dtype=torch.long)

print(f"  phoneme_ids: {x.shape}")
print(f"  x_lens: {x_lens.shape}")
print(f"  prompts: {prompts.shape}")
print(f"  bert_feature: {bert_feature.shape}")

# Run Python model with fixed seed
torch.manual_seed(42)
np.random.seed(42)

print("\nRunning Python GPT inference...")
pred_semantic, idx = model.model.infer_panel(
    x, x_lens, prompts, bert_feature,
    top_k=15, top_p=0.95, temperature=0.8
)
print(f"Output semantic tokens: {pred_semantic.shape}")
print(f"Tokens: {pred_semantic[0].tolist()}")

# Save inputs and outputs for Rust comparison
np.savetxt("gpt_py_phoneme_ids.txt", np.array(phoneme_ids), fmt='%d')
np.savetxt("gpt_py_prompts.txt", prompts.numpy().flatten(), fmt='%d')
np.savetxt("gpt_py_bert_feature.txt", bert_feature.numpy().flatten(), fmt='%.10f')
np.savetxt("gpt_py_output_tokens.txt", pred_semantic.numpy().flatten(), fmt='%d')

print(f"\nSaved gpt_py_phoneme_ids.txt, gpt_py_prompts.txt, gpt_py_bert_feature.txt, gpt_py_output_tokens.txt")

# Also save model outputs at each step for detailed comparison
# Run with seed=0 again to get same output
torch.manual_seed(42)
np.random.seed(42)

# Save the model's forward pass output
# Note: SinePositionalEmbedding.forward() already computes: x * x_scale + alpha * pe
# So we call it directly without adding x again
x_emb = model.model.ar_text_embedding(x)
x_emb = x_emb + model.model.bert_proj(bert_feature.transpose(1, 2))
x_emb = model.model.ar_text_position(x_emb)  # Already includes x + alpha*pe

y_emb = model.model.ar_audio_embedding(prompts)
y_pos = model.model.ar_audio_position(y_emb)  # Already includes y + alpha*pe
xy_pos = torch.cat([x_emb, y_pos], dim=1)

print(f"\nEmbedding shapes:")
print(f"  x_emb: {x_emb.shape}")
print(f"  y_pos: {y_pos.shape}")
print(f"  xy_pos: {xy_pos.shape}")

np.savetxt("gpt_py_xy_pos.txt", xy_pos.numpy().flatten(), fmt='%.10f')
print(f"Saved gpt_py_xy_pos.txt (embedding after text+audio fusion)")
