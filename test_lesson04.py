import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2 import modeling_qwen2

from config import load_config
from loader import load_weights
import model
from model import embed, rms_norm, attention

model.rope = lambda x, pos, theta: x

cfg = load_config("weights/config.json")
weights = load_weights("weights/model.safetensors")
tok = AutoTokenizer.from_pretrained("weights")
ids = tok.encode("The cat sat on the mat")
T = len(ids)

x = embed(weights, ids)
h = rms_norm(x, weights["model.layers.0.input_layernorm.weight"], cfg.rms_norm_eps)
out = attention(h, weights, 0, cfg, {})
assert list(out.shape) == [T, 896]
print("attention output shape:", list(out.shape))

modeling_qwen2.apply_rotary_pos_emb = lambda q, k, cos, sin, *a, **kw: (q, k)
ref = AutoModelForCausalLM.from_pretrained("weights", dtype=torch.float32, attn_implementation="eager").eval()
causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None]
with torch.no_grad():
    for i in [0, 8, 23]:
        h = rms_norm(x, weights[f"model.layers.{i}.input_layernorm.weight"], cfg.rms_norm_eps)
        pe = ref.model.rotary_emb(h[None], torch.arange(T)[None])
        theirs = ref.model.layers[i].self_attn(h[None], position_embeddings=pe, attention_mask=causal)[0][0]
        diff = (attention(h, weights, i, cfg, {}) - theirs).abs().max().item()
        assert diff < 1e-4, (i, diff)
        print(f"layer {i:2d}: matches HF attention (RoPE off), max difference {diff:.1e}")
