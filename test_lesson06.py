import torch
from transformers import AutoTokenizer

from config import load_config
from loader import load_weights
from model import embed, rms_norm, attention

cfg = load_config("weights/config.json")
weights = load_weights("weights/model.safetensors")
ids = AutoTokenizer.from_pretrained("weights").encode("The cat sat on the mat")
T = len(ids)
x = embed(weights, ids)

for i in [0, 8, 23]:
    h = rms_norm(x, weights[f"model.layers.{i}.input_layernorm.weight"], cfg.rms_norm_eps)
    full = attention(h, weights, i, cfg, {})

    cache = {}
    attention(h[:5], weights, i, cfg, cache)
    assert cache[i][0].shape == (5, 2, 64)
    last = attention(h[5:], weights, i, cfg, cache)
    assert cache[i][0].shape == (6, 2, 64)
    d1 = (last - full[5:]).abs().max().item()

    cache = {}
    rows = torch.cat([attention(h[t:t + 1], weights, i, cfg, cache) for t in range(T)])
    d2 = (rows - full).abs().max().item()

    cache = {}
    chunks = torch.cat([attention(h[:3], weights, i, cfg, cache), attention(h[3:], weights, i, cfg, cache)])
    d3 = (chunks - full).abs().max().item()

    assert max(d1, d2, d3) < 1e-4, (i, d1, d2, d3)
    print(f"layer {i:2d}: prefill 5 + decode 1: {d1:.1e} · one at a time: {d2:.1e} · 3 + 3: {d3:.1e}")
