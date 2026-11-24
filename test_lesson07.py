import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from loader import load_weights
from model import forward

cfg = load_config("weights/config.json")
weights = load_weights("weights/model.safetensors")
tok = AutoTokenizer.from_pretrained("weights")
hf = AutoModelForCausalLM.from_pretrained("weights", dtype=torch.float32).eval()

for text in ["The cat sat on the", "The capital of France is", "1, 2, 3, 4,"]:
    ids = tok.encode(text)
    ours = forward(ids, weights, cfg, {})
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0]
    assert ours.shape == ref.shape == (len(ids), cfg.vocab_size)
    diff = (ours - ref).abs().max().item()
    assert diff < 1e-3, (text, diff)
    assert ours[-1].argmax() == ref[-1].argmax()
    p = torch.softmax(ours[-1], -1)
    best = p.argmax().item()
    print(f"{text!r:28} max diff {diff:.1e} · next: {tok.decode([best])!r} {p[best]:.1%}")

ids = tok.encode("The cat sat on the")
full = forward(ids, weights, cfg, {})
cache = {}
forward(ids[:4], weights, cfg, cache)
assert len(cache) == 24 and cache[0][0].shape == (4, 2, 64)
last = forward(ids[4:], weights, cfg, cache)
assert cache[23][0].shape == (5, 2, 64)
d = (last[0] - full[-1]).abs().max().item()
assert d < 1e-3, d
print(f"prefill 4 + decode 1 vs all at once: max diff {d:.1e}")
