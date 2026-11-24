import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import load_config
from generate import generate, sample
from loader import load_weights
from model import forward

cfg = load_config("weights/config.json")
weights = load_weights("weights/model.safetensors")
tok = AutoTokenizer.from_pretrained("weights")
hf = AutoModelForCausalLM.from_pretrained("weights", dtype=torch.float32).eval()
eos = tok.eos_token_id

for text in ["The cat sat on the", "Once upon a time", "The capital of France is"]:
    ids = tok.encode(text)
    ours = list(generate(ids, weights, cfg, 30, 0, 1.0, eos))
    with torch.no_grad():
        ref = hf.generate(torch.tensor([ids]), max_new_tokens=30, do_sample=False)[0, len(ids):].tolist()
    assert ours == ref, (text, tok.decode(ours), tok.decode(ref))
    print(f"greedy matches HF, 30 tokens: {text!r} →{tok.decode(ours[:12])!r}…")

logits = forward(tok.encode("The cat sat on the"), weights, cfg, {})[-1]
gen = torch.Generator().manual_seed(0)
mat = tok.encode(" mat")[0]

assert sample(logits, 0, 1.0, gen) == mat == int(logits.argmax())
assert all(sample(logits, 1.0, 1e-9, gen) == mat for _ in range(50))
print("temperature 0 = argmax = ' mat'; top-p ≈ 0 always picks it too")

top7 = set(logits.topk(7).indices.tolist())
draws = [sample(logits, 1.0, 0.9, gen) for _ in range(300)]
assert set(draws) <= top7, set(draws) - top7
print(f"top-p 0.9: 300 draws, {len(set(draws))} different tokens, all inside the 7 kept · ' mat' {draws.count(mat) / 3:.0f}%")

ids = tok.encode("Once upon a time")
a = list(generate(ids, weights, cfg, 20, 0.7, 0.9, eos, seed=0))
b = list(generate(ids, weights, cfg, 20, 0.7, 0.9, eos, seed=0))
c = list(generate(ids, weights, cfg, 20, 0.7, 0.9, eos, seed=1))
assert a == b and a != c
print(f"seed 0 twice: identical · seed 1: different →{tok.decode(c)!r}")

assert list(generate(tok.encode("The cat sat on the"), weights, cfg, 10, 0, 1.0, eos=mat)) == []
print("EOS stops the loop")
