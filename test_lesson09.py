import subprocess
import sys
import tempfile

import torch
from transformers import AutoTokenizer

PROMPTS = ["The cat sat on the", "Habari ya asubuhi, rafiki", "Ẹ kú àárọ̀. Báwo"]
GREEDY = ["The cat sat on the", "Once upon a time"]
LONG = "One two three four five six seven eight nine ten eleven twelve"
tok = AutoTokenizer.from_pretrained("weights-aya")

def hf_answers(path: str):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained("weights-aya", dtype=torch.float32, attn_implementation="eager").eval()
    out = {}
    with torch.no_grad():
        for text in PROMPTS:
            out[text] = hf(torch.tensor([tok.encode(text)])).logits[0]
        for text in GREEDY:
            ids = tok.encode(text)
            out["greedy " + text] = hf.generate(torch.tensor([ids]), max_new_tokens=20, do_sample=False)[0, len(ids):].tolist()
        hf.config.sliding_window = 4
        for layer in hf.model.layers:
            if layer.self_attn.sliding_window is not None:
                layer.self_attn.sliding_window = 4
        out["window 4"] = hf(torch.tensor([tok.encode(LONG)]), use_cache=False).logits[0]
    torch.save(out, path)

if len(sys.argv) == 3 and sys.argv[1] == "hf":
    hf_answers(sys.argv[2])
    sys.exit()

path = tempfile.mktemp(suffix=".pt")
subprocess.run([sys.executable, __file__, "hf", path], check=True)
ref = torch.load(path)
print("HF answers saved; now loading our engine")

import aya
from generate import generate
from loader import load_folder
from model import rms_norm, rope as qwen_rope

cfg = aya.load_config("weights-aya/config.json")
weights = aya.Float32(load_folder("weights-aya", float32=False))

for text in PROMPTS:
    ours = aya.forward(tok.encode(text), weights, cfg, {})
    diff = (ours - ref[text]).abs().max().item()
    assert diff < 2e-3 and ours[-1].argmax() == ref[text][-1].argmax(), (text, diff)
    print(f"logits match HF: {text!r} [{ours.shape[0]}, {ours.shape[1]}] max diff {diff:.1e} → {tok.decode([int(ours[-1].argmax())])!r}")

for text in GREEDY:
    ours = list(generate(tok.encode(text), weights, cfg, 20, 0, 1.0, tok.eos_token_id, forward=aya.forward))
    assert ours == ref["greedy " + text], (tok.decode(ours), tok.decode(ref["greedy " + text]))
    print(f"greedy matches HF, 20 tokens: {text!r} →{tok.decode(ours)!r}")

ids = tok.encode("The cat sat on the")
cache = {}
aya.forward(ids[:-1], weights, cfg, cache)
step = aya.forward(ids[-1:], weights, cfg, cache)[-1]
full = aya.forward(ids, weights, cfg, {})[-1]
print(f"prefill {len(ids) - 1} + decode 1 = all at once: max diff {(step - full).abs().max().item():.1e}")
assert torch.allclose(step, full, atol=1e-3)

cfg.sliding_window = 4
ours = aya.forward(tok.encode(LONG), weights, cfg, {})
diff = (ours - ref["window 4"]).abs().max().item()
cfg.sliding_window = 4096
assert diff < 2e-3, diff
print(f"sliding window 4 on a {ours.shape[0]}-token prompt matches HF: max diff {diff:.1e}")

mat, ids = tok.encode(" mat", add_special_tokens=False)[0], tok.encode("The cat sat on the")
def p_mat():
    return torch.softmax(aya.forward(ids, weights, cfg, {})[-1], -1)[mat].item()
print(f"' mat' probability, correct port: {p_mat():.1%}")
right_norm, right_rope, right_types = aya.layer_norm, aya.rope, cfg.layer_types
aya.layer_norm = rms_norm
print(f"  with RMSNorm instead of LayerNorm: {p_mat():.1%}")
aya.layer_norm = right_norm
aya.rope = qwen_rope
print(f"  with RoPE pairs as halves:         {p_mat():.1%}")
aya.rope = right_rope
cfg.layer_types = ["sliding_attention"] * 36
print(f"  with RoPE on all 36 layers:        {p_mat():.1%}")
cfg.layer_types = right_types
