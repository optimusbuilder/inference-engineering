import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import load_config
from loader import load_weights
from model import embed, rms_norm

cfg = load_config("weights/config.json")
weights = load_weights("weights/model.safetensors")
tok = AutoTokenizer.from_pretrained("weights")

ids = tok.encode("The cat sat")
assert ids == [785, 8251, 7578]
print("tokens:", ids)

x = embed(weights, ids)
assert list(x.shape) == [3, 896]
print("embedding shape:", list(x.shape), " first of 'The':", x[0, 0].item())

w = weights["model.layers.0.input_layernorm.weight"]
y = rms_norm(x, w, cfg.rms_norm_eps)
rms_the = x[0].pow(2).mean().add(cfg.rms_norm_eps).sqrt().item()
assert abs(rms_the - 0.012436) < 1e-6
assert abs((x / rms_the)[0, 0].item() - -2.611) < 1e-3
assert abs(y[0, 0].item() - 0.1205) < 1e-4
print("worked example: rms 0.012436, -0.0325 -> -2.611 -> 0.1205")

ref = AutoModelForCausalLM.from_pretrained("weights", dtype=torch.float32).eval()
with torch.no_grad():
    ids_t = torch.tensor([ids])
    ref_x = ref.model.embed_tokens(ids_t)[0]
    assert torch.equal(x, ref_x)
    print("embedding matches HF exactly")
    for name, norm in [("layers.0.input_layernorm", ref.model.layers[0].input_layernorm),
                       ("layers.12.post_attention_layernorm", ref.model.layers[12].post_attention_layernorm),
                       ("norm (final)", ref.model.norm)]:
        ours = rms_norm(x, weights[f"model.{name.split(' ')[0]}.weight"], cfg.rms_norm_eps)
        diff = (ours - norm(ref_x)).abs().max().item()
        assert diff < 1e-5, (name, diff)
        print(f"rms_norm matches HF {name}: max difference {diff:.1e}")
