import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from config import load_config
from loader import load_weights
from model import embed, rms_norm, rope, attention

cfg = load_config("weights/config.json")
weights = load_weights("weights/model.safetensors")
tok = AutoTokenizer.from_pretrained("weights")
ids = tok.encode("The cat sat on the mat")
T = len(ids)
pos = torch.arange(T)
torch.manual_seed(0)

x = torch.randn(T, 14, 64)
r = rope(x, pos, cfg.rope_theta)
assert r.shape == x.shape
pair_len = lambda t: (t[..., :32] ** 2 + t[..., 32:] ** 2).sqrt()
assert torch.allclose(pair_len(r), pair_len(x), atol=1e-5)
assert torch.equal(r[0], x[0])
print("rope keeps shapes and lengths; position 0 is unchanged")

q, k = torch.randn(1, 1, 64), torch.randn(1, 1, 64)
score = lambda m, n: (rope(q, torch.tensor([m]), cfg.rope_theta) * rope(k, torch.tensor([n]), cfg.rope_theta)).sum().item()
assert abs(score(2, 1) - score(1002, 1001)) < 1e-3
print(f"score at (2, 1) = {score(2, 1):.4f}, at (1002, 1001) = {score(1002, 1001):.4f}")

ref = AutoModelForCausalLM.from_pretrained("weights", dtype=torch.float32, attn_implementation="eager").eval()

cos, sin = ref.model.rotary_emb(x, pos[None])
q, k = torch.randn(T, 14, 64), torch.randn(T, 2, 64)
hq, hk = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin)
diff = max((rope(q, pos, cfg.rope_theta) - hq[0].transpose(0, 1)).abs().max().item(),
           (rope(k, pos, cfg.rope_theta) - hk[0].transpose(0, 1)).abs().max().item())
assert diff < 1e-4, diff
print(f"rope matches HF's, max difference {diff:.1e}")

x = embed(weights, ids)
causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None]
with torch.no_grad():
    for i in [0, 8, 23]:
        h = rms_norm(x, weights[f"model.layers.{i}.input_layernorm.weight"], cfg.rms_norm_eps)
        pe = ref.model.rotary_emb(h[None], pos[None])
        theirs = ref.model.layers[i].self_attn(h[None], position_embeddings=pe, attention_mask=causal)[0][0]
        diff = (attention(h, weights, i, cfg, {}) - theirs).abs().max().item()
        assert diff < 1e-4, (i, diff)
        print(f"layer {i:2d}: matches HF attention (RoPE on), max difference {diff:.1e}")
