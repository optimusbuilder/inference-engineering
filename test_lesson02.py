from safetensors.torch import load_file

from config import load_config
from loader import load_weights

cfg = load_config("weights/config.json")
assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.head_dim) == (896, 24, 64)
assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (14, 2)
print("config ok:", cfg)

ours = load_weights("weights/model.safetensors")
assert len(ours) == 290
assert "lm_head.weight" not in ours
assert sum(t.numel() for t in ours.values()) == 494_032_768
assert list(ours["model.layers.0.self_attn.k_proj.weight"].shape) == [128, 896]
print("290 tensors, 494,032,768 parameters")

the = ours["model.embed_tokens.weight"][785]
print("first numbers of 'The':", [round(x, 4) for x in the[:4].tolist()])
assert abs(the[0].item() - -0.032470703125) < 1e-12

ref = load_file("weights/model.safetensors")
for name, t in ref.items():
    assert ours[name].dtype == t.float().dtype and ours[name].equal(t.float()), name
print("every tensor matches the safetensors library exactly")
