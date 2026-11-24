import json
from dataclasses import dataclass, fields

import torch

from model import embed, mlp

@dataclass
class AyaConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    layer_norm_eps: float
    rope_theta: float
    sliding_window: int
    layer_types: list[str]
    logit_scale: float

def load_config(path: str) -> AyaConfig:
    with open(path) as f:
        raw = json.load(f)
    return AyaConfig(**{f.name: raw[f.name] for f in fields(AyaConfig)})

class Float32(dict):
    def __getitem__(self, name: str) -> torch.Tensor:
        return super().__getitem__(name).float()

def layer_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x - x.mean(-1, keepdim=True)
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    speed = theta ** (-torch.arange(half) / half)
    angle = (pos[:, None] * speed)[:, None, :]
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)

def attention(x: torch.Tensor, weights: dict[str, torch.Tensor], i: int, cfg, cache: dict) -> torch.Tensor:
    p = f"model.layers.{i}.self_attn."
    T, H, KV, D = x.shape[0], cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q = (x @ weights[p + "q_proj.weight"].T).view(T, H, D)
    k = (x @ weights[p + "k_proj.weight"].T).view(T, KV, D)
    v = (x @ weights[p + "v_proj.weight"].T).view(T, KV, D)
    start = cache[i][0].shape[0] if i in cache else 0
    pos = torch.arange(start, start + T)
    sliding = cfg.layer_types[i] == "sliding_attention"
    if sliding:
        q, k = rope(q, pos, cfg.rope_theta), rope(k, pos, cfg.rope_theta)
    if i in cache:
        k, v = torch.cat([cache[i][0], k]), torch.cat([cache[i][1], v])
    cache[i] = (k, v)
    k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores = q @ k.transpose(1, 2) / D ** 0.5
    kpos = torch.arange(k.shape[1])
    hidden = kpos[None, :] > pos[:, None]
    if sliding:
        hidden |= kpos[None, :] <= pos[:, None] - cfg.sliding_window
    att = torch.softmax(scores.masked_fill(hidden, float("-inf")), dim=-1)
    out = (att @ v).transpose(0, 1).reshape(T, H * D)
    return out @ weights[p + "o_proj.weight"].T

def forward(token_ids: list[int], weights: dict[str, torch.Tensor], cfg, cache: dict) -> torch.Tensor:
    eps = cfg.layer_norm_eps
    x = embed(weights, token_ids)
    for i in range(cfg.num_hidden_layers):
        h = layer_norm(x, weights[f"model.layers.{i}.input_layernorm.weight"], eps)
        x = x + attention(h, weights, i, cfg, cache) + mlp(h, weights, i)
    x = layer_norm(x, weights["model.norm.weight"], eps)
    return x @ weights["model.embed_tokens.weight"].T * cfg.logit_scale
