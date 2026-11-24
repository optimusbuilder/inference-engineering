import torch

def embed(weights: dict[str, torch.Tensor], token_ids: list[int]) -> torch.Tensor:
    return weights["model.embed_tokens.weight"][token_ids]

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms * weight

def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    speed = theta ** (-torch.arange(half) / half)
    angle = (pos[:, None] * speed)[:, None, :]
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)

def attention(x: torch.Tensor, weights: dict[str, torch.Tensor], i: int, cfg, cache: dict) -> torch.Tensor:
    p = f"model.layers.{i}.self_attn."
    T, H, KV, D = x.shape[0], cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q = x @ weights[p + "q_proj.weight"].T + weights[p + "q_proj.bias"]
    k = x @ weights[p + "k_proj.weight"].T + weights[p + "k_proj.bias"]
    v = x @ weights[p + "v_proj.weight"].T + weights[p + "v_proj.bias"]
    q, k, v = q.view(T, H, D), k.view(T, KV, D), v.view(T, KV, D)
    start = cache[i][0].shape[0] if i in cache else 0
    pos = torch.arange(start, start + T)
    q, k = rope(q, pos, cfg.rope_theta), rope(k, pos, cfg.rope_theta)
    if i in cache:
        k, v = torch.cat([cache[i][0], k]), torch.cat([cache[i][1], v])
    cache[i] = (k, v)
    k, v = k.repeat_interleave(H // KV, dim=1), v.repeat_interleave(H // KV, dim=1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores = q @ k.transpose(1, 2) / D ** 0.5
    future = torch.triu(torch.ones(T, k.shape[1], dtype=torch.bool), diagonal=start + 1)
    att = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    out = att @ v
    out = out.transpose(0, 1).reshape(T, H * D)
    return out @ weights[p + "o_proj.weight"].T

def mlp(x: torch.Tensor, weights: dict[str, torch.Tensor], i: int) -> torch.Tensor:
    p = f"model.layers.{i}.mlp."
    gate = x @ weights[p + "gate_proj.weight"].T
    up = x @ weights[p + "up_proj.weight"].T
    h = torch.nn.functional.silu(gate) * up
    return h @ weights[p + "down_proj.weight"].T

def forward(token_ids: list[int], weights: dict[str, torch.Tensor], cfg, cache: dict) -> torch.Tensor:
    eps = cfg.rms_norm_eps
    x = embed(weights, token_ids)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        h = rms_norm(x, weights[p + "input_layernorm.weight"], eps)
        x = x + attention(h, weights, i, cfg, cache)
        h = rms_norm(x, weights[p + "post_attention_layernorm.weight"], eps)
        x = x + mlp(h, weights, i)
    x = rms_norm(x, weights["model.norm.weight"], eps)
    return x @ weights["model.embed_tokens.weight"].T
