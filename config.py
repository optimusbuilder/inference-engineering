import json
from dataclasses import dataclass, fields


@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = json.load(f)
    return Config(**{f.name: raw[f.name] for f in fields(Config)})
