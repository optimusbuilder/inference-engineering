import json
import struct

import torch

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def load_weights(path: str, float32: bool = True) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = bytearray(f.read())

    header.pop("__metadata__", None)
    weights = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        t = torch.frombuffer(data[start:end], dtype=DTYPES[info["dtype"]])
        weights[name] = t.reshape(info["shape"]).float() if float32 else t.reshape(info["shape"])
    return weights


def load_folder(folder: str, float32: bool = True) -> dict[str, torch.Tensor]:
    with open(f"{folder}/model.safetensors.index.json") as f:
        files = set(json.load(f)["weight_map"].values())
    weights = {}
    for file in sorted(files):
        weights.update(load_weights(f"{folder}/{file}", float32))
    return weights
