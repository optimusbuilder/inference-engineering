import argparse
import sys
import time

import torch
from transformers import AutoTokenizer

import aya
from config import load_config
from loader import load_folder, load_weights
from model import forward

def sample(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> int:
    if temperature == 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, dim=-1)
    probs, order = probs.sort(descending=True)
    before = probs.cumsum(0) - probs
    probs[before >= top_p] = 0
    pick = torch.multinomial(probs, 1, generator=gen)
    return int(order[pick])

def generate(prompt_ids: list[int], weights: dict[str, torch.Tensor], cfg, max_new: int,
             temperature: float, top_p: float, eos: int, seed: int = 0, forward=forward):
    gen = torch.Generator().manual_seed(seed)
    cache, ids = {}, prompt_ids
    for _ in range(max_new):
        logits = forward(ids, weights, cfg, cache)
        token = sample(logits[-1], temperature, top_p, gen)
        if token == eos:
            return
        yield token
        ids = [token]

def main():
    ap = argparse.ArgumentParser(description="Generate text with our own engine: Qwen 2.5 0.5B or Tiny Aya.")
    ap.add_argument("prompt")
    ap.add_argument("--max-new", type=int, default=50, help="most tokens to add (default 50)")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = greedy (default 0.7)")
    ap.add_argument("--top-p", type=float, default=0.9, help="keep tokens until their total reaches p (default 0.9)")
    ap.add_argument("--seed", type=int, default=0, help="same seed, same text (default 0)")
    ap.add_argument("--model", choices=["qwen", "aya"], default="qwen", help="Qwen 2.5 0.5B or Tiny Aya base")
    args = ap.parse_args()

    if args.model == "qwen":
        folder, cfg, fwd = "weights", load_config("weights/config.json"), forward
        weights = load_weights("weights/model.safetensors")
    else:
        folder, cfg, fwd = "weights-aya", aya.load_config("weights-aya/config.json"), aya.forward
        weights = aya.Float32(load_folder("weights-aya", float32=False))
    tok = AutoTokenizer.from_pretrained(folder)
    prompt_ids = tok.encode(args.prompt)

    print(args.prompt, end="", flush=True)
    out, shown, start = [], "", time.time()
    for token in generate(prompt_ids, weights, cfg, args.max_new, args.temperature, args.top_p, tok.eos_token_id, args.seed, fwd):
        if not out:
            first = time.time()
        out.append(token)
        text = tok.decode(out)
        print(text[len(shown):], end="", flush=True)
        shown = text
    print(flush=True)
    if len(out) > 1:
        rate = (len(out) - 1) / (time.time() - first)
        print(f"[{len(prompt_ids)} prompt tokens · first token after {first - start:.2f} s · "
              f"then {rate:.1f} tokens/s]", file=sys.stderr)

if __name__ == "__main__":
    main()
