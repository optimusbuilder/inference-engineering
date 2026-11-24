# inference-engineering
---

## Why I built this

I have been messing around with parallel computing to understand how inference works with GPUs. The original goal was to build an inference engine designed for a gpu but before i start that, i want to understand what the goal of inference is as a whole. what does inference mean and what one would do. this project is me attempting to dive into inference and understand how one builds an engine that can use an already existing model. i have minimal writing input in the code and more of fundamental work and learning by seeing and manipulating things myself. 

## What it does

The engine was originally built for qwen 2,5 0.5B. Then, i ported it to support Cohere's Tiny Aya. It's an inference engine that runs on your cpu. the qwen model should run fine on a 16gb macbook. but the cohere model is slighly bigger and needs about 13gb to run normally. we used bf16 though so that reduced it. but the model is still pretty big and while it might work fine on a 16gb ram computer, 24gb is probbaly better so your computer can still do other things and not generate tokens ridiculolys slow
how does the engine work?
the engine was built to simulate a normal inference engine. so it has preety much every aspect you would think of it a normal engine. the engine has a prefill phase where we go oevr all the tokens in the prompt and fill in their keys and values into the kc cache at each layer. after the prefill stage, we have the decode layer to decode the next token. so the initail foreard pass run just does prefill and after that, it starts yielding tokens out
## Does it work?

yes! it works. i sampled the model and compared its reults with hugging face.  for the qwen model, our greedy matches hf's generate token on to 30 tokens for 3 separte prompts. i also tested the tiny aya agaisnt hf and the logits matched up to ~1e-4. the greedy also matched up to 20 tokens for 2 prompts

## Quick start

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# weights (Tiny Aya is gated: accept the license on its HF page, then `hf auth login`)
.venv/bin/hf download Qwen/Qwen2.5-0.5B --local-dir weights
.venv/bin/hf download CohereLabs/tiny-aya-base --local-dir weights-aya

# generate
.venv/bin/python generate.py "Once upon a time" --max-new 60
.venv/bin/python generate.py "Habari ya asubuhi! Leo" --model aya

# test against hugging face
.venv/bin/python test_lesson08.py
.venv/bin/python test_lesson09.py
```

## Files

| file | what it does |
|---|---|
| `config.py` | reads `config.json` |
| `loader.py` | reads the safetensors weights |
| `model.py` | the Qwen forward pass: RMSNorm, RoPE, attention, KV cache, MLP |
| `aya.py` | the Tiny Aya forward pass |
| `generate.py` | sampling and the generation loop |
| `test_lessonNN.py` | checks each piece against hugging face |

## Porting to Tiny Aya

What changed from Qwen:
- LayerNorm instead of RMSNorm
- parallel block: attention and MLP read the same input
- RoPE rotates adjacent pairs, and only on 3 of every 4 layers
- sliding-window attention (4096 tokens) on those same layers
- no Q/K/V biases

Three of the four wrong versions I tried still wrote readable text, so you can't spot a bad port by reading the output.

## Speed

On a 24 GB Mac, CPU, float32:
- Qwen 0.5B: ~25 tokens/s
- Tiny Aya 3.35B: ~2 tokens/s

## Next

I plan on rewriting the code myself in c++.

## Licenses

- Qwen 2.5 0.5B: Apache 2.0
- Tiny Aya: CC-BY-NC (non-commercial)
