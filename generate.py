"""
Gera texto a partir do checkpoint treinado.

Uso:
    python generate.py --prompt "A noite" --temperature 0.8 --tokens 300
"""
import argparse

import torch

from model import MiniGPT

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="outputs/ckpt.pt")
p.add_argument("--prompt", default="O ")
p.add_argument("--tokens", type=int, default=300)
p.add_argument("--temperature", type=float, default=0.8, help="<1 mais previsível, >1 mais aleatório")
p.add_argument("--top_k", type=int, default=20, help="só sorteia entre os k tokens mais prováveis")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
cfg, chars = ckpt["config"], ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}

model = MiniGPT(len(chars), cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], 0.0)
model.load_state_dict(ckpt["model"])
model.eval()

prompt = "".join(c for c in args.prompt if c in stoi) or chars[0]
idx = torch.tensor([[stoi[c] for c in prompt]])
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print("".join(chars[i] for i in out[0].tolist()))
