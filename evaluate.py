"""
Avalia o modelo treinado em textos diferentes: mede loss e perplexidade.

Uso:
    python evaluate.py data/avaliacao/*.txt

Sempre inclui como referência a parte de validação do texto de treino
(o trecho que o modelo nunca viu, mas do mesmo livro).
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F

from model import MiniGPT

p = argparse.ArgumentParser()
p.add_argument("files", nargs="*", help="arquivos .txt para avaliar")
p.add_argument("--ckpt", default="outputs/ckpt.pt")
p.add_argument("--max_chars", type=int, default=50_000, help="limite de caracteres por texto (velocidade)")
args = p.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
cfg, chars = ckpt["config"], ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
block_size = cfg["block_size"]

model = MiniGPT(len(chars), block_size, cfg["n_embd"], cfg["n_head"], cfg["n_layer"], 0.0)
model.load_state_dict(ckpt["model"])
model.eval()


@torch.no_grad()
def evaluate(text):
    """Percorre o texto em janelas de `block_size` e calcula a loss média por caractere."""
    text = text[: args.max_chars]
    # caracteres que o modelo nunca viu no treino não têm token: são descartados
    known = [c for c in text if c in stoi]
    unknown_pct = 100 * (1 - len(known) / max(1, len(text)))
    ids = torch.tensor([stoi[c] for c in known])

    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(ids) - 1, block_size):
        x = ids[i : i + block_size]
        y = ids[i + 1 : i + block_size + 1]
        x = x[: len(y)]
        if len(x) == 0:
            break
        logits, _ = model(x.unsqueeze(0))
        total_loss += F.cross_entropy(logits[0], y, reduction="sum").item()
        total_tokens += len(y)

    loss = total_loss / total_tokens
    return loss, math.exp(loss), unknown_pct


# referência: a validação do próprio texto de treino
with open(cfg["data_path"], encoding="utf-8") as f:
    train_text = f.read()
val_text = train_text[int(cfg["train_split"] * len(train_text)) :]
textos = [("[referência] validação do treino", val_text)]

for path in args.files:
    with open(path, encoding="utf-8") as f:
        textos.append((os.path.basename(path), f.read()))

print(f"\nModelo: {args.ckpt} | vocabulário: {len(chars)} caracteres")
print(f"Chute aleatório: loss {math.log(len(chars)):.2f} | perplexidade {len(chars)}\n")
print(f"{'texto':<38} {'loss':>6} {'perplex.':>9} {'chars desconhecidos':>20}")
print("-" * 76)
for nome, texto in textos:
    loss, ppl, unk = evaluate(texto)
    print(f"{nome:<38} {loss:6.3f} {ppl:9.2f} {unk:19.1f}%")
print("\nQuanto MENOR a perplexidade, melhor o modelo 'entende' aquele texto.")
