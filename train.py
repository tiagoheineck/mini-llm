"""
Treino de uma mini-LLM em nível de caractere.

Uso:
    python train.py                                  # usa config.yaml
    python train.py --learning_rate 1e-3 --n_layer 2 # sobrescreve parâmetros
"""
import argparse
import json
import math
import os
import time

import torch
import yaml

from model import MiniGPT


# ------------------------------------------------------------------
# 1. Configuração: config.yaml + overrides pela linha de comando
# ------------------------------------------------------------------
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args, rest = parser.parse_known_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # cada chave do YAML vira um argumento --chave opcional
    for key, value in cfg.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    return vars(parser.parse_args())


cfg = load_config()
torch.manual_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("PARÂMETROS DE TREINO")
for k, v in cfg.items():
    print(f"  {k:<16} = {v}")
print(f"  {'device':<16} = {device}")
print("=" * 60)


# ------------------------------------------------------------------
# 2. Dados e tokenização (cada caractere é um token)
# ------------------------------------------------------------------
with open(cfg["data_path"], encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(cfg["train_split"] * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Texto: {len(text):,} caracteres | vocabulário: {vocab_size} tokens")
print(f"Treino: {len(train_data):,} tokens | validação: {len(val_data):,} tokens")


def get_batch(split):
    """Pega `batch_size` trechos aleatórios. O alvo (y) é o texto deslocado 1 posição."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg["block_size"] - 1, (cfg["batch_size"],))
    x = torch.stack([d[i : i + cfg["block_size"]] for i in ix])
    y = torch.stack([d[i + 1 : i + cfg["block_size"] + 1] for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------------------
# 3. Modelo e otimizador
# ------------------------------------------------------------------
model = MiniGPT(
    vocab_size=vocab_size,
    block_size=cfg["block_size"],
    n_embd=cfg["n_embd"],
    n_head=cfg["n_head"],
    n_layer=cfg["n_layer"],
    dropout=cfg["dropout"],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Modelo: {n_params:,} parâmetros ({n_params / 1e6:.2f} M)")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
)


def get_lr(it):
    """Learning rate com warmup linear + decaimento em cosseno."""
    if it < cfg["warmup_iters"]:
        return cfg["learning_rate"] * (it + 1) / cfg["warmup_iters"]
    progress = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


@torch.no_grad()
def estimate_loss():
    model.eval()  # desliga dropout
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(prompt="O ", n_tokens=200):
    model.eval()
    idx = torch.tensor([encode(prompt)], device=device)
    text_out = decode(model.generate(idx, n_tokens, temperature=0.8, top_k=20)[0].tolist())
    model.train()
    return text_out


# ------------------------------------------------------------------
# 4. Loop de treino
# ------------------------------------------------------------------
os.makedirs(cfg["out_dir"], exist_ok=True)
history = []
t0 = time.time()
print(f"\nLoss inicial esperada (chute aleatório) ≈ ln({vocab_size}) = {math.log(vocab_size):.2f}\n")

for it in range(cfg["max_iters"] + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # --- avaliação periódica ---
    if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
        losses = estimate_loss()
        # overfitting = loss de treino cai, mas a de validação volta a subir
        best_val = min([h["val"] for h in history], default=float("inf"))
        flag = "  <- val subiu: possível overfitting" if losses["val"] > best_val + 0.05 else ""
        print(
            f"passo {it:5d} | loss treino {losses['train']:.4f} | "
            f"loss val {losses['val']:.4f} | lr {lr:.2e} | {time.time() - t0:6.1f}s{flag}"
        )
        history.append({"iter": it, "lr": lr, **losses})

    if it > 0 and it % cfg["sample_interval"] == 0:
        print("-" * 60 + f"\nAmostra no passo {it}:\n{sample()}\n" + "-" * 60)

    if it == cfg["max_iters"]:
        break

    # --- passo de treino: forward -> loss -> backward -> update ---
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    optimizer.step()

# ------------------------------------------------------------------
# 5. Salvar resultados
# ------------------------------------------------------------------
torch.save(
    {"model": model.state_dict(), "config": cfg, "chars": chars},
    os.path.join(cfg["out_dir"], "ckpt.pt"),
)
with open(os.path.join(cfg["out_dir"], "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\nTreino concluído em {time.time() - t0:.1f}s")
print(f"Checkpoint salvo em {cfg['out_dir']}/ckpt.pt")
print("\nTexto final gerado:\n" + sample(n_tokens=400))
