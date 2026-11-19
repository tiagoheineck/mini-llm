"""
Servidor web simples (só biblioteca padrão + torch) para conversar com a mini-LLM.

    python server.py           -> http://localhost:8000

Endpoints:
    GET  /                 página de chat (static/index.html)
    POST /api/generate     gera texto em streaming, um caractere por vez
    POST /api/next         probabilidades do próximo caractere
    GET  /api/info         informações do modelo
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from model import MiniGPT

CKPT = os.environ.get("CKPT", "outputs/ckpt.pt")
PORT = int(os.environ.get("PORT", 8000))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

ckpt = torch.load(CKPT, map_location="cpu")
cfg, chars = ckpt["config"], ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
model = MiniGPT(len(chars), cfg["block_size"], cfg["n_embd"], cfg["n_head"], cfg["n_layer"], 0.0)
model.load_state_dict(ckpt["model"])
model.eval()
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"Modelo carregado: {CKPT} ({N_PARAMS:,} parâmetros)")


def encode(text):
    """Converte texto em ids; caracteres fora do vocabulário são ignorados."""
    ids = [stoi[c] for c in text if c in stoi]
    ignored = sorted({c for c in text if c not in stoi})
    return ids or [stoi[" "]], ignored


@torch.no_grad()
def next_probs(ids):
    logits, _ = model(torch.tensor([ids[-cfg["block_size"]:]]))
    return logits[0, -1]


def stream_generate(ids, n_tokens, temperature, top_k):
    """Laço autoregressivo: prevê 1 caractere, anexa ao contexto, repete."""
    for _ in range(n_tokens):
        logits = next_probs(ids) / max(temperature, 1e-3)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, len(chars)))
            logits[logits < v[-1]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
        yield chars[nxt]


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self._json({
                "params": N_PARAMS, "vocab": len(chars), "block_size": cfg["block_size"],
                "n_layer": cfg["n_layer"], "n_embd": cfg["n_embd"], "n_head": cfg["n_head"],
                "data_path": cfg["data_path"], "max_iters": cfg["max_iters"],
            })
        else:
            self.send_error(404)

    def do_POST(self):
        req = self._body()
        ids, ignored = encode(req.get("prompt", ""))

        if self.path == "/api/next":
            probs = F.softmax(next_probs(ids), dim=-1)
            top = torch.topk(probs, 10)
            self._json({
                "ignored": ignored,
                "top": [{"char": chars[i], "p": round(p, 4)} for p, i in zip(top.values.tolist(), top.indices.tolist())],
            })
        elif self.path == "/api/generate":
            n = max(1, min(int(req.get("tokens", 300)), 2000))
            temp = float(req.get("temperature", 0.8))
            top_k = int(req.get("top_k", 20))
            # resposta em streaming: sem Content-Length, conexão fecha no fim
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for ch in stream_generate(ids, n, temp, top_k):
                    self.wfile.write(ch.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # usuário clicou em "parar"
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass  # silencia o log de cada requisição


if __name__ == "__main__":
    print(f"Abra http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
