"""
Monta o guia (site estático) a partir do template oficial do curso.

    python guia/build.py            -> gera _site/index.html

- guia/template.html : template web do CC IFC Videira (não editar)
- guia/conteudo.html : seções do guia (<section> ... </section>)
- {{codigo:caminho|legenda}} dentro do conteúdo é trocado pelo arquivo real do
  projeto, com realce de sintaxe e legenda numerada. Assim o código do guia é
  sempre o mesmo código do repositório.
"""
import html
import io
import keyword
import pathlib
import re
import tokenize

RAIZ = pathlib.Path(__file__).resolve().parent.parent
GUIA = RAIZ / "guia"
SAIDA = RAIZ / "_site"

META = {
    "titulo": "Construindo uma mini-LLM do zero",
    "rotulo": "Material didático · Roteiro prático",
    "subtitulo": "Treine um GPT de verdade, em miniatura, no seu computador, com PyTorch e Docker.",
    "info": [
        ("Professor", "Prof. Tiago Heineck"),
        ("Duração", "≈ 2 horas"),
        ("Pré-requisitos", "Python básico · Docker"),
    ],
    "rodape": (
        "Instituto Federal Catarinense · Campus Videira<br>"
        "Prof. Tiago Heineck · "
        '<a href="https://github.com/tiagoheineck/mini-llm">github.com/tiagoheineck/mini-llm</a>'
    ),
}


# ------------------------------------------------------------------
# Realce de sintaxe usando as classes do template (.kw .cm .nu)
# ------------------------------------------------------------------
def realce_python(codigo):
    linhas = codigo.splitlines(keepends=True)
    offsets = [0]
    for l in linhas:
        offsets.append(offsets[-1] + len(l))
    pos = lambda lc: offsets[lc[0] - 1] + lc[1]

    partes, ultimo = [], 0
    for tok in tokenize.generate_tokens(io.StringIO(codigo).readline):
        cls = None
        if tok.type == tokenize.COMMENT:
            cls = "cm"
        elif tok.type == tokenize.NUMBER:
            cls = "nu"
        elif tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
            cls = "kw"
        elif tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
            cls = "cm"  # docstrings como comentário
        if cls:
            ini, fim = pos(tok.start), pos(tok.end)
            partes.append(html.escape(codigo[ultimo:ini], quote=False))
            partes.append(f'<span class="{cls}">{html.escape(codigo[ini:fim], quote=False)}</span>')
            ultimo = fim
    partes.append(html.escape(codigo[ultimo:], quote=False))
    return "".join(partes)


def realce_simples(codigo):
    """YAML, Dockerfile, shell: comentários com # e palavras-chave do Dockerfile."""
    saida = []
    for linha in codigo.splitlines(keepends=True):
        corpo, _, nl = linha.partition("\n")
        m = re.match(r"^(\s*)(#.*)$", corpo) or re.match(r"^(.*?\s)(#.*)$", corpo)
        if m:
            saida.append(html.escape(m.group(1), quote=False) + f'<span class="cm">{html.escape(m.group(2), quote=False)}</span>')
        else:
            esc = html.escape(corpo, quote=False)
            esc = re.sub(r"^(FROM|WORKDIR|RUN|COPY|ENV|ENTRYPOINT|CMD)\b", r'<span class="kw">\1</span>', esc)
            esc = re.sub(r"^(\s*)([\w-]+):", r'\1<span class="kw">\2</span>:', esc) if not esc.startswith("<span") else esc
            saida.append(esc)
        saida.append("\n" if nl or linha.endswith("\n") else "")
    return "".join(saida)


contador = {"codigo": 0}


def figura_codigo(caminho, legenda):
    contador["codigo"] += 1
    texto = (RAIZ / caminho).read_text(encoding="utf-8").rstrip("\n")
    corpo = realce_python(texto + "\n").rstrip("\n") if caminho.endswith(".py") else realce_simples(texto)
    return (
        f'<figure class="codigo">\n'
        f'  <div class="cab"><span>{html.escape(caminho)}</span><button type="button" class="copiar">Copiar</button></div>\n'
        f"<pre><code>{corpo}</code></pre>\n"
        f"</figure>\n"
        f'<p class="leg">Código {contador["codigo"]} – {legenda}</p>'
    )


def comando(texto, rotulo="terminal"):
    """{{cmd:...}} vira um bloco de terminal com botão copiar."""
    return (
        f'<figure class="codigo">\n'
        f'  <div class="cab"><span>{rotulo}</span><button type="button" class="copiar">Copiar</button></div>\n'
        f"<pre><code>{realce_simples(texto.strip())}</code></pre>\n"
        f"</figure>"
    )


# ------------------------------------------------------------------
# Montagem
# ------------------------------------------------------------------
def substituir(padrao, novo, texto):
    resultado, n = re.subn(padrao, lambda _: novo, texto, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"Template mudou: não encontrei {padrao!r}")
    return resultado


def main():
    tpl = (GUIA / "template.html").read_text(encoding="utf-8")
    conteudo = (GUIA / "conteudo.html").read_text(encoding="utf-8")

    conteudo = re.sub(r"\{\{codigo:([^|}]+)\|([^}]+)\}\}", lambda m: figura_codigo(m[1].strip(), m[2].strip()), conteudo)
    conteudo = re.sub(r"\{\{cmd(?::([^}\n]+))?\}\}(.*?)\{\{/cmd\}\}", lambda m: comando(m[2], m[1] or "terminal"), conteudo, flags=re.S)

    t = META["titulo"]
    tpl = substituir(r"<title>.*?</title>", f"<title>{t} · Ciência da Computação IFC Videira</title>", tpl)
    tpl = substituir(r'<p class="rotulo">.*?</p>', f'<p class="rotulo">{META["rotulo"]}</p>', tpl)
    tpl = substituir(r"<h1>.*?</h1>", f"<h1>{t}</h1>", tpl)
    tpl = substituir(r'<p class="sub">.*?</p>', f'<p class="sub">{META["subtitulo"]}</p>', tpl)
    dl = "\n".join(f"      <div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in META["info"])
    tpl = substituir(r"<dl>.*?</dl>", f"<dl>\n{dl}\n    </dl>", tpl)
    tpl = substituir(r'<span class="t">.*?</span>', f'<span class="t">{t}</span>', tpl)
    tpl = substituir(r"<main>.*?</main>", f"<main>\n{conteudo}\n  </main>", tpl)
    tpl = substituir(r"<div>Instituto Federal Catarinense.*?</div>", f"<div>{META['rodape']}</div>", tpl)

    SAIDA.mkdir(exist_ok=True)
    (SAIDA / "index.html").write_text(tpl, encoding="utf-8")
    print(f"Guia gerado: {SAIDA / 'index.html'} ({contador['codigo']} blocos de código)")


if __name__ == "__main__":
    main()
