# Mini-LLM: treinando um GPT do zero com Docker

📘 **Guia passo a passo para estudantes:** https://tiagoheineck.github.io/mini-llm/
(gerado de `guia/` pela GitHub Action em `.github/workflows/pages.yml`; para visualizar localmente: `python guia/build.py` e abra `_site/index.html`)

Exemplo didático: um Transformer *decoder-only* (a mesma arquitetura do GPT) com ~0,8 M de
parâmetros, treinado em CPU sobre o texto de **Dom Casmurro** (Machado de Assis, domínio público).
O treino padrão leva ~5 minutos.

| Arquivo | O que é |
|---|---|
| `config.yaml` | **Todos os parâmetros de treino**, comentados |
| `model.py` | O modelo (atenção causal, blocos Transformer, geração) |
| `train.py` | Tokenização, loop de treino, avaliação, checkpoint |
| `generate.py` | Gera texto a partir do modelo treinado |
| `data/input.txt` | Texto de treino (troque por qualquer `.txt`) |

## Como rodar

```bash
docker compose build
```

```bash
docker compose run --rm train
```

```bash
docker compose run --rm generate --prompt "Capitu" --temperature 0.8
```

O checkpoint (`ckpt.pt`) e a curva de loss (`history.json`) ficam em `./outputs`.

## Interface web (playground)

```bash
docker compose up -d web
```

Abra http://localhost:8000. Na página você encontra:

- um chat com geração em streaming, caractere por caractere;
- controles de temperatura, top-k e tamanho da resposta;
- o modo **Diálogo**, que embrulha o prompt no formato de fala do livro (`--frase` + nova linha `--`). É um exemplo de *prompt engineering* em um modelo base;
- um painel com as **probabilidades do próximo caractere** enquanto você digita.

> Este é um **modelo base**: ele só continua texto. Perguntas e tarefas ("traduza…",
> "quanto é 2+2?") não são entendidas. Para isso seria preciso um *fine-tuning* com instruções
> (é o que transforma um GPT em um ChatGPT).

Para parar o servidor: `docker compose down`. Depois de retreinar, rode `docker compose restart web`.

## Avaliando o modelo em outros textos

```bash
docker compose run --rm evaluate
```

O comando mede a loss e a perplexidade (`exp(loss)`) do modelo em textos que ele nunca viu.
Resultado de um treino padrão:

| Texto | Perplexidade |
|---|---|
| Chute aleatório | 95 |
| Validação (final de Dom Casmurro) | 6,3 |
| Machado: *Memórias Póstumas* | 6,7 |
| Camões: *Os Lusíadas* (verso, português antigo) | 9,7 |
| Espanhol: *Dom Quixote* | 16,2 |
| Inglês: *Alice no País das Maravilhas* | 52,7 |

Quanto mais distante do texto de treino, pior o modelo vai. Ele aprendeu "português do Machado",
não "linguagem" em geral. Para avaliar outro arquivo:
`docker compose run --rm evaluate data/avaliacao/meu_texto.txt`.

## Experimentando com os parâmetros

Qualquer parâmetro do `config.yaml` pode ser sobrescrito na linha de comando, sem rebuild:

```bash
docker compose run --rm train --learning_rate 1e-2
```

Experimentos sugeridos:

| Experimento | Comando | O que observar |
|---|---|---|
| LR alto demais | `--learning_rate 1e-2` | loss oscila ou não cai |
| LR baixo demais | `--learning_rate 1e-5` | loss cai muito devagar |
| Modelo maior | `--n_layer 6 --n_embd 192 --n_head 6` | mais parâmetros, loss menor, treino mais lento |
| Modelo minúsculo | `--n_layer 1 --n_embd 32 --n_head 2` | o modelo não consegue aprender palavras |
| Contexto curto | `--block_size 8` | o texto perde coerência entre palavras |
| Forçar overfitting | `--dropout 0 --n_layer 8 --n_embd 256 --n_head 8 --max_iters 4000` | treino cai, validação volta a subir |
| Sem warmup | `--warmup_iters 0 --learning_rate 3e-3` | início instável |
| Temperatura (geração) | `generate --temperature 0.3` / `1.5` | repetitivo vs. caótico |

## Lendo a saída

```
passo  1000 | loss treino 1.8022 | loss val 2.1289 | lr 5.87e-04 | 133.2s
```

- **loss**: entropia cruzada do próximo caractere. Começa em ≈ ln(tamanho do vocabulário) ≈ 4,55
  (chute aleatório) e deve cair. ~1,8 já forma palavras; < 1,5 começa a parecer português de verdade.
- **loss treino vs. val**: a validação é o trecho final do livro, que o modelo nunca vê. Se a de
  treino continua caindo e a de validação **sobe**, o modelo está decorando (overfitting).
- **lr**: o learning rate do passo atual. Sobe no *warmup* e depois decai em cosseno até `min_lr`.

## De mini para "de verdade"

Mesmos conceitos, em outra escala: LLMs reais usam tokens de subpalavras (BPE) em vez de caracteres,
bilhões de parâmetros, contexto de milhares de tokens, trilhões de tokens de treino e muitas GPUs.
Se você tiver GPU NVIDIA, o `train.py` já usa `cuda` automaticamente (basta trocar a imagem do
PyTorch por uma com CUDA e liberar a GPU no compose).
