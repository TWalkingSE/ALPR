# Contributing

Este documento descreve como contribuir com o ALPR 2.0 sem quebrar o fluxo local, a documentacao e a validacao automatizada.

## Escopo

Contribuicoes sao bem-vindas para:

- correcao de bugs
- melhoria de detector, OCR, validacao e video
- testes automatizados
- documentacao operacional
- ergonomia da interface Streamlit

Mudancas grandes de arquitetura, troca de dependencias pesadas ou alteracoes de pipeline que mudem comportamento operacional devem vir acompanhadas de contexto tecnico claro e impacto esperado.

## Requisitos

- Python 3.11+
- ambiente virtual local
- pelo menos um peso YOLO de placas em `models/yolo/` para subir o fluxo local

O repositório nao inclui por padrao:

- `.env` com chaves privadas
- pesos YOLO locais `.pt`
- resultados gerados em `data/results/`

## Setup local

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
```

Se preferir, use:

```bash
python install_dependencies.py
```

Se quiser usar o fluxo Premium, copie `.env.example` para `.env` e configure `PLATE_RECOGNIZER_API_KEY`.

## Rodando a aplicacao

```bash
streamlit run app.py
```

## Testes

Suite padrao:

```bash
python -m pytest tests -q
```

Com verbosidade:

```bash
python -m pytest tests -v --tb=short
```

Testes marcados como `integration` ficam fora da execucao padrao. Se precisar rodar explicitamente:

```bash
python -m pytest -m integration
```

## Lint, formato e type check

Lint:

```bash
ruff check src/ tests/
```

Formato:

```bash
ruff format src/ tests/
```

Type check:

```bash
mypy src/ --ignore-missing-imports
```

## Pre-commit

Instalacao:

```bash
pre-commit install
```

Execucao manual:

```bash
pre-commit run --all-files
```

## Regras praticas para contribuir

- nao versione `.env`, chaves, tokens ou credenciais
- nao versione pesos grandes em `models/yolo/` nem midia gerada em `data/results/`
- mantenha as mudancas focadas; evite misturar refatoracao ampla com bugfix pequeno
- se o comportamento do pipeline mudar, atualize a documentacao correspondente
- se uma mudanca afetar heuristica, OCR, top-k, video ou Premium, inclua ou ajuste testes
- preserve o principio atual do projeto: OCR principal local, Plate Recognizer opcional e Ollama opcional depois do top-k

## Pull requests

Antes de abrir PR, o minimo esperado e:

1. a aplicacao ainda subir localmente
2. os testes relevantes passarem
3. a documentacao afetada estar alinhada
4. nenhum arquivo sensivel ou artefato grande ter entrado no diff

Na descricao do PR, deixe claro:

- problema resolvido
- abordagem adotada
- risco de regressao
- comandos usados para validar

## Dicas para midia e modelos

Se sua mudanca depende de pesos, fixtures grandes ou videos reais:

- nao envie o binario bruto para o repositório
- documente como reproduzir localmente
- use caminhos ignorados pelo Git para artefatos pesados

## Licenca

Ao contribuir, voce concorda em disponibilizar sua contribuicao sob a mesma licenca MIT usada neste projeto.