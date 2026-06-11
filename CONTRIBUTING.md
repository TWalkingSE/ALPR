# Contributing

Este documento descreve como contribuir com o ALPR 2.0 sem quebrar o fluxo local, a documentaÃ§Ã£o e a validaÃ§Ã£o automatizada.

## Escopo

ContribuiÃ§Ãµes sÃ£o bem-vindas para:

- correÃ§Ã£o de bugs
- melhoria de detector, OCR, validaÃ§Ã£o e vÃ­deo
- testes automatizados
- documentaÃ§Ã£o operacional
- ergonomia da interface Streamlit

MudanÃ§as grandes de arquitetura, troca de dependÃªncias pesadas ou alteraÃ§Ãµes de pipeline que mudem comportamento operacional devem vir acompanhadas de contexto tÃ©cnico claro e impacto esperado.

## Requisitos

- Python 3.11+
- ambiente virtual local
- pelo menos um peso YOLO de placas em `models/yolo/` para subir o fluxo local

O repositÃ³rio nÃ£o inclui por padrÃ£o:

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

## Rodando a aplicaÃ§Ã£o

```bash
streamlit run app.py
```

## Testes

SuÃ­te padrÃ£o:

```bash
python -m pytest tests -q
```

Com verbosidade:

```bash
python -m pytest tests -v --tb=short
```

Testes marcados como `integration` ficam fora da execuÃ§Ã£o padrÃ£o. Se precisar rodar explicitamente:

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

InstalaÃ§Ã£o:

```bash
pre-commit install
```

ExecuÃ§Ã£o manual:

```bash
pre-commit run --all-files
```

## Regras prÃ¡ticas para contribuir

- nÃ£o versione `.env`, chaves, tokens ou credenciais
- nÃ£o versione pesos grandes em `models/yolo/` nem mÃ­dia gerada em `data/results/`
- mantenha as mudanÃ§as focadas; evite misturar refatoraÃ§Ã£o ampla com bugfix pequeno
- se o comportamento do pipeline mudar, atualize a documentaÃ§Ã£o correspondente
- se uma mudanÃ§a afetar heurÃ­stica, OCR, top-k, vÃ­deo ou Premium, inclua ou ajuste testes
- preserve o princÃ­pio atual do projeto: OCR principal local, Plate Recognizer opcional e Ollama opcional depois do top-k

## Pull requests

Antes de abrir PR, o mÃ­nimo esperado Ã©:

1. a aplicaÃ§Ã£o ainda subir localmente
2. os testes relevantes passarem
3. a documentaÃ§Ã£o afetada estar alinhada
4. nenhum arquivo sensÃ­vel ou artefato grande ter entrado no diff

Na descriÃ§Ã£o do PR, deixe claro:

- problema resolvido
- abordagem adotada
- risco de regressÃ£o
- comandos usados para validar

## Dicas para mÃ­dia e modelos

Se sua mudanÃ§a depende de pesos, fixtures grandes ou vÃ­deos reais:

- nÃ£o envie o binÃ¡rio bruto para o repositÃ³rio
- documente como reproduzir localmente
- use caminhos ignorados pelo Git para artefatos pesados

## LicenÃ§a

Ao contribuir, vocÃª concorda em disponibilizar sua contribuiÃ§Ã£o sob a mesma licenÃ§a MIT usada neste projeto.
