# Contributing

Este documento descreve como contribuir com o ALPR 2.0 sem quebrar o fluxo local, a documentação e a validação automatizada.

## Escopo

Contribuições são bem-vindas para:

- correção de bugs
- melhoria de detector, OCR, validação e vídeo
- testes automatizados
- documentação operacional
- ergonomia da interface Streamlit

Mudanças grandes de arquitetura, troca de dependências pesadas ou alterações de pipeline que mudem comportamento operacional devem vir acompanhadas de contexto técnico claro e impacto esperado.

## Requisitos

- Python 3.11+
- ambiente virtual local
- pelo menos um peso YOLO de placas em `models/yolo/` para subir o fluxo local

O repositório não inclui por padrão:

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

## Rodando a aplicação

```bash
streamlit run app.py
```

## Testes

Suíte padrão:

```bash
python -m pytest tests -q
```

Com verbosidade:

```bash
python -m pytest tests -v --tb=short
```

Testes marcados como `integration` ficam fora da execução padrão. Se precisar rodar explicitamente:

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

## Medindo o impacto de mudanças na leitura

Se a sua mudança toca detector, pré-processamento, OCR, validação, top-k ou
votação temporal, os testes unitários **não** dizem se a leitura melhorou ou
piorou. Meça:

```bash
python scripts/evaluate.py --manifest data/fixtures/manifest.json --report-name antes
```

Aplique a mudança e compare:

```bash
python scripts/evaluate.py --report-name depois --compare data/results/evaluation/antes.json
```

Inclua o delta de `exact_match` e `char_accuracy` na descrição do PR. Ver
[data/fixtures/README.md](data/fixtures/README.md) para montar o manifesto.

## Código legado mantido

Alguns trechos não têm chamador em produção, mas estão cobertos por testes e
foram **deliberadamente preservados** na limpeza de 2026-08-17. São candidatos
a remoção futura, e não devem ser tratados como exemplo do padrão atual:

| Local | Situação |
|---|---|
| `PaddleOCREngine._parse_paddle_output` | Parser do formato PaddleOCR ≤ 2.x. O runtime atual usa `_parse_paddlex_output` |
| `ImagePreprocessor._deskew_image` e a flag `deskew` | **Inertes.** A correção de inclinação migrou para `GeometricNormalizer`, que roda antes do preprocessador |
| `constants.FORENSIC_TERMS`, `CONFIDENCE_LEVELS` | Vocabulário forense sem consumidor |
| `constants.PLATE_POSITIONS_OLD` / `PLATE_POSITIONS_MERCOSUL` | Duplicam listas que `validator.py` declara inline |
| `OCRManager.get_status` | Nenhum ponto da UI o consome hoje |

Ao remover qualquer um deles, remova o teste correspondente no mesmo commit — a
cobertura existente é do código, não do comportamento do produto.

Em contrapartida, estes são **fontes únicas** e não devem ser duplicados:
`src/constants.py` concentra os regex de placa, os pesos de confusão OCR e as
faixas de prefixo por UF. Já existiram cópias divergentes desses dados em
`validator.py`, `plate_patterns.py`, `ocr/confidence.py` e `ocr/paddle_engine.py`,
e elas discordavam entre si.

## Pre-commit

Instalação:

```bash
pre-commit install
```

Execução manual:

```bash
pre-commit run --all-files
```

## Regras práticas para contribuir

- não versione `.env`, chaves, tokens ou credenciais
- não versione pesos grandes em `models/yolo/` nem mídia gerada em `data/results/`
- mantenha as mudanças focadas; evite misturar refatoração ampla com bugfix pequeno
- se o comportamento do pipeline mudar, atualize a documentação correspondente
- se uma mudança afetar heurística, OCR, top-k, vídeo ou Premium, inclua ou ajuste testes
- preserve o princípio atual do projeto: OCR principal local, Plate Recognizer opcional e Ollama opcional depois do top-k

## Pull requests

Antes de abrir PR, o mínimo esperado é:

1. a aplicação ainda subir localmente
2. os testes relevantes passarem
3. a documentação afetada estar alinhada
4. nenhum arquivo sensível ou artefato grande ter entrado no diff

Na descrição do PR, deixe claro:

- problema resolvido
- abordagem adotada
- risco de regressão
- comandos usados para validar

## Dicas para mídia e modelos

Se sua mudança depende de pesos, fixtures grandes ou vídeos reais:

- não envie o binário bruto para o repositório
- documente como reproduzir localmente
- use caminhos ignorados pelo Git para artefatos pesados

## Licença

Ao contribuir, você concorda em disponibilizar sua contribuição sob a mesma licença MIT usada neste projeto.