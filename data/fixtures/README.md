# Fixtures versionadas

Use esta pasta para manter amostras rotuladas de regressão do ALPR 2.0.

Estrutura:

- `images/`: imagens individuais de placa
- `videos/`: vídeos curtos para consolidação temporal
- `manifest.template.json`: modelo versionado (não é usado diretamente)
- `manifest.json`: **o manifesto real que os scripts consomem** — você cria

## Como criar o manifesto

```bash
cp data/fixtures/manifest.template.json data/fixtures/manifest.json
```

Edite `manifest.json` com as suas amostras. Cada entrada aceita:

| Campo | Obrigatório | Descrição |
|---|---|---|
| `id` | sim | Identificador estável da fixture |
| `path` | sim | Caminho do arquivo, relativo a esta pasta |
| `expected_plate` | sim | Placa correta (o gabarito) |
| `media_type` | não | `image` (padrão) ou `video` |
| `expected_format` | não | `old` ou `mercosul` |
| `scenario_tags` | não | Ex.: `low_light`, `small_plate` — geram o breakdown por cenário |
| `notes` | não | Anotação livre |

As imagens e vídeos **não são versionados** (ver `.gitignore`); apenas o
manifesto e esta estrutura são. Mantenha as mídias em um armazenamento
compartilhado da equipe.

## Para que servem

```bash
python scripts/evaluate.py --manifest data/fixtures/manifest.json
```

Mede exact match, acurácia por caractere, taxa de falso positivo e latência —
inclusive quebrados por `scenario_tags`. Rode antes e depois de qualquer
mudança que possa afetar a leitura e compare com `--compare`.

```bash
python scripts/calibrate.py --manifest data/fixtures/manifest.json
```

Varre as combinações de threshold declaradas em `calibration:` no `config.yaml`
e mostra qual delas pontua melhor nas suas fixtures.

## Quantas amostras?

Métricas sobre menos de ~30 fixtures oscilam demais para servir de baseline.
Priorize diversidade — iluminação, distância, ângulo, placa antiga e Mercosul,
moto — sobre volume, e inclua os casos que hoje falham: são eles que medem
progresso.