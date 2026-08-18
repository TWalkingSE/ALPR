# Validação operacional do release 2.0

> **Atualização — revisão de código de 2026-08-17.** Correções de bugs, remoção
> de código morto, separação entre configuração de construção e de runtime, e
> três capacidades novas (avaliação por fixtures, modo headless, histórico
> persistente). Ver a seção [Revisão de 2026-08-17](#revisão-de-2026-08-17) no
> fim deste documento. Suíte: `486 passed`.

Este documento registra o que foi validado no workspace para o fechamento do ALPR 2.0 e o que ainda depende de mídia real, fixtures representativas ou credenciais externas.

## O que está consolidado

- `app.py` é o entrypoint definitivo da aplicação.
- O fluxo local de imagem segue o pipeline atual: detecção YOLOv11, fallback com SAHI quando necessário, recorte, normalização geométrica, preprocessamento, PaddleOCR e validação.
- O fluxo local de vídeo usa agregação temporal, ranking de placas e pode gerar vídeo anotado.
- A votação temporal permanece integrada ao processamento de vídeo para consolidar leituras entre frames.
- O fluxo Premium continua isolado em `src/premium_alpr.py`, encapsulado para a interface por `src/v2/premium.py`, e recebe a imagem completa.
- A chave Premium não é mais exposta na interface; ela deve ser lida do `.env` via `PLATE_RECOGNIZER_API_KEY`.
- A UI continua organizada em `src/v2/ui/`.
- A camada de aplicação e estado continua em `src/v2/application.py`, `src/v2/state.py` e `src/v2/contracts.py`.

## O que foi validado no workspace

- A suíte automatizada completa passou com `350 passed` usando `python -m pytest`.
- Os testes focados após os últimos ajustes de UI Premium e OCR também passaram no workspace.
- O app principal subiu em modo headless com `streamlit run app.py --server.headless true --server.fileWatcherType none`.
- O detector continua com detecção padrão na imagem inteira e usa SAHI como retry em imagens grandes quando não há detecções, quando a confiança está baixa ou quando a maior detecção ainda parece pequena demais no quadro.
- O OCR local continua funcional com variantes de preprocessamento, tratamento correto de entradas grayscale no adapter do PaddleOCR e ajustes adaptativos orientados por qualidade, SNR e motion blur.
- A árvore do projeto foi reduzida ao release 2.0, com remoção dos diretórios legados vazios de `models/crnn` e `models/super_resolution`.

## O que ainda depende de ambiente externo

1. Fixtures reais de imagem e vídeo.
O repositório ainda não inclui um conjunto próprio e representativo em `data/fixtures/` para medir acurácia, latência e estabilidade em cenários reais.

2. Credenciais válidas da API Premium.
Para validar o Plate Recognizer em ambiente real, ainda é necessário definir `PLATE_RECOGNIZER_API_KEY` no `.env` com uma chave funcional e com cota disponível.

3. Comparação controlada com amostras reais.
A comparação manual entre o pipeline local e o Premium ainda depende de um conjunto de referência externo, com placas conhecidas e diversidade de iluminação, distância e ângulo.

4. Validação visual do vídeo anotado.
Embora a geração de saída anotada esteja integrada ao pipeline, a inspeção final de legibilidade das caixas, textos e continuidade das anotações ainda depende de vídeo real.

## Checklist sugerido

1. Separar um pequeno conjunto de imagens reais de placa em `data/fixtures/images/`.
2. Separar ao menos um vídeo curto em `data/fixtures/videos/`.
3. Rodar `streamlit run app.py` e validar os três fluxos obrigatórios: imagem local, imagem Premium e vídeo local.
4. Confirmar se o vídeo anotado permanece legível em cenas com `skip_frames` alto e em modo `stationary` com early-stop.
5. Registrar divergências de leitura, latência e confiança em relação ao baseline escolhido.

---

## Revisão de 2026-08-17

### Bugs corrigidos

1. **`UnboundLocalError` mascarando erros no vídeo** (`src/video_processor.py`).
O bloco `finally` referenciava `video_writer` antes da atribuição. Uma falha ao
ler os metadados do vídeo estourava no `finally` e engolia a exceção original,
tornando o erro real indiagnosticável.

2. **Regra falsa enviada ao LLM** (`src/v2/ollama_validation.py`). O prompt
afirmava que a 5ª posição da placa Mercosul não pode ser vogal — o oposto do que
o próprio validador do projeto documenta. O modelo era instruído a descartar
candidatos válidos.

3. **Limiar de OCR por cenário era inerte** (`src/v2/pipeline.py`). O
`effective_ocr_threshold` era calculado e ajustado para `low_light`,
`small_plate` e `low_snr`, mas nenhuma decisão o consultava: toda a
configuração `scenarios.*.ocr_confidence_threshold` não tinha efeito. Agora
marca a leitura com o aviso `below_ocr_threshold`.

4. **`_detect_format` executado e descartado**. Usado como default de
`dict.get()`, que o Python avalia sempre — e a chave nunca faltava.

5. **Drift entre os defaults do código e o `config.yaml`**. Faltavam
`vehicle_attributes`, `video.confidence_threshold` e a zona de silêncio do
PaddleOCR. Travado por `tests/test_config_defaults_match_yaml.py`.

6. **`rank_unique_plates` mutava a entrada**, sendo chamada a cada rerun do
Streamlit.

7. **Laudo com tempo zerado**. `report_payload` copiava
`processing_time_ms` antes de ele ser medido. Descoberto pela verificação
end-to-end, não pela suíte.

### Duplicação eliminada

As faixas de prefixo por estado existiam em **duas cópias divergentes**
(`validator.py` e `plate_patterns.py`) — AP, AM, PE, RS e CE tinham faixas
diferentes — e ambas alimentavam o mesmo score de ranking. Consolidadas em
`src/constants.py` como união das duas. Os regex de placa, que tinham três
cópias, também foram unificados.

### Código morto removido

Protocolo de self-consistency de engines OCR que não existem mais,
`extract_best_frames`, wrappers de compatibilidade sem chamadores no validador,
métodos públicos nunca consumidos do `PlateNgramModel`, protocolos `Detector` e
`OCREngine` não usados, campos fantasma do `OCRResult`, e o hook
`structured_logger` que chamava um método privado de uma classe inexistente.

`Dicas.md` foi movido para `docs/legado/Dicas-v1.md` com aviso de obsolescência:
descrevia uma versão anterior da aplicação e contradizia o README.

### Performance

`AppConfig.signature()` tinha ~90 campos, e qualquer slider da sidebar —
inclusive o limiar que só decide se um PNG vai para o disco — recarregava o
modelo YOLO e reinicializava o PaddleOCR. Agora a assinatura cobre apenas a
identidade dos modelos; o resto é aplicado no pipeline vivo por
`LocalAnalysisPipeline.apply_runtime_config`. Travado por
`tests/test_runtime_config.py`.

A verificação de conectividade do Plate Recognizer (HTTP bloqueante, 5 s) saiu
do construtor e virou uma property preguiçosa com cache.

`_try_correction` do validador passou a ser memoizado — era invocado várias
vezes sobre os mesmos textos por candidato do top-k.

### Medição de performance (CPU)

O custo é dominado pelo OCR: ~100 s dos ~100 s totais por placa, escalando
linearmente com `ocr.max_variants` (~20 s por variante). Pré-processamento são
38 ms. **Reduzir variantes corta latência mas custa acurácia** — em teste, 5
variantes leram a placa completa e 3 leram apenas parte dela. Ver a seção
"Notas de performance" do README.

### Capacidades novas

- `scripts/evaluate.py` — baseline de acurácia sobre fixtures rotuladas
  (exact match, acurácia por caractere, falso positivo, latência, breakdown por
  cenário), com `--compare` para delta entre versões e `--fail-under` para CI.
  A infraestrutura de avaliação já existia em `src/v2/evaluation.py`, mas nada
  no projeto a executava.
- `scripts/calibrate.py` — grid-search dos thresholds declarados em
  `calibration:`, reaplicando a configuração sem recarregar os modelos.
- `scripts/alpr_cli.py` — leitura em lote headless (imagem, vídeo ou
  diretório), saída em texto/JSON/CSV.
- `scripts/api.py` — API HTTP opcional (FastAPI), extra `pip install -e ".[api]"`.
- `src/v2/storage.py` — histórico persistente em SQLite (stdlib) com busca por
  placa, detecção de reprocessamento via `sha256` e a aba **Histórico** na
  interface. Desligado por padrão (`storage.enabled`).

### Verificação executada

- Suíte automatizada: `486 passed`.
- `scripts/evaluate.py` end-to-end com o peso `yolo11l-plate.pt` e PaddleOCR
  reais: leitura correta, relatórios JSON/CSV gerados, `--compare` funcional.
- `scripts/alpr_cli.py` em arquivo único e em diretório, com saída CSV.
- API: `GET /v1/health`, `POST /v1/plates` e `GET /v1/plates/{placa}` — a
  cadeia pipeline → histórico → API foi validada de ponta a ponta.
- `streamlit run app.py` sobe em modo headless e responde HTTP 200.

### O que ainda depende de mídia real

O item 1 da lista original — fixtures representativas — **continua sendo a
lacuna principal**, mas agora existe a ferramenta para explorá-las:
`scripts/evaluate.py` e `scripts/calibrate.py`. Monte
`data/fixtures/manifest.json` seguindo `data/fixtures/README.md` e gere a
primeira baseline; sem ela, nenhuma mudança que afete a leitura pode ser
avaliada objetivamente.
