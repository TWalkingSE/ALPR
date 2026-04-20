# Validacao operacional do release 2.0

Este documento registra o que foi validado no workspace para o fechamento do ALPR 2.0 e o que ainda depende de midia real, fixtures representativas ou credenciais externas.

## O que esta consolidado

- `app.py` e o entrypoint definitivo da aplicacao.
- O fluxo local de imagem segue o pipeline atual: deteccao YOLOv11, fallback com SAHI quando necessario, recorte, normalizacao geometrica, preprocessamento, PaddleOCR e validacao.
- O fluxo local de video usa agregacao temporal, ranking de placas e pode gerar video anotado.
- A votacao temporal permanece integrada ao processamento de video para consolidar leituras entre frames.
- O fluxo Premium continua isolado em `src/premium_alpr.py`, encapsulado para a interface por `src/v2/premium.py`, e recebe a imagem completa.
- A chave Premium nao e mais exposta na interface; ela deve ser lida do `.env` via `PLATE_RECOGNIZER_API_KEY`.
- A UI continua organizada em `src/v2/ui/`.
- A camada de aplicacao e estado continua em `src/v2/application.py`, `src/v2/state.py` e `src/v2/contracts.py`.

## O que foi validado no workspace

- A suite automatizada completa passou com `350 passed` usando `python -m pytest`.
- Os testes focados apos os ultimos ajustes de UI Premium e OCR tambem passaram no workspace.
- O app principal subiu em modo headless com `streamlit run app.py --server.headless true --server.fileWatcherType none`.
- O detector continua com deteccao padrao na imagem inteira e usa SAHI como retry em imagens grandes quando nao ha deteccoes, quando a confianca esta baixa ou quando a maior deteccao ainda parece pequena demais no quadro.
- O OCR local continua funcional com variantes de preprocessamento, tratamento correto de entradas grayscale no adapter do PaddleOCR e ajustes adaptativos orientados por qualidade, SNR e motion blur.
- A arvore do projeto foi reduzida ao release 2.0, com remocao dos diretorios legados vazios de `models/crnn` e `models/super_resolution`.

## O que ainda depende de ambiente externo

1. Fixtures reais de imagem e video.
O repositorio ainda nao inclui um conjunto proprio e representativo em `data/fixtures/` para medir acuracia, latencia e estabilidade em cenarios reais.

2. Credenciais validas da API Premium.
Para validar o Plate Recognizer em ambiente real, ainda e necessario definir `PLATE_RECOGNIZER_API_KEY` no `.env` com uma chave funcional e com cota disponivel.

3. Comparacao controlada com amostras reais.
A comparacao manual entre o pipeline local e o Premium ainda depende de um conjunto de referencia externo, com placas conhecidas e diversidade de iluminacao, distancia e angulo.

4. Validacao visual do video anotado.
Embora a geracao de saida anotada esteja integrada ao pipeline, a inspecao final de legibilidade das caixas, textos e continuidade das anotacoes ainda depende de video real.

## Checklist sugerido

1. Separar um pequeno conjunto de imagens reais de placa em `data/fixtures/images/`.
2. Separar ao menos um video curto em `data/fixtures/videos/`.
3. Rodar `streamlit run app.py` e validar os tres fluxos obrigatorios: imagem local, imagem Premium e video local.
4. Confirmar se o video anotado permanece legivel em cenas com `skip_frames` alto e em modo `stationary` com early-stop.
5. Registrar divergencias de leitura, latencia e confianca em relacao ao baseline escolhido.