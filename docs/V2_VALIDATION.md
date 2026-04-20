# Validação operacional do release 2.0

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
