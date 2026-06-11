# ValidaÃ§Ã£o operacional do release 2.0

Este documento registra o que foi validado no workspace para o fechamento do ALPR 2.0 e o que ainda depende de mÃ­dia real, fixtures representativas ou credenciais externas.

## O que estÃ¡ consolidado

- `app.py` Ã© o entrypoint definitivo da aplicaÃ§Ã£o.
- O fluxo local de imagem segue o pipeline atual: detecÃ§Ã£o YOLOv11, fallback com SAHI quando necessÃ¡rio, recorte, normalizaÃ§Ã£o geomÃ©trica, preprocessamento, PaddleOCR e validaÃ§Ã£o.
- O fluxo local de vÃ­deo usa agregaÃ§Ã£o temporal, ranking de placas e pode gerar vÃ­deo anotado.
- A votaÃ§Ã£o temporal permanece integrada ao processamento de vÃ­deo para consolidar leituras entre frames.
- O fluxo Premium continua isolado em `src/premium_alpr.py`, encapsulado para a interface por `src/v2/premium.py`, e recebe a imagem completa.
- A chave Premium nÃ£o Ã© mais exposta na interface; ela deve ser lida do `.env` via `PLATE_RECOGNIZER_API_KEY`.
- A UI continua organizada em `src/v2/ui/`.
- A camada de aplicaÃ§Ã£o e estado continua em `src/v2/application.py`, `src/v2/state.py` e `src/v2/contracts.py`.

## O que foi validado no workspace

- A suÃ­te automatizada completa passou com `350 passed` usando `python -m pytest`.
- Os testes focados apÃ³s os Ãºltimos ajustes de UI Premium e OCR tambÃ©m passaram no workspace.
- O app principal subiu em modo headless com `streamlit run app.py --server.headless true --server.fileWatcherType none`.
- O detector continua com detecÃ§Ã£o padrÃ£o na imagem inteira e usa SAHI como retry em imagens grandes quando nÃ£o hÃ¡ detecÃ§Ãµes, quando a confianÃ§a estÃ¡ baixa ou quando a maior detecÃ§Ã£o ainda parece pequena demais no quadro.
- O OCR local continua funcional com variantes de preprocessamento, tratamento correto de entradas grayscale no adapter do PaddleOCR e ajustes adaptativos orientados por qualidade, SNR e motion blur.
- A Ã¡rvore do projeto foi reduzida ao release 2.0, com remoÃ§Ã£o dos diretÃ³rios legados vazios de `models/crnn` e `models/super_resolution`.

## O que ainda depende de ambiente externo

1. Fixtures reais de imagem e vÃ­deo.
O repositÃ³rio ainda nÃ£o inclui um conjunto prÃ³prio e representativo em `data/fixtures/` para medir acurÃ¡cia, latÃªncia e estabilidade em cenÃ¡rios reais.

2. Credenciais vÃ¡lidas da API Premium.
Para validar o Plate Recognizer em ambiente real, ainda Ã© necessÃ¡rio definir `PLATE_RECOGNIZER_API_KEY` no `.env` com uma chave funcional e com cota disponÃ­vel.

3. ComparaÃ§Ã£o controlada com amostras reais.
A comparaÃ§Ã£o manual entre o pipeline local e o Premium ainda depende de um conjunto de referÃªncia externo, com placas conhecidas e diversidade de iluminaÃ§Ã£o, distÃ¢ncia e Ã¢ngulo.

4. ValidaÃ§Ã£o visual do vÃ­deo anotado.
Embora a geraÃ§Ã£o de saÃ­da anotada esteja integrada ao pipeline, a inspeÃ§Ã£o final de legibilidade das caixas, textos e continuidade das anotaÃ§Ãµes ainda depende de vÃ­deo real.

## Checklist sugerido

1. Separar um pequeno conjunto de imagens reais de placa em `data/fixtures/images/`.
2. Separar ao menos um vÃ­deo curto em `data/fixtures/videos/`.
3. Rodar `streamlit run app.py` e validar os trÃªs fluxos obrigatÃ³rios: imagem local, imagem Premium e vÃ­deo local.
4. Confirmar se o vÃ­deo anotado permanece legÃ­vel em cenas com `skip_frames` alto e em modo `stationary` com early-stop.
5. Registrar divergÃªncias de leitura, latÃªncia e confianÃ§a em relaÃ§Ã£o ao baseline escolhido.
