# ALPR 2.0

ALPR 2.0 e um projeto de leitura de placas focado em execucao local, previsibilidade operacional e diagnostico. O caminho principal roda offline com YOLOv11 + OpenCV + PaddleOCR. O caminho Premium via Plate Recognizer existe como comparacao manual, nao como fallback automatico. A integracao com Ollama e opcional e aparece apenas depois do top-k como desempate controlado.

## Resumo rapido

- pipeline local principal para imagem e video
- OCR principal local e deterministico com PaddleOCR
- SAHI, normalizacao, preprocessamento adaptativo e votacao temporal
- Plate Recognizer opcional para comparacao manual
- Ollama opcional apenas depois do top-k
- Python 3.11+, interface Streamlit e licenca MIT
- validacao automatizada mais recente do workspace: `350 passed`

## Inicio rapido

Se voce so quer colocar o projeto de pe em um clone limpo:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
# coloque um peso YOLO .pt em models/yolo/
streamlit run app.py
```

Antes de rodar, tenha em mente:

- o repositorio publico nao inclui `.env`, pesos YOLO locais nem resultados gerados
- sem um arquivo `.pt` valido em `models/yolo/`, o fluxo local nao inicializa
- `PLATE_RECOGNIZER_API_KEY` so e necessario para o fluxo Premium
- Ollama continua opcional e fora do caminho principal do OCR

## Navegacao rapida

- [O que o projeto entrega](#o-que-o-projeto-entrega)
- [Limites praticos](#limites-praticos)
- [Fluxos da aplicacao](#fluxos-da-aplicacao)
- [Instalacao detalhada](#instalacao-detalhada)
- [Bootstrap de um clone limpo](#bootstrap-de-um-clone-limpo)
- [Contributing](CONTRIBUTING.md)
- [Validacao](#validacao)
- [Licenca](#licenca)

## O que o projeto entrega

- Analise local de imagem com detector de placas, normalizacao geometrica, preprocessamento, OCR e validacao.
- Validacao inteligente opcional via Ollama, usada apenas como desempate depois do ranking top-k.
- Analise local de video com agregacao entre frames, ranking de placas e saida anotada.
- Comparacao lado a lado entre o pipeline local e o Plate Recognizer.
- Captura opcional de artefatos para casos invalidos ou de baixa confianca.
- Infraestrutura de fixtures, baseline e calibracao para validacao regressiva.

## Limites praticos

O projeto melhora a chance de leitura com detector, normalizacao, preprocessamento, OCR, validacao e agregacao temporal, mas ele continua dependente da qualidade real da imagem ou do video. Em outras palavras: o projeto nao faz milagre.

Se a entrada vier ruim demais, o resultado pode nao existir, pode ficar abaixo do threshold esperado ou pode terminar apenas como candidato fraco. O comportamento correto nesses cenarios nao e inventar uma placa com confianca artificial.

Isso vale principalmente para casos como:

- placa muito pequena no quadro
- poucos pixels uteis na regiao da placa, mesmo quando a deteccao acontece
- desfoque por movimento
- foco ruim, lente suja ou vibracao da camera
- baixa iluminacao, contraluz ou excesso de brilho
- compressao forte, ruido, chuva ou reflexos
- zoom digital agressivo, frame muito comprimido ou bitrate baixo
- obstrucao parcial, sujeira, para-choque cobrindo caracteres ou angulo extremo
- placa amassada, tipografia degradada ou caracteres fisicamente ilegiveis

Na pratica, quando falta informacao visual suficiente, nenhum ajuste de preprocessamento, SAHI, top-k, Ollama ou comparacao com fluxo Premium consegue recuperar detalhes que nao existem no arquivo de entrada.

O que normalmente acontece nesses casos:

- o detector pode nao encontrar placa alguma
- o OCR pode ler apenas parte dos caracteres
- o validador pode rejeitar a leitura por formato inconsistente
- o pipeline pode manter varias alternativas proximas sem confianca para desempate
- o resultado final pode ser vazio, invalido ou abaixo do threshold configurado

Em video, a agregacao entre frames ajuda quando existem alguns quadros aproveitaveis. Ela nao recupera detalhe que nunca apareceu de forma legivel em nenhum frame. Se o video inteiro estiver ruim, muito comprimido, tremido ou distante, o comportamento esperado e nao haver leitura confiavel.

Regra pratica: para o sistema funcionar bem, a placa precisa aparecer com tamanho razoavel, contraste suficiente e pelo menos alguns frames ou imagens realmente legiveis. Quando isso nao acontece, a saida mais honesta do sistema e baixa confianca ou ausencia de leitura.

Se a meta operacional for aumentar acerto no mundo real, o maior ganho quase sempre vem da captura, nao do pos-processamento:

- aproximar mais a camera ou usar enquadramento em que a placa ocupe mais pixels
- reduzir blur com shutter melhor, estabilizacao ou menor velocidade relativa
- melhorar iluminacao e evitar reflexo direto
- preservar bitrate e resolucao em video, evitando compressao excessiva
- selecionar imagens e frames em que a placa esteja frontal ou pouco inclinada

## Fluxos da aplicacao

### Fluxo local

O fluxo local e orquestrado por `src/v2/pipeline.py` e segue esta ordem:

1. detectar placas na imagem completa
2. recortar os crops detectados
3. normalizar geometricamente o crop
4. preprocessar o crop
5. rodar OCR
6. validar o texto lido
7. rankear alternativas quando a leitura ainda estiver fraca
8. opcionalmente consultar o Ollama para desempate inteligente quando ainda houver ambiguidade suficiente
9. salvar artefatos diagnosticos, se habilitado

### Fluxo Premium

O fluxo Premium usa `src/premium_alpr.py` e envia a imagem completa para a API da Plate Recognizer.

Ele serve para comparacao e investigacao, nao para substituir automaticamente a leitura local.

## Detector

O detector local fica em `src/detector.py` e usa YOLOv11 treinado para placas.

Na pratica, ele faz o seguinte:

- garante que a imagem tenha 3 canais BGR antes da inferencia
- roda uma primeira inferencia na imagem inteira
- se `enable_sahi: true`, pode tentar uma segunda passada com SAHI em imagens grandes quando nao ha deteccoes, quando a confianca padrao esta baixa ou quando a maior deteccao ainda parece pequena demais no quadro
- aplica margem adaptativa no recorte da placa
- faz upscale automatico em crops muito pequenos para melhorar o OCR

### O que e SAHI

SAHI significa `Sliced Aided Hyper Inference`.

Em vez de rodar o detector so na imagem inteira, a imagem e dividida em blocos sobrepostos. O YOLO e executado em cada bloco, e as deteccoes repetidas nas regioes de sobreposicao sao unificadas depois por NMS.

No ALPR 2.0, o SAHI nao roda o tempo todo. Ele entra como segunda tentativa quando:

- `models.detector.enable_sahi` esta ligado
- a imagem e grande o suficiente para justificar slicing
- a deteccao padrao nao encontrou nenhuma placa
- a melhor deteccao da passada padrao ficou abaixo do limiar configurado de confianca
- a maior deteccao ainda ocupa area muito pequena no quadro, sugerindo placa distante

Quando a passada SAHI encontra algo util, o pipeline combina as deteccoes padrao e as sliced com NMS para evitar duplicatas.

Isso ajuda principalmente em:

- placas pequenas
- placas distantes
- imagens de camera de vigilancia ou rodovia
- cenas em que a placa ocupa poucos pixels no quadro

Configuracoes relevantes em `config.yaml`:

- `models.detector.enable_sahi`
- `models.detector.sahi_slice_size`
- `models.detector.sahi_overlap_ratio`
- `models.detector.sahi_retry_confidence_threshold`
- `models.detector.sahi_retry_area_ratio_threshold`
- `models.detector.sahi_retry_large_image_threshold`
- `models.detector.sahi_merge_iou_threshold`

## Normalizacao geometrica

A normalizacao fica em `src/geometric_normalizer.py`.

Ela entra entre o detector e o OCR e tenta transformar o crop da placa em uma imagem mais retificada.

O modulo sabe fazer:

- deteccao aproximada dos 4 cantos da placa
- transformacao de perspectiva
- correcao de rotacao
- equalizacao de contraste
- redimensionamento padronizado

No pipeline v2 atual, o normalizador e instanciado com:

- correcao de perspectiva ativa
- correcao de rotacao ativa
- redimensionamento padronizado ativo
- equalizacao de contraste desativada

A equalizacao de contraste foi deixada para o preprocessador, para nao duplicar etapas de contraste no mesmo crop.

## Preprocessamento

O preprocessamento fica em `src/preprocessor.py`.

Ele trabalha sobre o crop ja normalizado e pode gerar varias versoes da mesma placa para aumentar a chance do OCR acertar em cenarios dificeis.

O que o preprocessador faz:

- converte para grayscale quando necessario
- faz upscale quando a placa esta muito pequena para OCR
- aplica CLAHE adaptativo para melhorar contraste
- remove ruido com `fastNlMeansDenoising` ou bilateral, com reforco extra quando o SNR esta baixo
- aplica nitidez via unsharp mask
- aplica um passo extra de reforco quando o crop indica motion blur alto
- gera threshold adaptativo gaussiano como binarizacao principal
- quando habilitado, gera variantes extras com Otsu, Mean e versoes invertidas e nao invertidas
- aplica otimizacoes especificas para placas brasileiras, incluindo tentativas para Mercosul e formato antigo
- gera pequenas rotacoes e ajustes de gamma quando o modo adaptativo entende que a imagem precisa disso

O preprocessador tambem ajusta sua agressividade pela qualidade estimada do crop:

- imagem excelente: menos variantes, sem augmentation
- imagem suficiente: fluxo padrao
- imagem critica: fluxo padrao com augmentation
- imagem insuficiente: sharpen mais forte e mais tentativas

Esse ajuste nao depende apenas do score global. O preprocessor tambem reage a sinais objetivos como `snr` baixo e `motion_blur` alto para aumentar denoising, sharpening e numero de variantes quando necessario.

## O que acontece quando nao testamos multiplas variantes

Esse ponto e importante.

No projeto, `ocr.try_multiple_variants` controla duas coisas ao mesmo tempo:

- o preprocessador deixa de gerar o bloco extra de multiplas binarizacoes
- o `OCRManager` deixa de iterar sobre `preprocessed_variants`

Na pratica, quando `ocr.try_multiple_variants: false`, o OCR roda apenas sobre a imagem normalizada principal. As saidas extras do preprocessamento deixam de participar da decisao do OCR.

Ou seja:

- com `true`: o OCR pode testar varias versoes da placa e escolher a melhor
- com `false`: o OCR fica mais rapido e mais deterministico, mas abre mao das tentativas extras

Configuracoes relevantes:

- `ocr.try_multiple_variants`
- `ocr.max_variants`

## OCR e validacao

O OCR local usa `PaddleOCR` via `src/ocr/paddle_engine.py`, encapsulado pelo `OCRManager` em `src/ocr/manager.py`.

Ele continua sendo o OCR principal do projeto. Mesmo com a opcao de Ollama disponivel, a leitura primaria e local, classica e deterministica.

Depois da leitura, o texto passa por:

- limpeza do texto bruto
- reconstrucoes de confianca por caractere
- validacao de formato em `src/validator.py`
- ranking de alternativas no pipeline quando a leitura ainda esta abaixo do threshold esperado

O projeto trabalha com thresholds diferentes para OCR e fallback, e esses limiares podem ser flexibilizados por contexto, como baixa iluminacao e placa pequena.

### Correcao de orientacao do texto

O PaddleOCR pode rodar com correcao de orientacao de texto quando `ocr.paddle.use_angle_cls: true`.

Na pratica, isso habilita uma classificacao de orientacao da linha de texto antes do reconhecimento. Essa etapa ajuda quando o crop chega ao OCR com a linha da placa girada ou invertida o suficiente para atrapalhar a leitura.

Esse recurso e complementar ao normalizador geometrico:

- a normalizacao geometrica corrige perspectiva e rotacao do crop da placa
- a correcao de orientacao do texto atua no nivel da linha de texto dentro do OCR

Ela nao substitui a retificacao da placa. O caminho esperado continua sendo: primeiro normalizar o crop, depois deixar o OCR refinar a orientacao do texto se necessario.

Na maioria dos casos de placas BR, vale manter ligado. Se voce quiser reduzir custo e maximizar previsibilidade em entradas ja muito bem normalizadas, pode desligar.

Configuracao relevante:

- `ocr.paddle.use_angle_cls`

### Validacao inteligente opcional via Ollama

O projeto tambem pode usar Ollama para uma validacao inteligente opcional, mas ele nao entra como OCR principal.

O comportamento correto e este:

- o PaddleOCR faz a leitura principal
- o validador local e o ranking deterministico geram os candidatos top-k
- so depois disso o Ollama pode ser consultado como desempate, se estiver habilitado

Ou seja, o Ollama nao substitui o OCR local e nao roda antes do top-k.

Pontos importantes:

- e desabilitado por padrao
- roda localmente via endpoint do Ollama, sem depender de API externa
- so usa candidatos que o pipeline ja produziu; ele nao deve inventar uma placa nova
- pode abstencao quando a ambiguidade continua alta
- o override final so acontece se a confianca minima configurada for atendida

Se o Ollama estiver desligado, sem modelo instalado ou indisponivel, o pipeline continua funcionando com o caminho deterministico normal.

Configuracoes relevantes:

- `llm_validation.enabled`
- `llm_validation.base_url`
- `llm_validation.model`
- `llm_validation.allow_override`
- `llm_validation.ambiguity_gap_threshold`
- `llm_validation.min_decision_confidence`

## Video

O processamento de video fica em `src/video_processor.py`.

O modulo:

- abre o video
- processa 1 a cada `N` frames conforme `skip_frames`
- usa modo `moving` ou `stationary`
- consolida placas entre frames
- gera ranking das leituras mais provaveis
- opcionalmente salva um video anotado

### Modos de video

- `moving`: processa mais frames e prioriza capturar a placa em momentos diferentes
- `stationary`: processa menos frames, aplica filtro de nitidez e pode fazer early-stop quando a leitura estabiliza com alta confianca

### Gerar video anotado

Quando `video.generate_output_video: true`, o processador cria um video de saida em `data/results/` com o mesmo FPS e a mesma resolucao do arquivo original.

As anotacoes incluem:

- bounding box da placa
- texto lido
- confianca da leitura
- cor da anotacao de acordo com a confianca

Detalhes importantes do comportamento atual:

- frames processados recebem a anotacao daquele frame
- frames pulados reutilizam a ultima anotacao conhecida, para o video nao ficar "piscando"
- se o modo `stationary` atingir early-stop, os frames restantes podem continuar sendo gravados com a ultima anotacao consolidada

O nome do arquivo de saida segue o padrao:

`<nome_original>_alpr_<timestamp>.<ext>`

## Votacao temporal

A votacao temporal usa `src/temporal_voting.py` e e integrada por `src/video_processor.py`.

O objetivo e simples: a mesma placa aparece em varios frames, mas cada frame pode errar um caractere diferente. A votacao junta essas leituras para produzir uma versao mais confiavel.

O motor temporal faz o seguinte:

- associa leituras da mesma placa ao longo do video usando IoU de bbox e similaridade de texto
- cria `tracks` por placa
- aplica uma estrategia de consolidacao quando ha observacoes suficientes

Estrategias disponiveis:

- `positional`: vota caractere por caractere
- `majority`: vota pela placa completa mais frequente
- `hybrid`: combina as duas abordagens

No release atual, o modo padrao e `hybrid`.

Depois da votacao, o `VideoProcessor` ainda calcula um ranking composto das placas usando:

- numero de deteccoes
- melhor confianca individual
- confianca media
- qualidade media
- confirmacao por caractere
- extensao temporal da track
- bonus para leituras votadas

Configuracoes relevantes:

- `temporal_voting.enabled`
- `temporal_voting.strategy`
- `temporal_voting.min_observations`

## Fluxo Premium

O fluxo Premium usa Plate Recognizer apenas quando o usuario clica no botao dedicado da interface.

Pontos importantes:

- ele envia a imagem completa, nao o crop da placa
- ele roda separado do fluxo local
- ele nao substitui automaticamente a leitura local
- a chave deve ficar no `.env`, via `PLATE_RECOGNIZER_API_KEY`

Detalhes de configuracao e recomendacoes de threshold estao em `PLATE_RECOGNIZER_API.md`.

## Configuracao importante

Campos que mais mudam o comportamento do sistema:

- `models.detector.confidence`: threshold base do detector
- `models.detector.enable_sahi`: habilita sliced inference em segunda tentativa
- `ocr.try_multiple_variants`: liga ou desliga as variantes de OCR
- `ocr.max_variants`: limita quantas variantes entram no OCR
- `ocr.paddle.use_angle_cls`: habilita a correcao de orientacao da linha de texto no PaddleOCR
- `pipeline.ocr_confidence_threshold`: limiar minimo esperado do OCR local
- `pipeline.fallback_confidence_threshold`: abaixo disso, o pipeline tenta rankear alternativas
- `llm_validation.enabled`: liga ou desliga o desempate opcional via Ollama
- `llm_validation.model`: define o modelo Ollama quando voce nao quer usar a selecao automatica
- `llm_validation.ambiguity_gap_threshold`: define quando a ambiguidade top-2 justifica consultar o LLM
- `llm_validation.min_decision_confidence`: confianca minima exigida para aceitar override do LLM
- `premium_api.min_confidence`: limiar minimo para aceitar a leitura Premium
- `video.skip_frames`: controla amostragem no video
- `video.generate_output_video`: salva ou nao o video anotado
- `temporal_voting.enabled`: liga ou desliga consolidacao temporal

## Estrutura do projeto

```text
ALPR/
├── app.py
├── config.yaml
├── PLATE_RECOGNIZER_API.md
├── data/
│   ├── fixtures/
│   └── results/
├── docs/
│   └── V2_VALIDATION.md
├── models/
│   └── yolo/
├── src/
│   ├── detector.py
│   ├── geometric_normalizer.py
│   ├── premium_alpr.py
│   ├── preprocessor.py
│   ├── temporal_voting.py
│   ├── video_processor.py
│   ├── ocr/
│   └── v2/
└── tests/
```

## Instalacao detalhada

Para um clone limpo do repositório publico, o ponto principal e este: o projeto nao versiona segredos, midias geradas nem pesos locais grandes. Isso significa que a instalacao das bibliotecas so resolve parte do bootstrap.

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
```

Para GPU CUDA 12.x:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Ou use:

```bash
python install_dependencies.py
```

## Bootstrap de um clone limpo

Este repositório nao inclui por padrao:

- `.env` com chaves privadas
- pesos YOLO locais em `models/yolo/`
- resultados e videos gerados em `data/results/`

Depois de instalar as dependencias:

1. copie `.env.example` para `.env` apenas se quiser usar o fluxo Premium com Plate Recognizer
2. baixe pelo menos um peso YOLO de placas e coloque o arquivo `.pt` em `models/yolo/`
3. use um nome esperado pelo projeto, como `yolo11l-plate.pt`, ou ajuste o modelo selecionado na sidebar e na configuracao

Sem um arquivo `.pt` valido em `models/yolo/`, o fluxo local nao inicializa corretamente porque o detector precisa de um peso real fora do repositório.

O que continua opcional mesmo em um clone limpo:

- `PLATE_RECOGNIZER_API_KEY` no `.env` para comparacao Premium
- Ollama local, usado apenas como desempate depois do top-k

## Execucao

```bash
streamlit run app.py
```

## Baseline, fixtures e calibracao

O projeto inclui infraestrutura de avaliacao offline em `data/fixtures/` e `src/v2/evaluation.py` para:

- carregar fixtures rotulados
- gerar relatorios de baseline
- comparar mudancas entre versoes
- calibrar thresholds de detector, OCR e fallback

## Validacao

Na validacao mais recente do workspace, a suite automatizada passou com `350 passed`.

Para rodar os testes:

```bash
python -m pytest tests -q
```

Para o que ainda depende de midia real e validacao manual, consulte `docs/V2_VALIDATION.md`.

## Licenca

O codigo deste projeto e distribuido sob a licenca MIT. Veja o arquivo `LICENSE`.
