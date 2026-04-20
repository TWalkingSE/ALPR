# ALPR 2.0

ALPR 2.0 é um projeto de leitura de placas focado em execução local, previsibilidade operacional e diagnóstico. O caminho principal roda offline com YOLOv11 + OpenCV + PaddleOCR. O caminho Premium via Plate Recognizer existe como comparação manual, não como fallback automático. A integração com Ollama é opcional e aparece apenas depois do top-k como desempate controlado.

<img width="2067" height="1182" alt="image" src="https://github.com/user-attachments/assets/55ac4169-11a4-4a1e-82fe-996c3b2c6c46" />

## Resumo rápido

- pipeline local principal para imagem e vídeo
- OCR principal local e determinístico com PaddleOCR
- SAHI, normalização, preprocessamento adaptativo e votação temporal
- Plate Recognizer opcional para comparação manual
- Ollama opcional apenas depois do top-k
- Python 3.11+, interface Streamlit e licença MIT
- validação automatizada mais recente do workspace: `350 passed`

## Início rápido

Se você só quer colocar o projeto de pé em um clone limpo:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
# coloque um peso YOLO .pt em models/yolo/
streamlit run app.py
```

Antes de rodar, tenha em mente:

- o repositório público não inclui `.env`, pesos YOLO locais nem resultados gerados
- sem um arquivo `.pt` válido em `models/yolo/`, o fluxo local não inicializa
- `PLATE_RECOGNIZER_API_KEY` só é necessário para o fluxo Premium
- Ollama continua opcional e fora do caminho principal do OCR

## Navegação rápida

- [O que o projeto entrega](#o-que-o-projeto-entrega)
- [Limites práticos](#limites-práticos)
- [Fluxos da aplicação](#fluxos-da-aplicação)
- [Instalação detalhada](#instalação-detalhada)
- [Bootstrap de um clone limpo](#bootstrap-de-um-clone-limpo)
- [Contributing](CONTRIBUTING.md)
- [Validação](#validação)
- [Licença](#licença)

## O que o projeto entrega

- Análise local de imagem com detector de placas, normalização geométrica, preprocessamento, OCR e validação.
- Validação inteligente opcional via Ollama, usada apenas como desempate depois do ranking top-k.
- Análise local de vídeo com agregação entre frames, ranking de placas e saída anotada.
- Comparação lado a lado entre o pipeline local e o Plate Recognizer.
- Captura opcional de artefatos para casos inválidos ou de baixa confiança.
- Infraestrutura de fixtures, baseline e calibração para validação regressiva.

## Limites práticos

O projeto melhora a chance de leitura com detector, normalização, preprocessamento, OCR, validação e agregação temporal, mas ele continua dependente da qualidade real da imagem ou do vídeo. Em outras palavras: o projeto não faz milagre.

Se a entrada vier ruim demais, o resultado pode não existir, pode ficar abaixo do threshold esperado ou pode terminar apenas como candidato fraco. O comportamento correto nesses cenários não é inventar uma placa com confiança artificial.

Isso vale principalmente para casos como:

- placa muito pequena no quadro
- poucos pixels úteis na região da placa, mesmo quando a detecção acontece
- desfoque por movimento
- foco ruim, lente suja ou vibração da câmera
- baixa iluminação, contraluz ou excesso de brilho
- compressão forte, ruído, chuva ou reflexos
- zoom digital agressivo, frame muito comprimido ou bitrate baixo
- obstrução parcial, sujeira, para-choque cobrindo caracteres ou ângulo extremo
- placa amassada, tipografia degradada ou caracteres fisicamente ilegíveis

Na prática, quando falta informação visual suficiente, nenhum ajuste de preprocessamento, SAHI, top-k, Ollama ou comparação com fluxo Premium consegue recuperar detalhes que não existem no arquivo de entrada.

O que normalmente acontece nesses casos:

- o detector pode não encontrar placa alguma
- o OCR pode ler apenas parte dos caracteres
- o validador pode rejeitar a leitura por formato inconsistente
- o pipeline pode manter várias alternativas próximas sem confiança para desempate
- o resultado final pode ser vazio, inválido ou abaixo do threshold configurado

Em vídeo, a agregação entre frames ajuda quando existem alguns quadros aproveitáveis. Ela não recupera detalhe que nunca apareceu de forma legível em nenhum frame. Se o vídeo inteiro estiver ruim, muito comprimido, tremido ou distante, o comportamento esperado é não haver leitura confiável.

Regra prática: para o sistema funcionar bem, a placa precisa aparecer com tamanho razoável, contraste suficiente e pelo menos alguns frames ou imagens realmente legíveis. Quando isso não acontece, a saída mais honesta do sistema é baixa confiança ou ausência de leitura.

Se a meta operacional for aumentar acerto no mundo real, o maior ganho quase sempre vem da captura, não do pós-processamento:

- aproximar mais a câmera ou usar enquadramento em que a placa ocupe mais pixels
- reduzir blur com shutter melhor, estabilização ou menor velocidade relativa
- melhorar iluminação e evitar reflexo direto
- preservar bitrate e resolução em vídeo, evitando compressão excessiva
- selecionar imagens e frames em que a placa esteja frontal ou pouco inclinada

## Fluxos da aplicação

### Fluxo local

O fluxo local é orquestrado por `src/v2/pipeline.py` e segue esta ordem:

1. detectar placas na imagem completa
2. recortar os crops detectados
3. normalizar geometricamente o crop
4. preprocessar o crop
5. rodar OCR
6. validar o texto lido
7. rankear alternativas quando a leitura ainda estiver fraca
8. opcionalmente consultar o Ollama para desempate inteligente quando ainda houver ambiguidade suficiente
9. salvar artefatos diagnósticos, se habilitado

### Fluxo Premium

O fluxo Premium usa `src/premium_alpr.py` e envia a imagem completa para a API da Plate Recognizer.

Ele serve para comparação e investigação, não para substituir automaticamente a leitura local.

## Detector

O detector local fica em `src/detector.py` e usa YOLOv11 treinado para placas.

Na prática, ele faz o seguinte:

- garante que a imagem tenha 3 canais BGR antes da inferência
- roda uma primeira inferência na imagem inteira
- se `enable_sahi: true`, pode tentar uma segunda passada com SAHI em imagens grandes quando não há detecções, quando a confiança padrão está baixa ou quando a maior detecção ainda parece pequena demais no quadro
- aplica margem adaptativa no recorte da placa
- faz upscale automático em crops muito pequenos para melhorar o OCR

### O que é SAHI

SAHI significa `Sliced Aided Hyper Inference`.

Em vez de rodar o detector só na imagem inteira, a imagem é dividida em blocos sobrepostos. O YOLO é executado em cada bloco, e as detecções repetidas nas regiões de sobreposição são unificadas depois por NMS.

No ALPR 2.0, o SAHI não roda o tempo todo. Ele entra como segunda tentativa quando:

- `models.detector.enable_sahi` está ligado
- a imagem é grande o suficiente para justificar slicing
- a detecção padrão não encontrou nenhuma placa
- a melhor detecção da passada padrão ficou abaixo do limiar configurado de confiança
- a maior detecção ainda ocupa área muito pequena no quadro, sugerindo placa distante

Quando a passada SAHI encontra algo útil, o pipeline combina as detecções padrão e as sliced com NMS para evitar duplicatas.

Isso ajuda principalmente em:

- placas pequenas
- placas distantes
- imagens de câmera de vigilância ou rodovia
- cenas em que a placa ocupa poucos pixels no quadro

Configurações relevantes em `config.yaml`:

- `models.detector.enable_sahi`
- `models.detector.sahi_slice_size`
- `models.detector.sahi_overlap_ratio`
- `models.detector.sahi_retry_confidence_threshold`
- `models.detector.sahi_retry_area_ratio_threshold`
- `models.detector.sahi_retry_large_image_threshold`
- `models.detector.sahi_merge_iou_threshold`

## Normalização geométrica

A normalização fica em `src/geometric_normalizer.py`.

Ela entra entre o detector e o OCR e tenta transformar o crop da placa em uma imagem mais retificada.

O módulo sabe fazer:

- detecção aproximada dos 4 cantos da placa
- transformação de perspectiva
- correção de rotação
- equalização de contraste
- redimensionamento padronizado

No pipeline v2 atual, o normalizador é instanciado com:

- correção de perspectiva ativa
- correção de rotação ativa
- redimensionamento padronizado ativo
- equalização de contraste desativada

A equalização de contraste foi deixada para o preprocessador, para não duplicar etapas de contraste no mesmo crop.

## Preprocessamento

O preprocessamento fica em `src/preprocessor.py`.

Ele trabalha sobre o crop já normalizado e pode gerar várias versões da mesma placa para aumentar a chance do OCR acertar em cenários difíceis.

O que o preprocessador faz:

- converte para grayscale quando necessário
- faz upscale quando a placa está muito pequena para OCR
- aplica CLAHE adaptativo para melhorar contraste
- remove ruído com `fastNlMeansDenoising` ou bilateral, com reforço extra quando o SNR está baixo
- aplica nitidez via unsharp mask
- aplica um passo extra de reforço quando o crop indica motion blur alto
- gera threshold adaptativo gaussiano como binarização principal
- quando habilitado, gera variantes extras com Otsu, Mean e versões invertidas e não invertidas
- aplica otimizações específicas para placas brasileiras, incluindo tentativas para Mercosul e formato antigo
- gera pequenas rotações e ajustes de gamma quando o modo adaptativo entende que a imagem precisa disso

O preprocessador também ajusta sua agressividade pela qualidade estimada do crop:

- imagem excelente: menos variantes, sem augmentation
- imagem suficiente: fluxo padrão
- imagem crítica: fluxo padrão com augmentation
- imagem insuficiente: sharpen mais forte e mais tentativas

Esse ajuste não depende apenas do score global. O preprocessor também reage a sinais objetivos como `snr` baixo e `motion_blur` alto para aumentar denoising, sharpening e número de variantes quando necessário.

## O que acontece quando não testamos múltiplas variantes

Esse ponto é importante.

No projeto, `ocr.try_multiple_variants` controla duas coisas ao mesmo tempo:

- o preprocessador deixa de gerar o bloco extra de múltiplas binarizações
- o `OCRManager` deixa de iterar sobre `preprocessed_variants`

Na prática, quando `ocr.try_multiple_variants: false`, o OCR roda apenas sobre a imagem normalizada principal. As saídas extras do preprocessamento deixam de participar da decisão do OCR.

Ou seja:

- com `true`: o OCR pode testar várias versões da placa e escolher a melhor
- com `false`: o OCR fica mais rápido e mais determinístico, mas abre mão das tentativas extras

Configurações relevantes:

- `ocr.try_multiple_variants`
- `ocr.max_variants`

## OCR e validação

O OCR local usa `PaddleOCR` via `src/ocr/paddle_engine.py`, encapsulado pelo `OCRManager` em `src/ocr/manager.py`.

Ele continua sendo o OCR principal do projeto. Mesmo com a opção de Ollama disponível, a leitura primária é local, clássica e determinística.

Depois da leitura, o texto passa por:

- limpeza do texto bruto
- reconstruções de confiança por caractere
- validação de formato em `src/validator.py`
- ranking de alternativas no pipeline quando a leitura ainda está abaixo do threshold esperado

O projeto trabalha com thresholds diferentes para OCR e fallback, e esses limiares podem ser flexibilizados por contexto, como baixa iluminação e placa pequena.

### Correção de orientação do texto

O PaddleOCR pode rodar com correção de orientação de texto quando `ocr.paddle.use_angle_cls: true`.

Na prática, isso habilita uma classificação de orientação da linha de texto antes do reconhecimento. Essa etapa ajuda quando o crop chega ao OCR com a linha da placa girada ou invertida o suficiente para atrapalhar a leitura.

Esse recurso é complementar ao normalizador geométrico:

- a normalização geométrica corrige perspectiva e rotação do crop da placa
- a correção de orientação do texto atua no nível da linha de texto dentro do OCR

Ela não substitui a retificação da placa. O caminho esperado continua sendo: primeiro normalizar o crop, depois deixar o OCR refinar a orientação do texto se necessário.

Na maioria dos casos de placas BR, vale manter ligado. Se você quiser reduzir custo e maximizar previsibilidade em entradas já muito bem normalizadas, pode desligar.

Configuração relevante:

- `ocr.paddle.use_angle_cls`

### Validação inteligente opcional via Ollama

O projeto também pode usar Ollama para uma validação inteligente opcional, mas ele não entra como OCR principal.

O comportamento correto é este:

- o PaddleOCR faz a leitura principal
- o validador local e o ranking determinístico geram os candidatos top-k
- só depois disso o Ollama pode ser consultado como desempate, se estiver habilitado

Ou seja, o Ollama não substitui o OCR local e não roda antes do top-k.

Pontos importantes:

- é desabilitado por padrão
- roda localmente via endpoint do Ollama, sem depender de API externa
- só usa candidatos que o pipeline já produziu; ele não deve inventar uma placa nova
- pode abstenção quando a ambiguidade continua alta
- o override final só acontece se a confiança mínima configurada for atendida

Se o Ollama estiver desligado, sem modelo instalado ou indisponível, o pipeline continua funcionando com o caminho determinístico normal.

Configurações relevantes:

- `llm_validation.enabled`
- `llm_validation.base_url`
- `llm_validation.model`
- `llm_validation.allow_override`
- `llm_validation.ambiguity_gap_threshold`
- `llm_validation.min_decision_confidence`

## Vídeo

O processamento de vídeo fica em `src/video_processor.py`.

O módulo:

- abre o vídeo
- processa 1 a cada `N` frames conforme `skip_frames`
- usa modo `moving` ou `stationary`
- consolida placas entre frames
- gera ranking das leituras mais prováveis
- opcionalmente salva um vídeo anotado

### Modos de vídeo

- `moving`: processa mais frames e prioriza capturar a placa em momentos diferentes
- `stationary`: processa menos frames, aplica filtro de nitidez e pode fazer early-stop quando a leitura estabiliza com alta confiança

### Gerar vídeo anotado

Quando `video.generate_output_video: true`, o processador cria um vídeo de saída em `data/results/` com o mesmo FPS e a mesma resolução do arquivo original.

As anotações incluem:

- bounding box da placa
- texto lido
- confiança da leitura
- cor da anotação de acordo com a confiança

Detalhes importantes do comportamento atual:

- frames processados recebem a anotação daquele frame
- frames pulados reutilizam a última anotação conhecida, para o vídeo não ficar "piscando"
- se o modo `stationary` atingir early-stop, os frames restantes podem continuar sendo gravados com a última anotação consolidada

O nome do arquivo de saída segue o padrão:

`<nome_original>_alpr_<timestamp>.<ext>`

## Votação temporal

A votação temporal usa `src/temporal_voting.py` e é integrada por `src/video_processor.py`.

O objetivo é simples: a mesma placa aparece em vários frames, mas cada frame pode errar um caractere diferente. A votação junta essas leituras para produzir uma versão mais confiável.

O motor temporal faz o seguinte:

- associa leituras da mesma placa ao longo do vídeo usando IoU de bbox e similaridade de texto
- cria `tracks` por placa
- aplica uma estratégia de consolidação quando há observações suficientes

Estratégias disponíveis:

- `positional`: vota caractere por caractere
- `majority`: vota pela placa completa mais frequente
- `hybrid`: combina as duas abordagens

No release atual, o modo padrão é `hybrid`.

Depois da votação, o `VideoProcessor` ainda calcula um ranking composto das placas usando:

- número de detecções
- melhor confiança individual
- confiança média
- qualidade média
- confirmação por caractere
- extensão temporal da track
- bônus para leituras votadas

Configurações relevantes:

- `temporal_voting.enabled`
- `temporal_voting.strategy`
- `temporal_voting.min_observations`

## Fluxo Premium

O fluxo Premium usa Plate Recognizer apenas quando o usuário clica no botão dedicado da interface.

Pontos importantes:

- ele envia a imagem completa, não o crop da placa
- ele roda separado do fluxo local
- ele não substitui automaticamente a leitura local
- a chave deve ficar no `.env`, via `PLATE_RECOGNIZER_API_KEY`

Detalhes de configuração e recomendações de threshold estão em `PLATE_RECOGNIZER_API.md`.

## Configuração importante

Campos que mais mudam o comportamento do sistema:

- `models.detector.confidence`: threshold base do detector
- `models.detector.enable_sahi`: habilita sliced inference em segunda tentativa
- `ocr.try_multiple_variants`: liga ou desliga as variantes de OCR
- `ocr.max_variants`: limita quantas variantes entram no OCR
- `ocr.paddle.use_angle_cls`: habilita a correção de orientação da linha de texto no PaddleOCR
- `pipeline.ocr_confidence_threshold`: limiar mínimo esperado do OCR local
- `pipeline.fallback_confidence_threshold`: abaixo disso, o pipeline tenta rankear alternativas
- `llm_validation.enabled`: liga ou desliga o desempate opcional via Ollama
- `llm_validation.model`: define o modelo Ollama quando você não quer usar a seleção automática
- `llm_validation.ambiguity_gap_threshold`: define quando a ambiguidade top-2 justifica consultar o LLM
- `llm_validation.min_decision_confidence`: confiança mínima exigida para aceitar override do LLM
- `premium_api.min_confidence`: limiar mínimo para aceitar a leitura Premium
- `video.skip_frames`: controla amostragem no vídeo
- `video.generate_output_video`: salva ou não o vídeo anotado
- `temporal_voting.enabled`: liga ou desliga consolidação temporal

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

## Instalação detalhada

Para um clone limpo do repositório público, o ponto principal é este: o projeto não versiona segredos, mídias geradas nem pesos locais grandes. Isso significa que a instalação das bibliotecas só resolve parte do bootstrap.

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

Este repositório não inclui por padrão:

- `.env` com chaves privadas
- pesos YOLO locais em `models/yolo/`
- resultados e vídeos gerados em `data/results/`

Depois de instalar as dependências:

1. copie `.env.example` para `.env` apenas se quiser usar o fluxo Premium com Plate Recognizer
2. baixe pelo menos um peso YOLO de placas e coloque o arquivo `.pt` em `models/yolo/`
3. use um nome esperado pelo projeto, como `yolo11l-plate.pt`, ou ajuste o modelo selecionado na sidebar e na configuração

Sem um arquivo `.pt` válido em `models/yolo/`, o fluxo local não inicializa corretamente porque o detector precisa de um peso real fora do repositório.

O que continua opcional mesmo em um clone limpo:

- `PLATE_RECOGNIZER_API_KEY` no `.env` para comparação Premium
- Ollama local, usado apenas como desempate depois do top-k

## Execução

```bash
streamlit run app.py
```

## Baseline, fixtures e calibração

O projeto inclui infraestrutura de avaliação offline em `data/fixtures/` e `src/v2/evaluation.py` para:

- carregar fixtures rotulados
- gerar relatórios de baseline
- comparar mudanças entre versões
- calibrar thresholds de detector, OCR e fallback

## Validação

Na validação mais recente do workspace, a suíte automatizada passou com `350 passed`.

Para rodar os testes:

```bash
python -m pytest tests -q
```

Para o que ainda depende de mídia real e validação manual, consulte `docs/V2_VALIDATION.md`.

## Licença

O código deste projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE`.
