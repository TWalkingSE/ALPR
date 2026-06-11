# ALPR 2.0

ALPR 2.0 Ã© um projeto de leitura de placas focado em execuÃ§Ã£o local, previsibilidade operacional e diagnÃ³stico. O caminho principal roda offline com YOLOv11 + OpenCV + PaddleOCR. O caminho Premium via Plate Recognizer existe como comparaÃ§Ã£o manual, nÃ£o como fallback automÃ¡tico. A integraÃ§Ã£o com Ollama Ã© opcional e aparece apenas depois do top-k como desempate controlado.

## Resumo rÃ¡pido

- pipeline local principal para imagem e vÃ­deo
- OCR principal local e determinÃ­stico com PaddleOCR
- SAHI, normalizaÃ§Ã£o, preprocessamento adaptativo e votaÃ§Ã£o temporal
- Plate Recognizer opcional para comparaÃ§Ã£o manual
- Ollama opcional apenas depois do top-k
- Python 3.11+, interface Streamlit e licenÃ§a MIT
- validaÃ§Ã£o automatizada mais recente do workspace: `384 passed`

## InÃ­cio rÃ¡pido

Se vocÃª sÃ³ quer colocar o projeto de pÃ© em um clone limpo:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
# coloque um peso YOLO .pt em models/yolo/
streamlit run app.py
```

Antes de rodar, tenha em mente:

- o repositÃ³rio pÃºblico nÃ£o inclui `.env`, pesos YOLO locais nem resultados gerados
- sem um arquivo `.pt` vÃ¡lido em `models/yolo/`, o fluxo local nÃ£o inicializa
- `PLATE_RECOGNIZER_API_KEY` sÃ³ Ã© necessÃ¡rio para o fluxo Premium
- Ollama continua opcional e fora do caminho principal do OCR

## NavegaÃ§Ã£o rÃ¡pida

- [O que o projeto entrega](#o-que-o-projeto-entrega)
- [Limites prÃ¡ticos](#limites-prÃ¡ticos)
- [Fluxos da aplicaÃ§Ã£o](#fluxos-da-aplicaÃ§Ã£o)
- [InstalaÃ§Ã£o detalhada](#instalaÃ§Ã£o-detalhada)
- [Bootstrap de um clone limpo](#bootstrap-de-um-clone-limpo)
- [Contributing](CONTRIBUTING.md)
- [ValidaÃ§Ã£o](#validaÃ§Ã£o)
- [LicenÃ§a](#licenÃ§a)

## O que o projeto entrega

- AnÃ¡lise local de imagem com detector de placas, normalizaÃ§Ã£o geomÃ©trica, preprocessamento, OCR e validaÃ§Ã£o.
- ValidaÃ§Ã£o inteligente opcional via Ollama, usada apenas como desempate depois do ranking top-k.
- AnÃ¡lise local de vÃ­deo com agregaÃ§Ã£o entre frames, ranking de placas e saÃ­da anotada.
- ComparaÃ§Ã£o lado a lado entre o pipeline local e o Plate Recognizer.
- Captura opcional de artefatos para casos invÃ¡lidos ou de baixa confianÃ§a.
- Infraestrutura de fixtures, baseline e calibraÃ§Ã£o para validaÃ§Ã£o regressiva.

## Limites prÃ¡ticos

O projeto melhora a chance de leitura com detector, normalizaÃ§Ã£o, preprocessamento, OCR, validaÃ§Ã£o e agregaÃ§Ã£o temporal, mas ele continua dependente da qualidade real da imagem ou do vÃ­deo. Em outras palavras: o projeto nÃ£o faz milagre.

Se a entrada vier ruim demais, o resultado pode nÃ£o existir, pode ficar abaixo do threshold esperado ou pode terminar apenas como candidato fraco. O comportamento correto nesses cenÃ¡rios nÃ£o Ã© inventar uma placa com confianÃ§a artificial.

Isso vale principalmente para casos como:

- placa muito pequena no quadro
- poucos pixels Ãºteis na regiÃ£o da placa, mesmo quando a detecÃ§Ã£o acontece
- desfoque por movimento
- foco ruim, lente suja ou vibraÃ§Ã£o da cÃ¢mera
- baixa iluminaÃ§Ã£o, contraluz ou excesso de brilho
- compressÃ£o forte, ruÃ­do, chuva ou reflexos
- zoom digital agressivo, frame muito comprimido ou bitrate baixo
- obstruÃ§Ã£o parcial, sujeira, para-choque cobrindo caracteres ou Ã¢ngulo extremo
- placa amassada, tipografia degradada ou caracteres fisicamente ilegÃ­veis

Na prÃ¡tica, quando falta informaÃ§Ã£o visual suficiente, nenhum ajuste de preprocessamento, SAHI, top-k, Ollama ou comparaÃ§Ã£o com fluxo Premium consegue recuperar detalhes que nÃ£o existem no arquivo de entrada.

O que normalmente acontece nesses casos:

- o detector pode nÃ£o encontrar placa alguma
- o OCR pode ler apenas parte dos caracteres
- o validador pode rejeitar a leitura por formato inconsistente
- o pipeline pode manter vÃ¡rias alternativas prÃ³ximas sem confianÃ§a para desempate
- o resultado final pode ser vazio, invÃ¡lido ou abaixo do threshold configurado

Em vÃ­deo, a agregaÃ§Ã£o entre frames ajuda quando existem alguns quadros aproveitÃ¡veis. Ela nÃ£o recupera detalhe que nunca apareceu de forma legÃ­vel em nenhum frame. Se o vÃ­deo inteiro estiver ruim, muito comprimido, tremido ou distante, o comportamento esperado Ã© nÃ£o haver leitura confiÃ¡vel.

Regra prÃ¡tica: para o sistema funcionar bem, a placa precisa aparecer com tamanho razoÃ¡vel, contraste suficiente e pelo menos alguns frames ou imagens realmente legÃ­veis. Quando isso nÃ£o acontece, a saÃ­da mais honesta do sistema Ã© baixa confianÃ§a ou ausÃªncia de leitura.

Se a meta operacional for aumentar acerto no mundo real, o maior ganho quase sempre vem da captura, nÃ£o do pÃ³s-processamento:

- aproximar mais a cÃ¢mera ou usar enquadramento em que a placa ocupe mais pixels
- reduzir blur com shutter melhor, estabilizaÃ§Ã£o ou menor velocidade relativa
- melhorar iluminaÃ§Ã£o e evitar reflexo direto
- preservar bitrate e resoluÃ§Ã£o em vÃ­deo, evitando compressÃ£o excessiva
- selecionar imagens e frames em que a placa esteja frontal ou pouco inclinada

## Fluxos da aplicaÃ§Ã£o

### Fluxo local

O fluxo local Ã© orquestrado por `src/v2/pipeline.py` e segue esta ordem:

1. detectar placas na imagem completa
2. recortar os crops detectados
3. normalizar geometricamente o crop
4. preprocessar o crop
5. rodar OCR
6. validar o texto lido
7. rankear alternativas quando a leitura ainda estiver fraca
8. opcionalmente consultar o Ollama para desempate inteligente quando ainda houver ambiguidade suficiente
9. salvar artefatos diagnÃ³sticos, se habilitado

### Fluxo Premium

O fluxo Premium usa `src/premium_alpr.py` e envia a imagem completa para a API da Plate Recognizer.

Ele serve para comparaÃ§Ã£o e investigaÃ§Ã£o, nÃ£o para substituir automaticamente a leitura local.

## Detector

O detector local fica em `src/detector.py` e usa YOLOv11 treinado para placas.

Na prÃ¡tica, ele faz o seguinte:

- garante que a imagem tenha 3 canais BGR antes da inferÃªncia
- roda uma primeira inferÃªncia na imagem inteira
- se `enable_sahi: true`, pode tentar uma segunda passada com SAHI em imagens grandes quando nÃ£o hÃ¡ detecÃ§Ãµes, quando a confianÃ§a padrÃ£o estÃ¡ baixa ou quando a maior detecÃ§Ã£o ainda parece pequena demais no quadro
- aplica margem adaptativa no recorte da placa
- faz upscale automÃ¡tico em crops muito pequenos para melhorar o OCR

### O que Ã© SAHI

SAHI significa `Sliced Aided Hyper Inference`.

Em vez de rodar o detector sÃ³ na imagem inteira, a imagem Ã© dividida em blocos sobrepostos. O YOLO Ã© executado em cada bloco, e as detecÃ§Ãµes repetidas nas regiÃµes de sobreposiÃ§Ã£o sÃ£o unificadas depois por NMS.

No ALPR 2.0, o SAHI nÃ£o roda o tempo todo. Ele entra como segunda tentativa quando:

- `models.detector.enable_sahi` estÃ¡ ligado
- a imagem Ã© grande o suficiente para justificar slicing
- a detecÃ§Ã£o padrÃ£o nÃ£o encontrou nenhuma placa
- a melhor detecÃ§Ã£o da passada padrÃ£o ficou abaixo do limiar configurado de confianÃ§a
- a maior detecÃ§Ã£o ainda ocupa Ã¡rea muito pequena no quadro, sugerindo placa distante

Quando a passada SAHI encontra algo Ãºtil, o pipeline combina as detecÃ§Ãµes padrÃ£o e as sliced com NMS para evitar duplicatas.

Isso ajuda principalmente em:

- placas pequenas
- placas distantes
- imagens de cÃ¢mera de vigilÃ¢ncia ou rodovia
- cenas em que a placa ocupa poucos pixels no quadro

ConfiguraÃ§Ãµes relevantes em `config.yaml`:

- `models.detector.enable_sahi`
- `models.detector.sahi_slice_size`
- `models.detector.sahi_overlap_ratio`
- `models.detector.sahi_retry_confidence_threshold`
- `models.detector.sahi_retry_area_ratio_threshold`
- `models.detector.sahi_retry_large_image_threshold`
- `models.detector.sahi_merge_iou_threshold`

## NormalizaÃ§Ã£o geomÃ©trica

A normalizaÃ§Ã£o fica em `src/geometric_normalizer.py`.

Ela entra entre o detector e o OCR e tenta transformar o crop da placa em uma imagem mais retificada.

O mÃ³dulo sabe fazer:

- detecÃ§Ã£o aproximada dos 4 cantos da placa
- transformaÃ§Ã£o de perspectiva
- correÃ§Ã£o de rotaÃ§Ã£o
- equalizaÃ§Ã£o de contraste
- redimensionamento padronizado

No pipeline v2 atual, o normalizador Ã© instanciado com:

- correÃ§Ã£o de perspectiva ativa
- correÃ§Ã£o de rotaÃ§Ã£o ativa
- redimensionamento padronizado ativo
- equalizaÃ§Ã£o de contraste desativada

A equalizaÃ§Ã£o de contraste foi deixada para o preprocessador, para nÃ£o duplicar etapas de contraste no mesmo crop.

## Preprocessamento

O preprocessamento fica em `src/preprocessor.py`.

Ele trabalha sobre o crop jÃ¡ normalizado e pode gerar vÃ¡rias versÃµes da mesma placa para aumentar a chance do OCR acertar em cenÃ¡rios difÃ­ceis.

O que o preprocessador faz:

- converte para grayscale quando necessÃ¡rio
- faz upscale quando a placa estÃ¡ muito pequena para OCR
- aplica CLAHE adaptativo para melhorar contraste
- remove ruÃ­do com `fastNlMeansDenoising` ou bilateral, com reforÃ§o extra quando o SNR estÃ¡ baixo
- aplica nitidez via unsharp mask
- aplica um passo extra de reforÃ§o quando o crop indica motion blur alto
- gera threshold adaptativo gaussiano como binarizaÃ§Ã£o principal
- quando habilitado, gera variantes extras com Otsu, Mean e versÃµes invertidas e nÃ£o invertidas
- aplica otimizaÃ§Ãµes especÃ­ficas para placas brasileiras, incluindo tentativas para Mercosul e formato antigo
- gera pequenas rotaÃ§Ãµes e ajustes de gamma quando o modo adaptativo entende que a imagem precisa disso

O preprocessador tambÃ©m ajusta sua agressividade pela qualidade estimada do crop:

- imagem excelente: menos variantes, sem augmentation
- imagem suficiente: fluxo padrÃ£o
- imagem crÃ­tica: fluxo padrÃ£o com augmentation
- imagem insuficiente: sharpen mais forte e mais tentativas

Esse ajuste nÃ£o depende apenas do score global. O preprocessor tambÃ©m reage a sinais objetivos como `snr` baixo e `motion_blur` alto para aumentar denoising, sharpening e nÃºmero de variantes quando necessÃ¡rio.

## O que acontece quando nÃ£o testamos mÃºltiplas variantes

Esse ponto Ã© importante.

No projeto, `ocr.try_multiple_variants` controla duas coisas ao mesmo tempo:

- o preprocessador deixa de gerar o bloco extra de mÃºltiplas binarizaÃ§Ãµes
- o `OCRManager` deixa de iterar sobre `preprocessed_variants`

Na prÃ¡tica, quando `ocr.try_multiple_variants: false`, o OCR roda apenas sobre a imagem normalizada principal. As saÃ­das extras do preprocessamento deixam de participar da decisÃ£o do OCR.

Ou seja:

- com `true`: o OCR pode testar vÃ¡rias versÃµes da placa e escolher a melhor
- com `false`: o OCR fica mais rÃ¡pido e mais determinÃ­stico, mas abre mÃ£o das tentativas extras

ConfiguraÃ§Ãµes relevantes:

- `ocr.try_multiple_variants`
- `ocr.max_variants`

## OCR e validaÃ§Ã£o

O OCR local usa `PaddleOCR` via `src/ocr/paddle_engine.py`, encapsulado pelo `OCRManager` em `src/ocr/manager.py`.

Ele continua sendo o OCR principal do projeto. Mesmo com a opÃ§Ã£o de Ollama disponÃ­vel, a leitura primÃ¡ria Ã© local, clÃ¡ssica e determinÃ­stica.

Depois da leitura, o texto passa por:

- limpeza do texto bruto
- reconstruÃ§Ãµes de confianÃ§a por caractere
- validaÃ§Ã£o de formato em `src/validator.py`
- ranking de alternativas no pipeline quando a leitura ainda estÃ¡ abaixo do threshold esperado

O validador trata o formato Mercosul (LLLNLNN) corretamente: a letra da 5Âª
posiÃ§Ã£o pode ser qualquer letra A-Z, incluindo vogais (a conversÃ£o do formato
antigo gera, por exemplo, `0->A`, `4->E`, `8->I`).

A confianÃ§a por caractere usa consenso posicional entre as variantes de OCR:
posiÃ§Ãµes em que as leituras divergem recebem confianÃ§a menor, sinalizando com
mais precisÃ£o qual caractere estÃ¡ incerto.

O projeto trabalha com thresholds diferentes para OCR e fallback, e esses limiares podem ser flexibilizados por contexto, como baixa iluminaÃ§Ã£o e placa pequena.

### CorreÃ§Ã£o de orientaÃ§Ã£o do texto

O PaddleOCR pode rodar com correÃ§Ã£o de orientaÃ§Ã£o de texto quando `ocr.paddle.use_angle_cls: true`.

Na prÃ¡tica, isso habilita uma classificaÃ§Ã£o de orientaÃ§Ã£o da linha de texto antes do reconhecimento. Essa etapa ajuda quando o crop chega ao OCR com a linha da placa girada ou invertida o suficiente para atrapalhar a leitura.

Esse recurso Ã© complementar ao normalizador geomÃ©trico:

- a normalizaÃ§Ã£o geomÃ©trica corrige perspectiva e rotaÃ§Ã£o do crop da placa
- a correÃ§Ã£o de orientaÃ§Ã£o do texto atua no nÃ­vel da linha de texto dentro do OCR

Ela nÃ£o substitui a retificaÃ§Ã£o da placa. O caminho esperado continua sendo: primeiro normalizar o crop, depois deixar o OCR refinar a orientaÃ§Ã£o do texto se necessÃ¡rio.

Na maioria dos casos de placas BR, vale manter ligado. Se vocÃª quiser reduzir custo e maximizar previsibilidade em entradas jÃ¡ muito bem normalizadas, pode desligar.

ConfiguraÃ§Ã£o relevante:

- `ocr.paddle.use_angle_cls`

### ValidaÃ§Ã£o inteligente opcional via Ollama

O projeto tambÃ©m pode usar Ollama para uma validaÃ§Ã£o inteligente opcional, mas ele nÃ£o entra como OCR principal.

O comportamento correto Ã© este:

- o PaddleOCR faz a leitura principal
- o validador local e o ranking determinÃ­stico geram os candidatos top-k
- sÃ³ depois disso o Ollama pode ser consultado como desempate, se estiver habilitado

Ou seja, o Ollama nÃ£o substitui o OCR local e nÃ£o roda antes do top-k.

Pontos importantes:

- Ã© desabilitado por padrÃ£o
- roda localmente via endpoint do Ollama, sem depender de API externa
- sÃ³ usa candidatos que o pipeline jÃ¡ produziu; ele nÃ£o deve inventar uma placa nova
- pode abstenÃ§Ã£o quando a ambiguidade continua alta
- o override final sÃ³ acontece se a confianÃ§a mÃ­nima configurada for atendida

Se o Ollama estiver desligado, sem modelo instalado ou indisponÃ­vel, o pipeline continua funcionando com o caminho determinÃ­stico normal.

ConfiguraÃ§Ãµes relevantes:

- `llm_validation.enabled`
- `llm_validation.base_url`
- `llm_validation.model`
- `llm_validation.allow_override`
- `llm_validation.ambiguity_gap_threshold`
- `llm_validation.min_decision_confidence`

## VÃ­deo

O processamento de vÃ­deo fica em `src/video_processor.py`.

O mÃ³dulo:

- abre o vÃ­deo
- processa 1 a cada `N` frames conforme `skip_frames`
- usa modo `moving` ou `stationary`
- consolida placas entre frames
- gera ranking das leituras mais provÃ¡veis
- opcionalmente salva um vÃ­deo anotado

### Modos de vÃ­deo

- `moving`: processa mais frames e prioriza capturar a placa em momentos diferentes
- `stationary`: processa menos frames, aplica filtro de nitidez e pode fazer early-stop quando a leitura estabiliza com alta confianÃ§a

### Gerar vÃ­deo anotado

Quando `video.generate_output_video: true`, o processador cria um vÃ­deo de saÃ­da em `data/results/` com o mesmo FPS e a mesma resoluÃ§Ã£o do arquivo original.

As anotaÃ§Ãµes incluem:

- bounding box da placa
- texto lido
- confianÃ§a da leitura
- cor da anotaÃ§Ã£o de acordo com a confianÃ§a

Detalhes importantes do comportamento atual:

- frames processados recebem a anotaÃ§Ã£o daquele frame
- frames pulados reutilizam a Ãºltima anotaÃ§Ã£o conhecida, para o vÃ­deo nÃ£o ficar "piscando"
- se o modo `stationary` atingir early-stop, os frames restantes podem continuar sendo gravados com a Ãºltima anotaÃ§Ã£o consolidada

O nome do arquivo de saÃ­da segue o padrÃ£o:

`<nome_original>_alpr_<timestamp>.<ext>`

## VotaÃ§Ã£o temporal

A votaÃ§Ã£o temporal usa `src/temporal_voting.py` e Ã© integrada por `src/video_processor.py`.

O objetivo Ã© simples: a mesma placa aparece em vÃ¡rios frames, mas cada frame pode errar um caractere diferente. A votaÃ§Ã£o junta essas leituras para produzir uma versÃ£o mais confiÃ¡vel.

O motor temporal faz o seguinte:

- associa leituras da mesma placa ao longo do vÃ­deo usando IoU de bbox e similaridade de texto
- cria `tracks` por placa
- aplica uma estratÃ©gia de consolidaÃ§Ã£o quando hÃ¡ observaÃ§Ãµes suficientes

EstratÃ©gias disponÃ­veis:

- `positional`: vota caractere por caractere
- `majority`: vota pela placa completa mais frequente
- `hybrid`: combina as duas abordagens

No release atual, o modo padrÃ£o Ã© `hybrid`.

Depois da votaÃ§Ã£o, o `VideoProcessor` ainda calcula um ranking composto das placas usando:

- nÃºmero de detecÃ§Ãµes
- melhor confianÃ§a individual
- confianÃ§a mÃ©dia
- qualidade mÃ©dia
- confirmaÃ§Ã£o por caractere
- extensÃ£o temporal da track
- bÃ´nus para leituras votadas

ConfiguraÃ§Ãµes relevantes:

- `temporal_voting.enabled`
- `temporal_voting.strategy`
- `temporal_voting.min_observations`

## Fluxo Premium

O fluxo Premium usa Plate Recognizer apenas quando o usuÃ¡rio clica no botÃ£o dedicado da interface.

Pontos importantes:

- ele envia a imagem completa, nÃ£o o crop da placa
- ele roda separado do fluxo local
- ele nÃ£o substitui automaticamente a leitura local
- a chave deve ficar no `.env`, via `PLATE_RECOGNIZER_API_KEY`

Detalhes de configuraÃ§Ã£o e recomendaÃ§Ãµes de threshold estÃ£o em `PLATE_RECOGNIZER_API.md`.

## Atributos do veÃ­culo (opcional)

AlÃ©m da placa, o projeto pode estimar atributos do veÃ­culo quando
`vehicle_attributes.enabled: true`. O foco principal continua sendo a placa;
esse mÃ³dulo Ã© complementar e desligado por padrÃ£o.

- Cor dominante: calculada offline por anÃ¡lise HSV da regiÃ£o do veÃ­culo
  (estimada ao redor da placa). Funciona sem pesos externos.
- Marca e modelo: exigem um classificador treinado. O mÃ³dulo expÃµe uma interface
  injetÃ¡vel (`MakeModelClassifier`); sem um classificador configurado, marca e
  modelo retornam vazios em vez de inventar um resultado.

O resultado Ã© anexado a cada leitura em `vehicle_attributes` (cor, confianÃ§a da
cor, marca, modelo, bounding box estimada do veÃ­culo e origem).

ConfiguraÃ§Ãµes relevantes:

- `vehicle_attributes.enabled`
- `vehicle_attributes.roi_width_scale`
- `vehicle_attributes.roi_height_scale`

## ConfiguraÃ§Ã£o importante

Campos que mais mudam o comportamento do sistema:

- `models.detector.confidence`: threshold base do detector
- `models.detector.enable_sahi`: habilita sliced inference em segunda tentativa
- `ocr.try_multiple_variants`: liga ou desliga as variantes de OCR
- `ocr.max_variants`: limita quantas variantes entram no OCR
- `ocr.paddle.use_angle_cls`: habilita a correÃ§Ã£o de orientaÃ§Ã£o da linha de texto no PaddleOCR
- `ocr.paddle.add_quiet_zone`: adiciona uma borda branca ao redor do crop antes do OCR para estabilizar a detecÃ§Ã£o da linha em placas muito recortadas (padrÃ£o desligado)
- `vehicle_attributes.enabled`: liga o reconhecimento opcional de atributos do veÃ­culo (cor offline; marca/modelo sÃ³ com classificador injetÃ¡vel)
- `pipeline.ocr_confidence_threshold`: limiar mÃ­nimo esperado do OCR local
- `pipeline.fallback_confidence_threshold`: abaixo disso, o pipeline tenta rankear alternativas
- `llm_validation.enabled`: liga ou desliga o desempate opcional via Ollama
- `llm_validation.model`: define o modelo Ollama quando vocÃª nÃ£o quer usar a seleÃ§Ã£o automÃ¡tica
- `llm_validation.ambiguity_gap_threshold`: define quando a ambiguidade top-2 justifica consultar o LLM
- `llm_validation.min_decision_confidence`: confianÃ§a mÃ­nima exigida para aceitar override do LLM
- `premium_api.min_confidence`: limiar mÃ­nimo para aceitar a leitura Premium
- `video.skip_frames`: controla amostragem no vÃ­deo
- `video.generate_output_video`: salva ou nÃ£o o vÃ­deo anotado
- `temporal_voting.enabled`: liga ou desliga consolidaÃ§Ã£o temporal

## Estrutura do projeto

```text
ALPR/
â”œâ”€â”€ app.py
â”œâ”€â”€ config.yaml
â”œâ”€â”€ PLATE_RECOGNIZER_API.md
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ fixtures/
â”‚   â””â”€â”€ results/
â”œâ”€â”€ docs/
â”‚   â””â”€â”€ V2_VALIDATION.md
â”œâ”€â”€ models/
â”‚   â””â”€â”€ yolo/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ detector.py
â”‚   â”œâ”€â”€ geometric_normalizer.py
â”‚   â”œâ”€â”€ premium_alpr.py
â”‚   â”œâ”€â”€ preprocessor.py
â”‚   â”œâ”€â”€ temporal_voting.py
â”‚   â”œâ”€â”€ video_processor.py
â”‚   â”œâ”€â”€ ocr/
â”‚   â””â”€â”€ v2/
â””â”€â”€ tests/
```

## InstalaÃ§Ã£o detalhada

Para um clone limpo do repositÃ³rio pÃºblico, o ponto principal Ã© este: o projeto nÃ£o versiona segredos, mÃ­dias geradas nem pesos locais grandes. Isso significa que a instalaÃ§Ã£o das bibliotecas sÃ³ resolve parte do bootstrap.

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

Este repositÃ³rio nÃ£o inclui por padrÃ£o:

- `.env` com chaves privadas
- pesos YOLO locais em `models/yolo/`
- resultados e vÃ­deos gerados em `data/results/`

Depois de instalar as dependÃªncias:

1. copie `.env.example` para `.env` apenas se quiser usar o fluxo Premium com Plate Recognizer
2. baixe pelo menos um peso YOLO de placas e coloque o arquivo `.pt` em `models/yolo/`
3. use um nome esperado pelo projeto, como `yolo11l-plate.pt`, ou ajuste o modelo selecionado na sidebar e na configuraÃ§Ã£o

Sem um arquivo `.pt` vÃ¡lido em `models/yolo/`, o fluxo local nÃ£o inicializa corretamente porque o detector precisa de um peso real fora do repositÃ³rio.

O que continua opcional mesmo em um clone limpo:

- `PLATE_RECOGNIZER_API_KEY` no `.env` para comparaÃ§Ã£o Premium
- Ollama local, usado apenas como desempate depois do top-k

## ExecuÃ§Ã£o

```bash
streamlit run app.py
```

## Baseline, fixtures e calibraÃ§Ã£o

O projeto inclui infraestrutura de avaliaÃ§Ã£o offline em `data/fixtures/` e `src/v2/evaluation.py` para:

- carregar fixtures rotulados
- gerar relatÃ³rios de baseline
- comparar mudanÃ§as entre versÃµes
- calibrar thresholds de detector, OCR e fallback

## ValidaÃ§Ã£o

Na validaÃ§Ã£o mais recente do workspace, a suÃ­te automatizada passou com `384 passed`.

Para rodar os testes:

```bash
python -m pytest tests -q
```

Para o que ainda depende de mÃ­dia real e validaÃ§Ã£o manual, consulte `docs/V2_VALIDATION.md`.

## LicenÃ§a

O cÃ³digo deste projeto Ã© distribuÃ­do sob a licenÃ§a MIT. Veja o arquivo `LICENSE`.
