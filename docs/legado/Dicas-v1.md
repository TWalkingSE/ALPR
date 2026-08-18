> ⚠️ **DOCUMENTO OBSOLETO — NÃO USE COMO REFERÊNCIA.**
>
> Este guia descreve uma versão anterior da aplicação e **contradiz o
> comportamento atual** em vários pontos: menciona `OPENAI_API_KEY` (o projeto
> não usa OpenAI), abas de "Detecção de Adulteração" e "Thresholds Avançados"
> que não existem na sidebar atual (`src/v2/ui/sidebar.py`), e um "Pipeline de
> 10 Etapas" que não corresponde a `src/v2/pipeline.py`.
>
> Está preservado apenas como registro histórico. A documentação correta e
> mantida é o [`README.md`](../../README.md) na raiz do projeto.

---

# 📖 Dicas — Guia Completo do Projeto ALPR

> **ALPR** — Automatic License Plate Recognition  
> Sistema completo de reconhecimento de placas veiculares brasileiras com pipeline forense.  
> Suporte a **imagem** (JPG, PNG, BMP, WEBP) e **vídeo** (MP4, AVI, MOV, MKV, WMV, WEBM, DAV).

---

## Índice

1. [Visão Geral do Sistema](#1-visão-geral-do-sistema)
2. [Requisitos e Instalação](#2-requisitos-e-instalação)
3. [Como Iniciar a Aplicação](#3-como-iniciar-a-aplicação)
4. [Interface Web (Streamlit)](#4-interface-web-streamlit)
   - 4.1 [Barra Lateral de Configurações](#41-barra-lateral-de-configurações)
   - 4.2 [Tab Imagem](#42-tab-imagem)
   - 4.3 [Tab Vídeo](#43-tab-vídeo)
5. [Pipeline de Processamento (10 Etapas)](#5-pipeline-de-processamento-10-etapas)
   - 5.1 [Etapa 0 — Avaliação de Qualidade](#51-etapa-0--avaliação-de-qualidade)
   - 5.2 [Etapa 1 — Detecção (YOLO)](#52-etapa-1--detecção-yolo)
   - 5.3 [Etapa 1.5 — Normalização Geométrica](#53-etapa-15--normalização-geométrica)
   - 5.4 [Etapa 2 — Pré-processamento](#54-etapa-2--pré-processamento)
   - 5.5 [Etapa 3 — OCR (Reconhecimento)](#55-etapa-3--ocr-reconhecimento)
   - 5.6 [Etapa 4 — Validação e Pós-processamento](#56-etapa-4--validação-e-pós-processamento)
   - 5.7 [Etapa 5 — Fallback (Alternativas)](#57-etapa-5--fallback-alternativas)
   - 5.8 [Etapa 6 — Correção via LLM](#58-etapa-6--correção-via-llm)
   - 5.9 [Etapa 7 — Detecção de Adulteração](#59-etapa-7--detecção-de-adulteração)
   - 5.10 [Etapa 8 — Classificação da Placa](#510-etapa-8--classificação-da-placa)
   - 5.11 [Etapa 9 — Laudo Pericial Forense](#511-etapa-9--laudo-pericial-forense)
6. [Engines OCR Disponíveis](#6-engines-ocr-disponíveis)
   - 6.1 [EasyOCR + TrOCR (Primário)](#61-easyocr--trocr-primário)
   - 6.2 [Plate Recognizer API (Fallback)](#62-plate-recognizer-api-fallback)
   - 6.3 [PARSeq (Opcional)](#63-parseq-opcional)
   - 6.4 [Sistema de Votação Multi-Engine](#64-sistema-de-votação-multi-engine)
7. [Processamento de Vídeo](#7-processamento-de-vídeo)
   - 7.1 [Modos de Análise](#71-modos-de-análise)
   - 7.2 [Resultados do Vídeo](#72-resultados-do-vídeo)
8. [Arquivo de Configuração (config.yaml)](#8-arquivo-de-configuração-configyaml)
9. [Estrutura do Projeto](#9-estrutura-do-projeto)
10. [Uso via Python (Sem Interface)](#10-uso-via-python-sem-interface)
11. [Formatos de Placas Brasileiras](#11-formatos-de-placas-brasileiras)
12. [Solução de Problemas Comuns](#12-solução-de-problemas-comuns)
13. [Dicas de Performance](#13-dicas-de-performance)

---

## 1. Visão Geral do Sistema

O ALPR é um sistema de reconhecimento automático de placas veiculares brasileiras que opera em **10 etapas sequenciais**:

```
Imagem/Vídeo
    │
    ▼
[0] Avaliação de Qualidade ── Classifica a imagem (Excelente/Suficiente/Crítica/Insuficiente)
    │
    ▼
[1] Detecção (YOLO) ──────── Localiza placas na imagem com bounding boxes
    │
    ▼
[1.5] Normalização Geométrica ── Corrige perspectiva, rotação, contraste
    │
    ▼
[2] Pré-processamento ────── Filtros OpenCV (contraste, ruído, nitidez, threshold)
    │
    ▼
[3] OCR ──────────────────── Leitura dos caracteres (EasyOCR + TrOCR primário → Plate Recognizer fallback)
    │
    ▼
[4] Validação ────────────── Verifica formato (AAA-1234 ou AAA1B23) e corrige erros
    │
    ▼
[5] Fallback ─────────────── Gera alternativas combinatórias se confiança baixa
    │
    ▼
[6] Correção LLM ─────────── GPT-4o corrige com visão multimodal (opcional)
    │
    ▼
[7] Adulteração ──────────── 9 técnicas forenses para detectar tampering
    │
    ▼
[8] Classificação ─────────── Tipo e cor da placa (particular, comercial, oficial…)
    │
    ▼
[9] Laudo Pericial ────────── Relatório forense estruturado (Markdown)
```

**Formatos suportados:**

| Tipo    | Extensões                          |
|---------|------------------------------------|
| Imagem  | JPG, JPEG, PNG, BMP, WEBP          |
| Vídeo   | MP4, AVI, MOV, MKV, WMV, WEBM, DAV|

---

## 2. Requisitos e Instalação

### Requisitos de Sistema

| Componente | Mínimo             | Recomendado                |
|------------|--------------------|-----------------------------|
| Python     | 3.11+              | 3.12                        |
| RAM        | 8 GB               | 16 GB                       |
| GPU        | Funciona em CPU    | NVIDIA com CUDA 12.x        |
| VRAM       | —                  | 8 GB+                       |
| Disco      | ~5 GB              | ~10 GB (com modelos)        |

### Passo a Passo de Instalação

```bash
# 1. Clone o repositório (ou copie a pasta do projeto)
cd ALPR

# 2. Crie um ambiente virtual
python -m venv venv

# 3. Ative o ambiente virtual
.\venv\Scripts\activate          # Windows (PowerShell)
.\venv\Scripts\activate.bat      # Windows (CMD)
source venv/bin/activate         # Linux / macOS

# 4. Instale o PyTorch COM SUPORTE A CUDA (importante!)
#    Ajuste a versão CUDA conforme sua GPU:
#    - RTX 40xx / RTX 50xx: cu126 ou cu128
#    - RTX 30xx: cu121 ou cu126
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 5. Instale as demais dependências
pip install -r requirements.txt

# 6. (Opcional) Configure variáveis de ambiente para LLM e API
#    Crie um arquivo .env na raiz do projeto com:
#    OPENAI_API_KEY=sk-...
#    PLATE_RECOGNIZER_API_KEY=...
```

> **⚠️ IMPORTANTE:** Se você instalar o PyTorch sem especificar `--index-url`, ele virá na versão **CPU-only** e a GPU não será utilizada, mesmo estando disponível. Sempre use o comando com `--index-url` apontando para a versão CUDA.

### Verificar se a GPU está Funcionando

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Saída esperada (exemplo):
```
CUDA: True
GPU: NVIDIA RTX 4500 Ada Generation
```

Se mostrar `CUDA: False`, reinstale o PyTorch com CUDA conforme o passo 4.

---

## 3. Como Iniciar a Aplicação

```bash
# Ative o venv (se ainda não estiver ativo)
.\venv\Scripts\activate

# Inicie a interface web
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em **http://localhost:8501**.

Alternativamente, se o `streamlit` não estiver no PATH:

```bash
python -m streamlit run app.py
```

---

## 4. Interface Web (Streamlit)

A interface é dividida em três áreas principais:

1. **Barra Lateral (esquerda):** Todas as configurações do pipeline
2. **Tab Imagem (centro):** Upload e processamento de imagens
3. **Tab Vídeo (centro):** Upload e processamento de vídeos

### 4.1 Barra Lateral de Configurações

A barra lateral permite configurar **todos** os parâmetros do pipeline sem editar arquivos:

#### 🎯 Detecção (YOLO)
- **Modelo YOLO:** Escolha entre os modelos disponíveis em `models/yolo/` — todos treinados especificamente para detecção de placas veiculares (1 classe: `License_Plate`)
  - `yolo11n-plate.pt` (nano) = mais rápido, menor precisão
  - `yolo11s-plate.pt` (small) = equilíbrio velocidade/precisão
  - `yolo11m-plate.pt` (medium) = bom equilíbrio geral
  - `yolo11l-plate.pt` (large) = **padrão**, alta precisão ✅
  - `yolo11x-plate.pt` (extra-large) = máxima precisão, mais lento
- **Confiança Detecção:** Slider de 0.1 a 1.0 (padrão: 0.5). Valores mais altos reduzem falsos positivos mas podem perder placas.

> **Nota:** Os modelos são do repositório HuggingFace `morsetechlab/yolov11-license-plate-detection`, treinados especificamente para detecção de placas. Modelos YOLO genéricos (COCO, 80 classes) **não funcionam** para detecção de placas.

#### 📝 OCR
- **EasyOCR (primário):** Checkbox para ativar/desativar. OCR com allowlist alfanumérico (GPU).
- **TrOCR (primário):** Checkbox para ativar/desativar. OCR transformer de alta precisão (GPU).
- **Estratégia de Votação (primário):** `confidence` (melhor confiança), `majority` (voto majoritário), `all` (todos)
- **🌐 Plate Recognizer API (Fallback):** Checkbox único para ativar/desativar o fallback via API comercial. Auto-habilitado quando a chave `PLATE_RECOGNIZER_API_KEY` está configurada no `.env`.
  - **Threshold de Fallback:** Slider de 0.3 a 1.0 (padrão: 0.70). Se a confiança dos primários for menor que este valor, o fallback é acionado.
  - **Regiões:** Lista de regiões para a API (padrão: `br`)
  - **Confiança Mínima:** Confiança mínima da API para aceitar resultado

#### 🤖 LLM (Correção Inteligente)
- **Habilitar LLM:** Ativa correção via GPT-4o
- **Provider:** openai
- **Modelo:** gpt-4o (suporta visão — envia a imagem da placa junto com o texto)
- **Threshold LLM:** Confiança abaixo da qual o LLM é acionado

#### 🔍 Detecção de Adulteração
- **Habilitar:** Ativa análise forense de tampering
- **Sensibilidade:** 0.0 (baixa) a 1.0 (alta)
- **Gerar Heatmap:** Mapa de calor de regiões suspeitas
- **Gerar Anotações:** Imagem com marcações visuais

#### 📋 Forense
- **Habilitar:** Ativa módulos forenses (qualidade, classificação, laudo)
- **Modo Estrito:** Critérios mais rigorosos na avaliação
- **Gerar Laudo:** Produz relatório pericial completo

#### 🎬 Vídeo
- **Pular Frames:** Skip de N frames entre processamentos (automático pelo modo)
- **Máximo de Frames:** Limite de frames a processar (0 = sem limite)
- **Gerar Vídeo de Saída:** Produz vídeo anotado com bounding boxes

#### ⚡ Performance
- **Usar GPU (CUDA):** Ativa aceleração por GPU

#### 🎚️ Thresholds Avançados
- **Threshold Confiança OCR:** Abaixo deste valor, o resultado é marcado como incerto
- **Threshold Fallback:** Se confiança < este valor, gera alternativas combinatórias

#### 🚀 Botão "Inicializar Pipeline"
Após configurar, clique para montar o pipeline. O sistema carregará todos os modelos e exibirá o status de cada componente.

---

### 4.2 Tab Imagem

1. **Upload:** Clique em "Browse files" e selecione uma imagem contendo placa(s) veicular(es)
2. **Preview:** A imagem será exibida com informações de resolução e tamanho
3. **Processar:** Clique em "🔍 Processar Imagem"
4. **Resultados:** Para cada placa detectada, exibe:
   - Imagem da placa recortada (crop original)
   - Imagem pré-processada (versão otimizada para OCR)
   - Texto da placa reconhecido
   - Confiança do OCR (%)
   - Formato (Antigo AAA-1234 / Mercosul AAA1B23)
   - Validação (✅ Válida / ❌ Inválida)
   - Confiança por caractere (quando disponível)
   - Engine OCR utilizado
   - Alternativas de fallback (se confiança baixa)
   - Resultado de adulteração (se habilitado)
   - Classificação da placa (se habilitado)
   - Laudo pericial (se habilitado)
   - Tempo de processamento por etapa

---

### 4.3 Tab Vídeo

1. **Upload:** Selecione um arquivo de vídeo (MP4, AVI, MOV, MKV, WMV, WEBM, DAV)
2. **Modo de Análise:** Escolha entre:
   - 🅿️ **Parado** — Veículo estacionário (pula mais frames, usa early-stop)
   - 🚗 **Em Movimento** — Veículo transitando (processa mais frames)
3. **Preview:** Exibe o vídeo com informações técnicas (FPS, resolução, duração, frames)
4. **Processar:** Clique em "🎬 Processar Vídeo"
5. **Acompanhamento em Tempo Real:**
   - Barra de progresso (frame atual / total)
   - Métricas: Frames processados, Placas detectadas, Velocidade (fps), Tempo decorrido
   - Preview do frame atual (atualizado periodicamente)
6. **Resultados Finais:**
   - **Leitura Confirmada por Caractere:** Mostra caracteres com ≥ 70% de certeza; posições incertas marcadas com `*`
   - **Combinações Detectadas:** Top 10 placas mais prováveis com score composto
   - **Timeline:** Quando cada placa foi vista ao longo do tempo
   - **Gráfico de Confiança:** Evolução da confiança ao longo dos frames
   - **Melhores Frames:** Os 5 frames com maior confiança
   - **Download:** Botão para baixar vídeo anotado com bounding boxes

---

## 5. Pipeline de Processamento (10 Etapas)

### 5.1 Etapa 0 — Avaliação de Qualidade

**Módulo:** `src/image_quality.py` → classe `ImageQualityAssessor`

Avalia a qualidade da imagem antes do processamento, classificando-a em 4 níveis:

| Nível        | Score    | Descrição                                          |
|-------------|----------|----------------------------------------------------|
| 🟢 Excelente  | ≥ 0.75   | Imagem de alta qualidade, ideal para análise       |
| 🟡 Suficiente | ≥ 0.50   | Qualidade adequada, com ressalvas menores          |
| 🟠 Crítica    | ≥ 0.25   | Qualidade comprometida, limitações significativas  |
| 🔴 Insuficiente | < 0.25 | Qualidade inadequada para análise confiável         |

**Métricas analisadas:**
- **Nitidez** (Laplacian variance): Detecta blur/desfoco
- **Resolução efetiva:** Largura mínima de 80px (ideal: 300px)
- **Contraste dinâmico:** Range do histograma
- **Nível de ruído:** Estimativa de granulosidade
- **Perspectiva:** Distorção e ângulo de captura
- **Artefatos JPEG:** Compressão excessiva
- **Iluminação:** Uniformidade da luz

**Configuração no `config.yaml`:**
```yaml
forensic:
  enabled: true            # Habilitar avaliação
  strict_mode: false       # Modo estrito (critérios mais rigorosos)
  min_quality_score: 0.25  # Score mínimo para aceitar
```

---

### 5.2 Etapa 1 — Detecção (YOLO)

**Módulo:** `src/detector.py` → classe `PlateDetector`

Detecta placas na imagem usando YOLOv11. Retorna bounding boxes (x1, y1, x2, y2) com score de confiança.

**Modelos disponíveis em `models/yolo/`:**

Todos os modelos são treinados especificamente para detecção de placas veiculares (1 classe: `License_Plate`), obtidos do repositório HuggingFace [`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).

| Modelo             | Tamanho  | Velocidade  | Precisão  | Uso recomendado                |
|--------------------|----------|-------------|-----------|--------------------------------|
| yolo11n-plate.pt   | ~5.2MB   | Muito rápida| Boa       | Tempo real, dispositivos fracos|
| yolo11s-plate.pt   | ~18.3MB  | Rápida      | Melhor    | Equilíbrio geral               |
| yolo11m-plate.pt   | ~38.6MB  | Média       | Alta      | Uso geral com GPU              |
| yolo11l-plate.pt   | ~48.8MB  | Lenta       | Muito alta| **Padrão** ✅                   |
| yolo11x-plate.pt   | ~109.1MB | Mais lenta  | Máxima    | Máxima precisão                |

**Funcionalidades:**
- Auto-detecção de device (CUDA/CPU/MPS)
- Margem configurável ao redor do crop (padrão: 5%)
- Suporte a múltiplas placas por imagem
- Listagem automática de modelos disponíveis

**Configuração:**
```yaml
models:
  detector:
    confidence: 0.5              # Threshold de confiança (0-1)
    default: yolo11l-plate.pt    # Modelo padrão (treinado para placas)
    device: auto                 # auto | cpu | cuda
    dir: models/yolo             # Diretório dos modelos
```

---

### 5.3 Etapa 1.5 — Normalização Geométrica

**Módulo:** `src/geometric_normalizer.py` → classe `GeometricNormalizer`

Retifica a imagem da placa para melhorar o OCR, corrigindo distorções de captura.

**Sub-etapas:**

1. **Detecção dos 4 cantos** da placa via contornos OpenCV
2. **Correção de perspectiva** — Transformação de homografia para retificar ângulo
3. **Correção de rotação** — Hough Lines para alinhar texto horizontalmente
4. **Equalização de contraste** — CLAHE adaptativo baseado no histograma
5. **Redimensionamento padronizado** — Para 300×100 pixels (configurável)

**Configuração:**
```yaml
geometric_normalization:
  enabled: true
  perspective_correction: true
  rotation_correction: true
  contrast_equalization: true
  standard_resize: true
  target_width: 300
  target_height: 100
```

---

### 5.4 Etapa 2 — Pré-processamento

**Módulo:** `src/preprocessor.py` → classe `ImagePreprocessor`

Aplica filtros OpenCV para otimizar a imagem para OCR. Gera **múltiplas variantes** (original + versões processadas) que são todas testadas pelo OCR.

**Filtros disponíveis:**

| Filtro                   | Descrição                                              |
|--------------------------|--------------------------------------------------------|
| `enhance_contrast`       | Melhoria de contraste via CLAHE adaptativo             |
| `remove_noise`           | Remoção de ruído (bilateral ou fastNlMeans)            |
| `sharpen`                | Nitidez via filtro Laplaciano/Unsharp Mask             |
| `adaptive_threshold`     | Binarização adaptativa (Otsu + Gaussian/Mean)          |
| `morphological_cleanup`  | Limpeza morfológica pós-binarização (remove pontos, fecha gaps) |
| `deskew`                 | Correção de inclinação via Hough Lines                 |
| `multi_binarization`     | Gera múltiplas binarizações (Otsu + Mean + Gaussian)   |
| `adaptive_clahe`         | Adapta parâmetros CLAHE automaticamente ao histograma  |
| `use_nlmeans_denoising`  | Usa fastNlMeansDenoising (preserva bordas melhor)      |

**Configuração:**
```yaml
preprocessing:
  enhance_contrast: true
  remove_noise: true
  sharpen: true
  adaptive_threshold: true
  morphological_cleanup: true
  deskew: true
  multi_binarization: true
  adaptive_clahe: true
  use_nlmeans_denoising: true
```

---

### 5.5 Etapa 3 — OCR (Reconhecimento)

**Módulo:** `src/ocr/multi_engine.py` → classe `MultiEngineOCR`

Sistema de dois níveis com votação:

1. **Nível Primário (EasyOCR + TrOCR):** Sempre executados em paralelo — alta precisão
2. **Nível Fallback (Plate Recognizer API):** Acionado **somente** quando a confiança dos primários é inferior ao threshold (padrão: 70%)

Cada engine recebe as **variantes pré-processadas** e produz resultados independentes. O sistema vota entre os resultados usando a estratégia configurada.

Veja a seção [6. Engines OCR Disponíveis](#6-engines-ocr-disponíveis) para detalhes de cada engine.

---

### 5.6 Etapa 4 — Validação e Pós-processamento

**Módulo:** `src/validator.py` → classe `PlateValidator`

Verifica se o texto reconhecido pelo OCR é uma placa brasileira válida e tenta corrigi-lo automaticamente.

**O que faz:**
1. Limpa o texto (remove espaços, hífens, converte para maiúsculas)
2. Verifica contra os padrões de placa (Antigo e Mercosul)
3. Se inválido, tenta **correção automática** substituindo caracteres visualmente similares nas posições corretas
4. Verifica o prefixo contra a tabela de estados brasileiros (DENATRAN/SENATRAN) para calcular plausibilidade
5. Classifica o formato: `old` (AAA-1234), `mercosul` (AAA1B23) ou `unknown`

**Tabela de caracteres similares usada na correção:**

| Caractere | Confusões comuns |
|-----------|------------------|
| 0 (zero)  | O, D, Q          |
| 1 (um)    | I, L, 7          |
| 2         | Z                |
| 5         | S                |
| 8         | B, 3             |
| 6         | G                |
| 9         | P                |
| 4         | A                |
| 7         | T                |

**Lógica de posição:** Na placa `AAA-1234`, as posições 1-3 devem ser **letras** e 4-7 devem ser **números**. Se na posição 4 aparecer `O`, o validador substitui por `0`. Se na posição 1 aparecer `0`, substitui por `O`.

---

### 5.7 Etapa 5 — Fallback (Alternativas)

**Módulo:** `src/fallback_generator.py` → classe `FallbackGenerator`

Quando a confiança do OCR está abaixo do threshold de fallback (padrão: 80%), gera **combinações alternativas** baseadas em caracteres ambíguos.

**Como funciona:**
1. Identifica posições ambíguas (caracteres com confiança baixa que possuem similares visuais)
2. Gera todas as combinações possíveis substituindo caracteres similares
3. Filtra apenas combinações que formam placas válidas
4. Ordena por probabilidade (baseada nas confianças individuais)
5. Retorna até `max_combinations` alternativas (padrão: 10)

**Exemplo:**
```
Leitura OCR: "AB0-1234" (confiança 65%)
Posição 3 ambígua: O ou 0?

Alternativas geradas:
  1. ABO-1234 (prob: 0.70) — O na posição 3 → antigo válido ❌ (3 letras + 4 números)
  2. AB0-1234 (prob: 0.65) — 0 na posição 3 → inválido (2 letras + 5 números)
  → Corrige para: ABO-1234
```

---

### 5.8 Etapa 6 — Correção via LLM

**Módulo:** `src/llm_corrector.py` → classe `LLMCorrector`

Usa o GPT-4o da OpenAI (com visão multimodal) para corrigir placas com baixa confiança.

**O que faz:**
1. Envia o texto OCR, a confiança, as alternativas de fallback e a **imagem da placa** para o GPT-4o
2. O LLM analisa visualmente a imagem e o contexto textual
3. Retorna a placa corrigida com nova confiança
4. O resultado é re-validado contra os padrões de placa

**Quando é acionado:** Somente se `confiança_OCR < llm_threshold` (padrão: 85%)

**Requisitos:**
- Variável de ambiente `OPENAI_API_KEY` no `.env`
- Conexão com internet
- Habilitado em `config.yaml` → `llm.enabled: true`

**Configuração:**
```yaml
llm:
  enabled: true          # Habilitar LLM
  provider: openai       # Provider
  model: gpt-4o          # Modelo (suporta visão)
  temperature: 0.5       # Criatividade (0 = determinístico)
  max_tokens: 100        # Limite de resposta
  confidence_threshold: 0.85  # Aciona se conf < este valor
  use_vision: true       # Enviar imagem da placa
  timeout: 30            # Timeout em segundos
```

---

### 5.9 Etapa 7 — Detecção de Adulteração

**Módulo:** `src/tampering_detector.py` → classe `PlateTamperingDetector`

Analisa a imagem da placa em busca de sinais de adulteração física usando **9 técnicas forenses:**

| # | Técnica                    | O que detecta                                        |
|---|----------------------------|------------------------------------------------------|
| 1 | Análise de Textura (LBP)   | Padrões de textura inconsistentes (fita, adesivo)    |
| 2 | Bordas Anômalas (Canny)    | Contornos irregulares de caracteres modificados      |
| 3 | Análise de Cor/Saturação   | Diferenças de cor ou saturação entre regiões         |
| 4 | Regiões Uniformes (Blobs)  | Áreas suspeitamente uniformes (fita cobrindo texto)  |
| 5 | Análise de Frequência (FFT)| Padrões regulares de fita adesiva na frequência      |
| 6 | Consistência Geométrica    | Geometria inconsistente entre caracteres             |
| 7 | Análise de Brilho          | Reflexos anormais de material sobreposto             |
| 8 | Análise de Traço (Stroke)  | Largura/estilo inconsistente dos traços              |
| 9 | Detecção de Obstrução      | Objetos cobrindo parcialmente a placa                |

**Saídas:**
- `is_tampered`: Booleano indicando se adulteração foi detectada
- `confidence`: 0.0 (intacta) a 1.0 (certeza de adulteração)
- `tampering_type`: Lista de tipos de adulteração encontrados
- `heatmap`: Mapa de calor visual das regiões suspeitas
- `annotated_image`: Imagem com marcações das anomalias

**Configuração:**
```yaml
tampering_detection:
  enabled: true
  sensitivity: 0.5       # 0.0 = baixa, 1.0 = alta
  min_confidence: 0.4    # Mínimo para reportar
  generate_heatmap: true
  generate_annotations: true
```

---

### 5.10 Etapa 8 — Classificação da Placa

**Módulo:** `src/plate_classifier.py` → classe `PlateColorClassifier`

Classifica a placa por **tipo** e **cor** com base na análise visual:

| Tipo               | Cor do fundo       | Uso                            |
|--------------------|--------------------|--------------------------------|
| Particular         | Cinza/Prateado     | Veículo de uso particular      |
| Aluguel/Comercial  | Vermelho           | Aluguel ou transporte comercial|
| Oficial/Governo    | Azul (escuro)      | Veículo oficial do governo     |
| Diplomática        | Azul (claro)       | Missão diplomática/consular    |
| Colecionador       | Preto              | Veículo de coleção/antiguidade |
| Teste/Fabricante   | Verde              | Veículo de teste/fabricante    |
| Mercosul Padrão    | Branco             | Padrão Mercosul (desde 2018)   |

Também detecta o **padrão visual** (placa antiga vs. Mercosul) pela presença de faixa azul superior e cor dos caracteres.

---

### 5.11 Etapa 9 — Laudo Pericial Forense

**Módulo:** `src/forensic_report.py` → classe `ForensicReportGenerator`

Gera um relatório pericial forense completo e estruturado com as seguintes seções:

1. **Integridade da Imagem** — Avaliação de qualidade técnica da imagem
2. **Identificação do Veículo** — Tipo de placa, cores, classificação
3. **Leitura da Placa** — Resultado OCR com notação de incerteza `[X/Y]` para caracteres ambíguos
4. **Análise de Adulteração** — Evidências de tampering encontradas
5. **Fundamentação Técnica** — Métodos e técnicas utilizados na análise
6. **Conclusão** — Veredito final com nível de confiança
7. **Permutações** — Alternativas quando há ambiguidade

**Linguagem forense padronizada:**
- ✅ "evidências indicam" (positivo)
- ✅ "compatível com" (provável)
- ✅ "indicativo de" (possível)
- ✅ "não é possível determinar com certeza" (incerto)
- ❌ Evita: "provavelmente", "talvez", "acho que", "com certeza"

**Níveis de confiança no laudo:**

| Nível         | Score    | Descrição                            |
|---------------|----------|--------------------------------------|
| Alta          | ≥ 0.80   | Identificação positiva e segura      |
| Média         | ≥ 0.60   | Identificação provável               |
| Baixa         | ≥ 0.40   | Indicativo, porém inconclusivo       |
| Insuficiente  | < 0.40   | Dados insuficientes para conclusão   |

**Configuração:**
```yaml
forensic:
  enabled: true
  strict_mode: false
  generate_report: true        # Gerar laudo
  quality_assessment: true     # Avaliação de qualidade
  plate_classification: true   # Classificação tipo/cor
  report_format: markdown      # markdown | text
```

---

## 6. Engines OCR Disponíveis

### 6.1 EasyOCR + TrOCR (Primário)

**Módulos:** `src/ocr/easyocr_engine.py` → `EasyOCREngine` | `src/ocr/trocr_engine.py` → `TrOCREngine`

EasyOCR e TrOCR rodam **em paralelo** como engines primários.

#### EasyOCR

| Característica    | Valor                          |
|-------------------|--------------------------------|
| Arquitetura       | CRAFT detector + recognition   |
| Allowlist         | A-Z + 0-9 (36 chars)          |
| VRAM              | ~500MB                         |
| Velocidade (GPU)  | ~200ms/imagem                  |
| Input             | RGB                            |

#### TrOCR

| Característica    | Valor                                   |
|-------------------|-----------------------------------------|
| Arquitetura       | ViT encoder + Transformer decoder       |
| Modelo            | microsoft/trocr-large-printed           |
| Parâmetros        | ~558M                                   |
| VRAM              | ~1.3GB                                  |
| Velocidade (GPU)  | ~50ms/imagem                            |
| Velocidade (CPU)  | ~500ms/imagem (muito lento)             |
| Input             | RGB 384×384                             |
| Confiança         | Per-token via decoder softmax           |

**Download automático:** O modelo TrOCR é baixado do HuggingFace na primeira execução (~1.3GB).

**Configuração:**
```yaml
ocr:
  primary:
    use_easyocr: true
    easyocr_gpu: true
    easyocr_languages:
      - en
    use_trocr: true
    voting_strategy: confidence
  trocr_model: microsoft/trocr-large-printed
```

---

### 6.2 Plate Recognizer API (Fallback)

**Módulo:** `src/ocr/platerecognizer_engine.py` → classe `PlateRecognizerAPIEngine`

| Característica    | Valor                                    |
|-------------------|------------------------------------------|
| Tipo              | API REST comercial                       |
| URL               | https://api.platerecognizer.com/v1/      |
| Precisão          | Muito alta (treinado em placas)          |
| Velocidade        | ~500ms (depende da rede)                 |
| Requer            | API key (PLATE_RECOGNIZER_API_KEY)       |
| Regiões           | br, us, eu, cn, kr, au …                |

**O Plate Recognizer API é acionado como fallback** quando a confiança dos engines primários (EasyOCR + TrOCR) está abaixo do threshold configurado (padrão: 70%). Requer chave de API configurada no `.env`.

**Configuração:**
```yaml
ocr:
  fallback_ocr:
    enabled: true
    fallback_ocr_threshold: 0.70
    use_platerecognizer_api: true
    platerecognizer_api_key: ""  # Carregar via .env
    platerecognizer_regions:
      - br
    platerecognizer_min_confidence: 0.5
```

---

### 6.3 PARSeq (Opcional)

**Módulo:** `src/ocr/parseq_engine.py` → classe `PARSeqEngine`

| Característica    | Valor                             |
|---|---|
| Arquitetura       | Transformer (permutation-aware)   |
| Modelo            | baudm/parseq                      |
| Parâmetros        | ~24M                              |
| VRAM              | ~200MB                            |
| Velocidade (GPU)  | ~5ms/imagem                       |
| Confiança         | Per-character via softmax nativo  |

Motor alternativo muito mais leve que o TrOCR (24M vs 558M params) e 10x mais rápido, porém ligeiramente menos preciso. Pode ser usado como fallback adicional.

**Desabilitado por padrão.** Para habilitar:
```yaml
ocr:
  parseq:
    enabled: true
    parseq_model: parseq
    parseq_repo: baudm/parseq
```

---

### 6.4 Sistema de Votação Multi-Engine

Quando múltiplos engines retornam resultados, o `MultiEngineOCR` aplica votação para escolher o melhor:

| Estratégia    | Descrição                                                         |
|---------------|-------------------------------------------------------------------|
| `confidence`  | Escolhe o resultado com **maior confiança** (padrão) ✅            |
| `majority`    | Escolhe o texto que aparece **mais vezes** entre os engines        |
| `all`         | Retorna todos os resultados sem votação (para debug)               |

O sistema também agrupa caracteres visualmente similares (ex: O/0/D/Q pertencem ao mesmo grupo) e seleciona o caractere correto baseado no tipo esperado da posição (letra vs. número).

---

## 7. Processamento de Vídeo

**Módulo:** `src/video_processor.py` → classe `VideoProcessor`

### 7.1 Modos de Análise

| Modo          | Skip Frames | Early-Stop | Filtro Nitidez | Descrição                                  |
|---------------|-------------|-----------|----------------|---------------------------------------------|
| 🅿️ Parado    | 5           | Sim (85%) | Sim (threshold 50) | Menos frames, prioriza qualidade, para quando confiança alta |
| 🚗 Movimento  | 2           | Não       | Não            | Mais frames, captura placa em diferentes posições |

**Early-stop (modo Parado):** Se a mesma placa for detectada com confiança ≥ 85% em 3 frames consecutivos, o processamento é reduzido significativamente, pois a placa já foi identificada com certeza.

**Filtro de Nitidez (modo Parado):** Frames com Laplacian variance abaixo do threshold (50) são pulados, priorizando frames nítidos.

### 7.2 Resultados do Vídeo

O processamento de vídeo gera:

1. **Placas Únicas:** Agregação das detecções em placas únicas, limitada às Top 10 mais prováveis
2. **Leitura Confirmada:** Caracteres com ≥ 70% de certeza; incertos marcados com `*`
3. **Score Composto:** Combina número de detecções, melhor confiança e confiança média
4. **Timeline:** Registro temporal de quando cada placa foi vista
5. **Melhores Frames:** Os 5 frames com maior confiança de detecção
6. **Vídeo Anotado:** Arquivo de vídeo com bounding boxes, texto e cores:
   - 🟢 Verde: Placa válida
   - 🟠 Laranja: Placa inválida
   - 🔴 Vermelho: Placa com sinais de adulteração

**Formatos de saída de vídeo:**

| Extensão | Codec |
|----------|-------|
| .mp4     | mp4v  |
| .avi     | XVID  |
| .mov     | mp4v  |
| .mkv     | XVID  |
| .wmv     | WMV2  |
| .webm    | VP80  |
| .dav     | mp4v  |

---

## 8. Arquivo de Configuração (config.yaml)

O arquivo `config.yaml` na raiz do projeto contém **todas** as configurações centralizadas. Ele é carregado automaticamente na inicialização.

**Seções principais:**

| Seção                       | Descrição                                          |
|-----------------------------|-----------------------------------------------------|
| `models.detector`           | Modelo YOLO, confiança, device                      |
| `ocr.primary`               | Engines primários (EasyOCR + TrOCR) e estratégia de votação |
| `ocr.fallback_ocr`          | Engine fallback (Plate Recognizer API) e threshold    |
| `ocr.parseq`                | PARSeq (opcional)                                    |
| `llm`                       | OpenAI GPT-4o, threshold, visão                      |
| `pipeline`                  | Margens de crop, thresholds de confiança             |
| `preprocessing`             | Filtros de imagem (contraste, ruído, nitidez…)       |
| `geometric_normalization`   | Perspectiva, rotação, contraste, resize              |
| `tampering_detection`       | Sensibilidade, heatmap, anotações                    |
| `forensic`                  | Qualidade, classificação, laudo pericial             |
| `video`                     | Skip frames, formato, saída                          |
| `output`                    | Diretório de saída, CSV, timestamps                  |

> **Dica:** A maioria das configurações pode ser alterada pela interface web (barra lateral) sem precisar editar o arquivo diretamente.

---

## 9. Estrutura do Projeto

```
ALPR/
├── app.py                          # Entrada principal (Streamlit)
├── config.yaml                     # Configurações centralizadas
├── requirements.txt                # Dependências Python
├── pyproject.toml                  # Metadados do projeto
├── Dicas.md                        # Este documento
├── README.md                       # README principal
├── .env                            # Variáveis de ambiente (API keys) — NÃO versionar
│
├── models/
│   ├── yolo/                       # Modelos de detecção (treinados para placas)
│   │   ├── yolo11n-plate.pt        #   Nano (mais rápido)
│   │   ├── yolo11s-plate.pt        #   Small
│   │   ├── yolo11m-plate.pt        #   Medium
│   │   ├── yolo11l-plate.pt        #   Large (padrão) ✅
│   │   └── yolo11x-plate.pt        #   Extra-large (mais preciso)
│
├── src/                            # Código-fonte principal
│   ├── __init__.py                 #   Exports públicos
│   ├── constants.py                #   Caracteres similares, padrões de placa
│   ├── config_manager.py           #   Carregamento de config.yaml
│   ├── detector.py                 #   Detecção YOLO (Etapa 1)
│   ├── geometric_normalizer.py     #   Normalização geométrica (Etapa 1.5)
│   ├── preprocessor.py             #   Pré-processamento (Etapa 2)
│   ├── validator.py                #   Validação de placas (Etapa 4)
│   ├── fallback_generator.py       #   Alternativas combinatórias (Etapa 5)
│   ├── llm_corrector.py            #   Correção via GPT-4o (Etapa 6)
│   ├── tampering_detector.py       #   Detecção de adulteração (Etapa 7)
│   ├── image_quality.py            #   Avaliação de qualidade (Etapa 0/8)
│   ├── plate_classifier.py         #   Classificação tipo/cor (Etapa 8)
│   ├── forensic_report.py          #   Laudo pericial (Etapa 9)
│   ├── pipeline_lpr.py             #   Orquestrador do pipeline
│   ├── video_processor.py          #   Processamento de vídeo
│   │
│   ├── ocr/                        # Pacote OCR modular
│   │   ├── base.py                 #   Classe abstrata OCREngine
│   │   ├── easyocr_engine.py       #   EasyOCR (primário)
│   │   ├── trocr_engine.py         #   TrOCR (primário)
│   │   ├── parseq_engine.py        #   PARSeq (opcional)
│   │   ├── platerecognizer_engine.py #  Plate Recognizer API (fallback)
│   │   └── multi_engine.py         #   Orquestrador + votação
│   │
│   ├── ui/                         # Componentes de interface
│   │   ├── sidebar.py              #   Barra lateral de configuração
│   │   └── display.py              #   Exibição de resultados
│   │
│   ├── forensic/                   # Re-exports do módulo forense
│   └── tampering/                  # Re-exports do módulo de adulteração
│
├── data/
│   └── results/                    # Saídas (vídeos anotados, CSVs)
│
├── tests/                          # Testes automatizados
│   ├── conftest.py
│   ├── test_validator.py
│   ├── test_fallback_generator.py
│   ├── test_preprocessor.py
│   └── ...
│
├── scripts/                        # Scripts auxiliares
│   ├── download_rodosol.py         #   Baixar dataset RodoSol
│   └── finetune_yolo.py            #   Fine-tuning do YOLO
│
└── docs/                           # Documentação adicional
    └── FINE_TUNING_RODOSOL.md
```

---

## 10. Uso via Python (Sem Interface)

Você pode usar o pipeline diretamente via Python, sem a interface Streamlit:

### Exemplo Básico (Imagem)

```python
import cv2
from src.detector import PlateDetector
from src.preprocessor import ImagePreprocessor
from src.geometric_normalizer import GeometricNormalizer
from src.ocr.multi_engine import MultiEngineOCR
from src.validator import PlateValidator
from src.fallback_generator import FallbackGenerator
from src.pipeline_lpr import LPRPipeline

# Carregar imagem
image = cv2.imread("foto_carro.jpg")

# Montar pipeline
pipeline = LPRPipeline(
    detector=PlateDetector(model_path="models/yolo/yolo11l-plate.pt"),
    preprocessor=ImagePreprocessor(),
    ocr_engine=MultiEngineOCR(use_easyocr=True, use_trocr=True),
    validator=PlateValidator(),
    fallback_generator=FallbackGenerator(max_combinations=10),
    llm_corrector=None,  # Sem LLM
    config={
        'crop_margin': 0.05,
        'ocr_confidence_threshold': 0.6,
        'fallback_confidence_threshold': 0.8,
    },
    geometric_normalizer=GeometricNormalizer(),
)

# Processar
results = pipeline.process_image(image)

for result in results:
    print(f"Placa: {result.plate_text}")
    print(f"Confiança: {result.confidence:.1%}")
    print(f"Formato: {result.format_type}")
    print(f"Válida: {result.is_valid}")
    print(f"Engine: {result.ocr_engine}")
    print(f"Tempo: {result.processing_time_ms:.0f}ms")
    print(f"Confiança por caractere: {result.char_confidences}")
    if result.alternative_plates:
        print(f"Alternativas: {[a['text'] for a in result.alternative_plates]}")
    print("---")
```

### Exemplo Completo (Com Forense)

```python
from src.tampering_detector import PlateTamperingDetector
from src.image_quality import ImageQualityAssessor
from src.plate_classifier import PlateColorClassifier
from src.forensic_report import ForensicReportGenerator

pipeline = LPRPipeline(
    detector=PlateDetector(model_path="models/yolo/yolo11l-plate.pt"),
    preprocessor=ImagePreprocessor(),
    ocr_engine=MultiEngineOCR(use_easyocr=True, use_trocr=True),
    validator=PlateValidator(),
    fallback_generator=FallbackGenerator(),
    llm_corrector=None,
    config={'crop_margin': 0.05, 'ocr_confidence_threshold': 0.6, 'fallback_confidence_threshold': 0.8},
    geometric_normalizer=GeometricNormalizer(),
    tampering_detector=PlateTamperingDetector(sensitivity=0.5),
    quality_assessor=ImageQualityAssessor(),
    plate_classifier=PlateColorClassifier(),
    report_generator=ForensicReportGenerator(),
)

results = pipeline.process_image(image)

for result in results:
    # Resultado de adulteração
    if result.tampering_analyzed and result.tampering_result:
        print(f"Adulterada: {result.tampering_result.is_tampered}")
        print(f"Confiança: {result.tampering_result.confidence:.1%}")
    
    # Qualidade da imagem
    if result.quality_assessed and result.quality_result:
        print(f"Qualidade: {result.quality_result.quality_label} ({result.quality_result.quality_score:.1%})")
    
    # Classificação
    if result.plate_classified and result.plate_classification:
        print(f"Tipo: {result.plate_classification.plate_label}")
    
    # Laudo pericial
    if result.forensic_report:
        print(result.forensic_report.report_markdown)
```

### Exemplo de Processamento de Vídeo

```python
from src.video_processor import VideoProcessor, VehicleMode

video_proc = VideoProcessor(
    skip_frames=2,
    max_frames=0,
    generate_output_video=True,
    output_dir='data/results',
    vehicle_mode=VehicleMode.MOVING,
)

# Informações do vídeo
info = video_proc.get_video_info("video.mp4")
print(f"Duração: {info['duration_formatted']}, FPS: {info['fps']}, Frames: {info['total_frames']}")

# Processar
video_result = video_proc.process_video(
    video_path="video.mp4",
    pipeline=pipeline,
    detector_confidence=0.5,
)

# Resultados
print(f"Frames processados: {video_result.processed_frames}")
print(f"Placas únicas: {len(video_result.unique_plates)}")
print(f"Vídeo anotado: {video_result.output_video_path}")

for plate, info in video_result.unique_plates.items():
    print(f"  {info['plate_text']} — {info['total_detections']} detecções, conf: {info['best_confidence']:.1%}")
```

---

## 11. Formatos de Placas Brasileiras

O sistema reconhece dois formatos de placa:

### Formato Antigo (AAA-1234)

```
┌─────────────────────────┐
│  BRASIL                 │
│                         │
│   A B C - 1 2 3 4       │
│                         │
│  UF · CIDADE            │
└─────────────────────────┘

Posições: L L L - N N N N
  L = Letra (A-Z)
  N = Número (0-9)

Regex: ^[A-Z]{3}[0-9]{4}$

Exemplo: ABC-1234, XYZ-9876
```

### Formato Mercosul (AAA1B23)

```
┌─────────────────────────┐
│ ████ BRASIL ████████████│  ← faixa azul
│                         │
│   A B C 1 D 2 3         │
│                         │
│  BR · CIDADE · UF       │
└─────────────────────────┘

Posições: L L L N L N N
  L = Letra (A-Z)
  N = Número (0-9)

Regex: ^[A-Z]{3}[0-9][A-Z][0-9]{2}$

Exemplo: ABC1D23, XYZ4A56
```

### Tabela de Prefixos por Estado

O sistema contém uma tabela completa de faixas de placas por estado brasileiro (fonte: DENATRAN/SENATRAN) usada para validação e ranking de plausibilidade. Exemplos:

| Estado | Faixas de prefixo        |
|--------|--------------------------|
| SP     | BFA–GKI, CPA–GKZ, QWA–QZZ |
| RJ     | KMF–LVE, LAA–LZZ, QQA–QSZ |
| MG     | GKJ–HOK, HAA–HQZ, OUA–OZZ |
| PR     | AAA–BEZ, AXA–BFZ, QMA–QMZ |
| RS     | IAA–JZZ, ICA–IJZ, QUA–QUZ |
| BA     | JAA–JZZ, NUA–NZZ          |

---

## 12. Solução de Problemas Comuns

### GPU não detectada (CUDA: False)

**Causa:** PyTorch instalado sem CUDA (versão `+cpu`).

**Solução:**
```bash
# Verificar versão atual
python -c "import torch; print(torch.__version__)"
# Se mostrar "2.x.x+cpu", reinstalar:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

### TrOCR muito lento

**Causa:** PyTorch rodando em CPU (o TrOCR tem 558M parâmetros).

**Solução:** Instalar PyTorch com CUDA (veja acima). Em CPU, cada imagem leva ~500ms; em GPU, ~50ms.

### "Nenhum engine OCR disponível"

**Causa:** Nenhum engine foi inicializado com sucesso.

**Solução:**
1. Verifique se o PyTorch está instalado: `pip install torch`
2. Para EasyOCR: `pip install easyocr`
3. Para TrOCR: `pip install transformers sentencepiece`

### Vídeo não processa / Erro de codec

**Causa:** Codec do vídeo não suportado pelo OpenCV.

**Solução:**
1. Instale `pip install opencv-contrib-python` (versão contrib tem mais codecs)
2. Tente converter o vídeo para MP4/H264 antes de processar
3. Instale `pip install av` para suporte adicional

### LLM não funciona

**Causa:** API key não configurada ou inválida.

**Solução:**
1. Crie arquivo `.env` na raiz: `OPENAI_API_KEY=sk-...`
2. Habilite em `config.yaml`: `llm.enabled: true`
3. Verifique se há saldo na conta OpenAI

### TrOCR erro "Tensor on device meta"

**Causa:** Versões recentes do `transformers` usam carregamento lazy com tensores em dispositivo `meta`, que não são materializados automaticamente.

**Solução:** O sistema já inclui correção automática (`_materialize_meta_tensors`) que detecta e materializa tensores pendentes. Se o erro persistir, tente:
```bash
pip install --upgrade transformers torch
```

### Placa detectada mas texto incorreto

**Possíveis ações:**
1. **Aumente o threshold de fallback** (aciona Plate Recognizer mais frequentemente): `fallback_ocr_threshold: 0.80`
2. **Use modelo YOLO maior** (melhor crop): `yolo11l-plate.pt` ou `yolo11x-plate.pt`
3. **Habilite LLM** para correção inteligente: `llm.enabled: true`
4. **Habilite PARSeq** como fallback adicional: `parseq.enabled: true`
5. Verifique a qualidade da imagem (resolução, iluminação, ângulo)

### YOLO não detecta placas (0 detecções)

**Causa:** Modelos YOLO genéricos (COCO, 80 classes) **não possuem** a classe "placa". Somente modelos treinados especificamente para placas funcionam.

**Solução:** Certifique-se de usar os modelos `*-plate.pt` em `models/yolo/`. Eles foram obtidos do repositório HuggingFace `morsetechlab/yolov11-license-plate-detection` e possuem 1 classe: `License_Plate`.

---

## 13. Dicas de Performance

### Velocidade vs. Precisão

| Cenário                 | Modelo YOLO        | OCR            | Tempo/imagem (GPU) |
|-------------------------|--------------------|----------------|---------------------|
| Máxima velocidade       | yolo11n-plate      | EasyOCR only          | ~200ms              |
| Equilíbrio              | yolo11m-plate      | EasyOCR + TrOCR       | ~250ms              |
| Máxima precisão         | yolo11x-plate      | EasyOCR + TrOCR + LLM | ~500ms-2s           |
| **Padrão recomendado** ✅| yolo11l-plate      | EasyOCR + TrOCR       | ~250ms              |

### Otimizações para Vídeo

- **Modo Parado:** Use para câmeras fixas/estacionamento — 3x mais rápido com early-stop
- **Skip frames alto:** `skip_frames: 10` para vídeos longos com movimento lento
- **Limitar frames:** `max_frames: 500` para testes rápidos
- **Desabilitar forense:** Para vídeo, desabilite `tampering_detection` e `forensic` (são pesados e mais úteis para imagem)

### Uso de Memória

| Componente            | RAM         | VRAM (GPU)  |
|-----------------------|-------------|-------------|
| YOLO (yolo11l)        | ~200MB      | ~500MB      |
| EasyOCR               | ~500MB      | ~500MB      |
| TrOCR (large)         | ~2GB        | ~1.3GB      |
| PARSeq                | ~100MB      | ~200MB      |
| Pipeline completo     | ~3GB        | ~2.5GB      |

> **Dica:** Se sua GPU tem menos de 4GB de VRAM, considere usar `yolo11s-plate.pt` e desabilitar o TrOCR (usar apenas EasyOCR). Alternativamente, use `trocr_model: microsoft/trocr-small-printed` para uma versão menor do TrOCR.

### Testes Automatizados

Execute os testes para verificar que tudo funciona:

```bash
python -m pytest tests/ -v
```

---

> **Última atualização:** Fevereiro 2026  
> **Versão do sistema:** ALPR Forense v4.0
