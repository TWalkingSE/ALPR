# Plate Recognizer API no ALPR 2.0

Este documento explica como o fluxo Premium funciona no projeto e como interpretar a configuração de confiança mínima.

## Papel da API no projeto

No ALPR 2.0, a Plate Recognizer não é um fallback automático do OCR local.

Ela funciona como um segundo fluxo, acionado manualmente, para comparar o resultado local com uma API externa que já faz seu próprio pipeline completo de:

- detecção da placa
- OCR
- classificação de região
- candidatos alternativos
- metadados do veículo

No projeto atual:

- o fluxo local roda primeiro, se o usuário quiser
- o fluxo Premium é opcional
- o resultado Premium aparece lado a lado com o local
- a UI não expõe a chave da API; ela deve ficar no `.env`

## Quantas consultas o usuário tem de graça

No momento desta documentação, o plano `FREE` do `Plate Recognizer Snapshot` oferece:

- `2.500 lookups por mês`

Pontos importantes:

- isso é uma `free trial`, pensada para avaliação
- a própria página de pricing informa que a free trial não deve ser usada em produção
- no plano gratuito, a API Cloud tem limite de `1 lookup por segundo`
- nos planos pagos, a documentação oficial informa `8 lookups por segundo` por padrão

No contexto deste projeto, cada clique no botão `Analisar com Plate Recognizer` consome 1 lookup.

## Como a API Plate Recognizer funciona

O projeto usa a linha `Snapshot Cloud API`.

### Autenticação

A API usa token no header HTTP:

```http
Authorization: Token SUA_API_KEY
```

### Endpoint principal

O endpoint principal para leitura de placas em imagem é:

```http
POST https://api.platerecognizer.com/v1/plate-reader/
```

Para evitar ambiguidade:

- o fluxo `FREE` considerado neste documento não faz consulta de vídeo
- no projeto atual, a integração Premium com Plate Recognizer é apenas para imagem
- vídeos continuam sendo tratados pelo pipeline local do ALPR 2.0

### O que o projeto envia

O projeto atual envia:

- a `imagem completa` em multipart/form-data
- o parâmetro `regions`, hoje normalmente com `['br']`

Internamente, `src/premium_alpr.py` faz:

1. codifica a imagem como JPEG
2. monta o payload multipart com `upload=@image.jpg`
3. envia o token no header
4. envia `regions=['br']`
5. pega o melhor item de `results[]` com maior `score`
6. filtra esse item pelo `min_confidence` configurado no projeto

### Resposta principal da API

Os campos mais importantes da resposta oficial são:

- `results[].plate`: texto da placa
- `results[].score`: confiança do OCR da placa
- `results[].dscore`: confiança da detecção da placa
- `results[].box`: bbox da placa
- `results[].region.code`: código da região estimada
- `results[].vehicle.type`: tipo do veículo
- `results[].candidates`: candidatos alternativos

O projeto converte isso para `PremiumALPRResult`.

### Estatísticas de uso

Para consultar consumo, a API tem:

```http
GET https://api.platerecognizer.com/v1/statistics/
```

A resposta inclui campos como:

- `usage.calls`: quantas chamadas você já consumiu no período atual
- `usage.resets_on`: quando a contagem reseta
- `total_calls`: cota máxima do plano atual

O projeto usa esse endpoint em `_check_availability()` para validar credencial e ter uma ideia do uso atual.

## Configuração no projeto

### 1. Criar ou editar o `.env`

```env
PLATE_RECOGNIZER_API_KEY=sua_chave_real_aqui
```

### 2. Habilitar o fluxo Premium

No `config.yaml`:

```yaml
premium_api:
  enabled: true
  provider: platerecognizer
  regions:
    - br
  min_confidence: 0.5
  timeout: 30
```

### 3. Rodar a interface

```bash
streamlit run app.py
```

Na aba de imagem, use o botão `Analisar com Plate Recognizer`.

## Como o projeto interpreta o `min_confidence`

O `min_confidence` do projeto não muda o threshold interno do motor da Plate Recognizer na nuvem.

Ele funciona como filtro local depois da resposta.

Ou seja:

- a API responde com um `score`
- o projeto pega o melhor resultado
- se `score < premium_api.min_confidence`, o resultado volta com `success=True`, mas marcado como abaixo do mínimo

Isso é importante porque você continua vendo que a API respondeu, mas a leitura pode ser tratada como fraca para fins operacionais.

## Melhor confiança mínima Premium

Se a pergunta for "qual valor usar por padrão neste projeto", a melhor resposta prática é:

- `0.70` como ponto de partida mais equilibrado

Motivo:

- `0.50` é permissivo demais para uso operacional e tende a deixar passar leituras mais duvidosas
- `0.70` costuma equilibrar melhor `recall` e `precisão` quando a API está sendo usada como segunda opinião
- `0.80` ou mais reduz falso positivo, mas começa a perder placas mais difíceis, sobretudo em baixa iluminação, blur ou distância

Resumo prático:

| Objetivo | Valor sugerido |
|---|---|
| comparação exploratória, POC, não perder candidatos | `0.50` |
| uso geral no ALPR 2.0, com comparação lado a lado | `0.70` |
| fluxo mais conservador, evitando falso positivo | `0.80` |

Se você estiver usando a API só para comparar com o pipeline local, `0.70` tende a ser o melhor default.

Se estiver fazendo triagem humana depois, `0.50` ainda faz sentido.

## Melhor prática para `regions`

A documentação oficial recomenda informar os estados ou países mais prováveis observados pela câmera, tipicamente os 3 ou 4 principais.

No projeto atual, o default é:

- `['br']`

Isso faz sentido quando o alvo principal são placas brasileiras.

Se sua câmera observa frota mista de países vizinhos, vale ampliar esse campo.

## Erros comuns

### Token inválido ou problema de crédito

Na documentação oficial da Snapshot Cloud, erros de credencial ou falta de créditos podem aparecer como `403`.

Na prática, dependendo do endpoint, plano ou momento da verificação, integrações também podem ver respostas como `401` ou `402`.

### Rate limit

`429` significa excesso de requisições. Na free trial, o limite oficial informado é `1 lookup por segundo`.

### Nenhuma placa detectada

Nesse caso, a chamada HTTP pode ter funcionado normalmente, mas `results[]` vem vazio.

## Uso programático

```python
import os
import cv2
from src.premium_alpr import PremiumALPRProvider

image = cv2.imread("foto.jpg")

premium = PremiumALPRProvider(
    provider='platerecognizer',
    api_key=os.getenv('PLATE_RECOGNIZER_API_KEY'),
    regions=['br'],
    min_confidence=0.70,
    timeout=30,
)

result = premium.analyze_full_image(image)

if result.is_valid:
    print(result.plate_text, result.confidence)
else:
    print(result.error)
```

## Resumo objetivo

- plano grátis atual: `2.500 lookups/mês`
- uso grátis é para avaliação, não para produção
- plano `FREE` considerado aqui não cobre consulta de vídeo
- endpoint principal: `POST /v1/plate-reader/`
- o projeto envia a imagem completa, não o crop
- `min_confidence` é filtro local aplicado depois da resposta
- melhor default prático para este projeto: `0.70`

## Referências oficiais

- https://platerecognizer.com/pricing/
- https://guides.platerecognizer.com/docs/snapshot/api-reference
- https://guides.platerecognizer.com/docs/tech-references/country-codes
