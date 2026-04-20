# Plate Recognizer API no ALPR 2.0

Este documento explica como o fluxo Premium funciona no projeto e como interpretar a configuracao de confianca minima.

## Papel da API no projeto

No ALPR 2.0, a Plate Recognizer nao e um fallback automatico do OCR local.

Ela funciona como um segundo fluxo, acionado manualmente, para comparar o resultado local com uma API externa que ja faz seu proprio pipeline completo de:

- deteccao da placa
- OCR
- classificacao de regiao
- candidatos alternativos
- metadados do veiculo

No projeto atual:

- o fluxo local roda primeiro, se o usuario quiser
- o fluxo Premium e opcional
- o resultado Premium aparece lado a lado com o local
- a UI nao expoe a chave da API; ela deve ficar no `.env`

## Quantas consultas o usuario tem de graca

No momento desta documentacao, o plano `FREE` do `Plate Recognizer Snapshot` oferece:

- `2.500 lookups por mes`

Pontos importantes:

- isso e uma `free trial`, pensada para avaliacao
- a propria pagina de pricing informa que a free trial nao deve ser usada em producao
- no plano gratuito, a API Cloud tem limite de `1 lookup por segundo`
- nos planos pagos, a documentacao oficial informa `8 lookups por segundo` por padrao

No contexto deste projeto, cada clique no botao `Analisar com Plate Recognizer` consome 1 lookup.

## Como a API Plate Recognizer funciona

O projeto usa a linha `Snapshot Cloud API`.

### Autenticacao

A API usa token no header HTTP:

```http
Authorization: Token SUA_API_KEY
```

### Endpoint principal

O endpoint principal para leitura de placas em imagem e:

```http
POST https://api.platerecognizer.com/v1/plate-reader/
```

Para evitar ambiguidade:

- o fluxo `FREE` considerado neste documento nao faz consulta de video
- no projeto atual, a integracao Premium com Plate Recognizer e apenas para imagem
- videos continuam sendo tratados pelo pipeline local do ALPR 2.0

### O que o projeto envia

O projeto atual envia:

- a `imagem completa` em multipart/form-data
- o parametro `regions`, hoje normalmente com `['br']`

Internamente, `src/premium_alpr.py` faz:

1. codifica a imagem como JPEG
2. monta o payload multipart com `upload=@image.jpg`
3. envia o token no header
4. envia `regions=['br']`
5. pega o melhor item de `results[]` com maior `score`
6. filtra esse item pelo `min_confidence` configurado no projeto

### Resposta principal da API

Os campos mais importantes da resposta oficial sao:

- `results[].plate`: texto da placa
- `results[].score`: confianca do OCR da placa
- `results[].dscore`: confianca da deteccao da placa
- `results[].box`: bbox da placa
- `results[].region.code`: codigo da regiao estimada
- `results[].vehicle.type`: tipo do veiculo
- `results[].candidates`: candidatos alternativos

O projeto converte isso para `PremiumALPRResult`.

### Estatisticas de uso

Para consultar consumo, a API tem:

```http
GET https://api.platerecognizer.com/v1/statistics/
```

A resposta inclui campos como:

- `usage.calls`: quantas chamadas voce ja consumiu no periodo atual
- `usage.resets_on`: quando a contagem reseta
- `total_calls`: cota maxima do plano atual

O projeto usa esse endpoint em `_check_availability()` para validar credencial e ter uma ideia do uso atual.

## Configuracao no projeto

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

Na aba de imagem, use o botao `Analisar com Plate Recognizer`.

## Como o projeto interpreta o `min_confidence`

O `min_confidence` do projeto nao muda o threshold interno do motor da Plate Recognizer na nuvem.

Ele funciona como filtro local depois da resposta.

Ou seja:

- a API responde com um `score`
- o projeto pega o melhor resultado
- se `score < premium_api.min_confidence`, o resultado volta com `success=True`, mas marcado como abaixo do minimo

Isso e importante porque voce continua vendo que a API respondeu, mas a leitura pode ser tratada como fraca para fins operacionais.

## Melhor confianca minima Premium

Se a pergunta for "qual valor usar por padrao neste projeto", a melhor resposta pratica e:

- `0.70` como ponto de partida mais equilibrado

Motivo:

- `0.50` e permissivo demais para uso operacional e tende a deixar passar leituras mais duvidosas
- `0.70` costuma equilibrar melhor `recall` e `precisao` quando a API esta sendo usada como segunda opiniao
- `0.80` ou mais reduz falso positivo, mas comeca a perder placas mais dificeis, sobretudo em baixa iluminacao, blur ou distancia

Resumo pratico:

| Objetivo | Valor sugerido |
|---|---|
| comparacao exploratoria, POC, nao perder candidatos | `0.50` |
| uso geral no ALPR 2.0, com comparacao lado a lado | `0.70` |
| fluxo mais conservador, evitando falso positivo | `0.80` |

Se voce estiver usando a API so para comparar com o pipeline local, `0.70` tende a ser o melhor default.

Se estiver fazendo triagem humana depois, `0.50` ainda faz sentido.

## Melhor pratica para `regions`

A documentacao oficial recomenda informar os estados ou paises mais provaveis observados pela camera, tipicamente os 3 ou 4 principais.

No projeto atual, o default e:

- `['br']`

Isso faz sentido quando o alvo principal sao placas brasileiras.

Se sua camera observa frota mista de paises vizinhos, vale ampliar esse campo.

## Erros comuns

### Token invalido ou problema de credito

Na documentacao oficial da Snapshot Cloud, erros de credencial ou falta de creditos podem aparecer como `403`.

Na pratica, dependendo do endpoint, plano ou momento da verificacao, integracoes tambem podem ver respostas como `401` ou `402`.

### Rate limit

`429` significa excesso de requisicoes. Na free trial, o limite oficial informado e `1 lookup por segundo`.

### Nenhuma placa detectada

Nesse caso, a chamada HTTP pode ter funcionado normalmente, mas `results[]` vem vazio.

## Uso programatico

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

- plano gratis atual: `2.500 lookups/mes`
- uso gratis e para avaliacao, nao para producao
- plano `FREE` considerado aqui nao cobre consulta de video
- endpoint principal: `POST /v1/plate-reader/`
- o projeto envia a imagem completa, nao o crop
- `min_confidence` e filtro local aplicado depois da resposta
- melhor default pratico para este projeto: `0.70`

## Referencias oficiais

- https://platerecognizer.com/pricing/
- https://guides.platerecognizer.com/docs/snapshot/api-reference
- https://guides.platerecognizer.com/docs/tech-references/country-codes
