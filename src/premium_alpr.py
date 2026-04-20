# src/premium_alpr.py
"""
Premium ALPR Provider — integração com APIs comerciais completas de ALPR.

Diferentemente do OCR local (que roda apenas no crop da placa já detectada),
este módulo envia a IMAGEM COMPLETA para uma API externa que executa seu
próprio pipeline completo: detecção + OCR + validação + classificação.

Usado como "Tier Premium" acionado sob demanda pelo usuário via botão na UI.
Nunca faz parte do fluxo automático.

Providers suportados:
- platerecognizer: https://platerecognizer.com/ (Plate Recognizer API)
"""

import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PremiumALPRResult:
    """Resultado de uma chamada à API Premium."""
    success: bool
    plate_text: str = ''
    format_type: str = 'unknown'  # 'old' | 'mercosul' | 'unknown'
    confidence: float = 0.0
    region: str = ''
    vehicle_type: str = ''
    bbox: Optional[List[List[int]]] = None  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    provider: str = ''
    api_cost_calls: int = 0  # +1 em cada chamada bem-sucedida
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    alternates: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.success and len(self.plate_text) >= 7


class PremiumALPRProvider:
    """
    Provider para APIs comerciais de ALPR.

    Uso:
        provider = PremiumALPRProvider(
            provider='platerecognizer',
            api_key=os.getenv('PLATE_RECOGNIZER_API_KEY'),
            regions=['br'],
        )
        result = provider.analyze_full_image(image)
        if result.is_valid:
            print(result.plate_text)
    """

    def __init__(
        self,
        provider: str = 'platerecognizer',
        api_key: str = '',
        regions: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        timeout: int = 30,
        enabled: bool = True,
        log_all_calls: bool = True,
        structured_logger: Any = None,
    ):
        """
        Args:
            provider: Nome do provider ('platerecognizer').
            api_key: Chave de API.
            regions: Lista de códigos de região (['br'] para Brasil).
            min_confidence: Confiança mínima para considerar resultado válido.
            timeout: Timeout HTTP em segundos.
            enabled: Se o provider está ativo.
            log_all_calls: Se deve registrar cada chamada no structured_logger.
            structured_logger: Instância de StructuredLogger (opcional).
        """
        self.provider = provider.lower()
        self.api_key = api_key or ''
        self.regions = regions or ['br']
        self.min_confidence = min_confidence
        self.timeout = timeout
        self.enabled = enabled
        self.log_all_calls = log_all_calls
        self.structured_logger = structured_logger
        self.total_calls = 0
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Verifica se o provider pode ser usado."""
        if not self.enabled:
            return False
        if self.provider != 'platerecognizer':
            logger.warning(f"Provider desconhecido: {self.provider}")
            return False
        if not self.api_key or len(self.api_key) < 10 or self.api_key == 'sua_chave_aqui':
            logger.debug("API Premium: API key não configurada")
            return False

        # Verificar conectividade (não é fatal se falhar agora)
        try:
            resp = httpx.get(
                'https://api.platerecognizer.com/v1/statistics/',
                headers={'Authorization': f'Token {self.api_key}'},
                timeout=5.0,
            )
            if resp.status_code == 200:
                stats = resp.json().get('usage', {})
                logger.info(
                    f"✅ API Premium (Plate Recognizer) disponível. "
                    f"Uso: {stats.get('calls', 0)}/{stats.get('max_calls', 'N/A')}"
                )
                return True
            if resp.status_code == 401:
                logger.error("API Premium: chave inválida (401)")
                return False
            logger.warning(f"API Premium retornou status {resp.status_code}")
            return True  # Ainda tentamos na chamada real
        except Exception as e:
            logger.warning(f"API Premium: falha ao verificar: {e}")
            return True  # Não bloquear; tenta no uso real

    def analyze_full_image(self, image: np.ndarray) -> PremiumALPRResult:
        """
        Envia a IMAGEM COMPLETA para a API e retorna o resultado.

        Args:
            image: Imagem BGR (numpy array).

        Returns:
            PremiumALPRResult com o resultado estruturado.
        """
        if not self.enabled:
            return PremiumALPRResult(
                success=False,
                provider=self.provider,
                error='Provider desabilitado',
            )

        if not self.available:
            return PremiumALPRResult(
                success=False,
                provider=self.provider,
                error='API key ausente ou inválida',
            )

        if image is None or image.size == 0:
            return PremiumALPRResult(
                success=False,
                provider=self.provider,
                error='Imagem vazia',
            )

        if self.provider == 'platerecognizer':
            result = self._call_platerecognizer(image)
        else:
            result = PremiumALPRResult(
                success=False,
                provider=self.provider,
                error=f'Provider {self.provider} não implementado',
            )

        # Registrar no structured_logger
        if self.log_all_calls and self.structured_logger:
            try:
                self.structured_logger._write_event({
                    'event_type': 'premium_api_call',
                    'provider': self.provider,
                    'success': result.success,
                    'plate_text': result.plate_text,
                    'confidence': round(result.confidence, 4),
                    'format_type': result.format_type,
                    'region': result.region,
                    'vehicle_type': result.vehicle_type,
                    'error': result.error,
                })
            except Exception as e:
                logger.debug(f"Falha ao logar chamada Premium: {e}")

        return result

    def _call_platerecognizer(self, image: np.ndarray) -> PremiumALPRResult:
        """Chama a API do Plate Recognizer com a imagem completa."""
        try:
            # Codificar imagem como JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = BytesIO(buffer.tobytes())

            files = {'upload': ('image.jpg', image_bytes, 'image/jpeg')}
            data = {'regions': self.regions}
            headers = {'Authorization': f'Token {self.api_key}'}

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    'https://api.platerecognizer.com/v1/plate-reader/',
                    files=files,
                    data=data,
                    headers=headers,
                )

            if resp.status_code == 429:
                return PremiumALPRResult(
                    success=False,
                    provider='platerecognizer',
                    error='Limite de requisições excedido (429)',
                )
            if resp.status_code == 402:
                return PremiumALPRResult(
                    success=False,
                    provider='platerecognizer',
                    error='Créditos da API esgotados (402)',
                )

            resp.raise_for_status()
            data_json = resp.json()

            self.total_calls += 1

            results = data_json.get('results', [])
            if not results:
                return PremiumALPRResult(
                    success=True,
                    provider='platerecognizer',
                    error='Nenhuma placa detectada pela API',
                    raw_response=data_json,
                    api_cost_calls=1,
                )

            # Pegar o melhor resultado
            best = max(results, key=lambda r: r.get('score', 0))
            plate_raw = best.get('plate', '').strip().upper()
            plate_clean = plate_raw.replace(' ', '').replace('-', '')
            score = float(best.get('score', 0.0))

            if score < self.min_confidence:
                return PremiumALPRResult(
                    success=True,
                    provider='platerecognizer',
                    plate_text=plate_clean,
                    confidence=score,
                    error=f'Confiança abaixo do mínimo ({score:.2f} < {self.min_confidence:.2f})',
                    raw_response=data_json,
                    api_cost_calls=1,
                )

            # Determinar formato
            import re
            if re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', plate_clean):
                fmt = 'mercosul'
            elif re.match(r'^[A-Z]{3}[0-9]{4}$', plate_clean):
                fmt = 'old'
            else:
                fmt = 'unknown'

            # Bbox
            box = best.get('box', {})
            bbox = None
            if box:
                bbox = [
                    [box.get('xmin', 0), box.get('ymin', 0)],
                    [box.get('xmax', 0), box.get('ymin', 0)],
                    [box.get('xmax', 0), box.get('ymax', 0)],
                    [box.get('xmin', 0), box.get('ymax', 0)],
                ]

            return PremiumALPRResult(
                success=True,
                plate_text=plate_clean,
                format_type=fmt,
                confidence=score,
                region=best.get('region', {}).get('code', ''),
                vehicle_type=best.get('vehicle', {}).get('type', ''),
                bbox=bbox,
                provider='platerecognizer',
                api_cost_calls=1,
                raw_response=data_json,
                alternates=[
                    {
                        'plate': c.get('plate', ''),
                        'score': c.get('score', 0.0),
                    }
                    for c in best.get('candidates', [])[:5]
                ],
            )

        except Exception as e:
            logger.error(f"Erro em PremiumALPR._call_platerecognizer: {e}")
            return PremiumALPRResult(
                success=False,
                provider='platerecognizer',
                error=str(e),
            )
