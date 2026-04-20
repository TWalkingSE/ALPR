# tests/test_detector.py
"""
Testes para o PlateDetector (src/detector.py).

Cobre:
  - Inicialização e validação de parâmetros
  - _ensure_bgr() — conversão de canais de entrada
  - _parse_yolo_results() — normalização de saídas YOLO
  - _nms_detections() — supressão de duplicatas
  - extract_plate_regions() — margem adaptativa e upscale
  - list_available_models() — scan de diretório
  - detect() — pipeline completo com modelo mockado
  - SAHI — detect com slicing
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.detector import PlateDetector


# ============================================================================
# Helpers
# ============================================================================

def _make_bgr(h: int = 100, w: int = 200) -> np.ndarray:
    """Cria imagem BGR dummy."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_yolo_box(x1, y1, x2, y2, conf, cls_id=0):
    """Cria mock de box YOLO com atributos .xyxy, .conf, .cls."""
    box = MagicMock()
    # box.xyxy[0].tolist() → [x1, y1, x2, y2]
    xyxy = MagicMock()
    xyxy.tolist.return_value = [x1, y1, x2, y2]
    box.xyxy = [xyxy]
    # box.conf[0] → float
    box.conf = [float(conf)]
    # box.cls[0] → int
    box.cls = [int(cls_id)]
    return box


def _make_yolo_result(boxes: list, names: dict | None = None):
    """Cria mock de resultado YOLO com .boxes (iterable)."""
    result = MagicMock()
    result.boxes = boxes if boxes else None
    result.names = names or {0: 'plate'}
    return result


@pytest.fixture
def mock_yolo(tmp_path):
    """Patch `ultralytics.YOLO` + cria arquivo fake .pt para passar no exists() check."""
    fake_model = tmp_path / "fake.pt"
    fake_model.write_bytes(b"dummy")

    # MagicMock da classe YOLO: quando instanciada, retorna um MagicMock que é
    # callable (para model(image, conf=..)) e tem .to(device)
    mock_yolo_instance = MagicMock()
    mock_yolo_instance.to = MagicMock()
    # model(image, conf=..) retorna lista vazia por padrão
    mock_yolo_instance.return_value = []

    mock_yolo_cls = MagicMock(return_value=mock_yolo_instance)

    with patch.dict('sys.modules', {'ultralytics': MagicMock(YOLO=mock_yolo_cls)}):
        yield {'cls': mock_yolo_cls, 'instance': mock_yolo_instance, 'path': str(fake_model)}


@pytest.fixture
def detector(mock_yolo):
    """Detector com YOLO mockado."""
    det = PlateDetector(
        model_path=mock_yolo['path'],
        confidence=0.5,
        device='cpu',
        enable_sahi=False,
    )
    return det


# ============================================================================
# Inicialização
# ============================================================================

class TestInitialization:
    def test_missing_model_path_raises(self):
        with pytest.raises(ValueError, match="model_path"):
            PlateDetector(model_path=None)

    def test_valid_init(self, mock_yolo):
        d = PlateDetector(model_path=mock_yolo['path'], confidence=0.3, device='cpu')
        assert d.model_path == mock_yolo['path']
        assert d.confidence == 0.3
        assert d.device_str == 'cpu'
        assert d.is_loaded  # mockado

    def test_device_auto_detects(self, mock_yolo):
        with patch('torch.cuda.is_available', return_value=False):
            d = PlateDetector(model_path=mock_yolo['path'], device='auto')
            assert d.device_str == 'cpu'

    def test_device_explicit_cpu(self, mock_yolo):
        # Testamos explicit cpu (não cuda) para evitar issues com fallback
        d = PlateDetector(model_path=mock_yolo['path'], device='cpu')
        assert d.device_str == 'cpu'

    def test_sahi_params_stored(self, mock_yolo):
        d = PlateDetector(
            model_path=mock_yolo['path'], device='cpu',
            enable_sahi=True, sahi_slice_size=512, sahi_overlap_ratio=0.3,
        )
        assert d.enable_sahi is True
        assert d.sahi_slice_size == 512
        assert d.sahi_overlap_ratio == 0.3


# ============================================================================
# _ensure_bgr — normalização de canais
# ============================================================================

class TestEnsureBGR:
    def test_bgr_unchanged(self, detector):
        img = _make_bgr(100, 200)
        result = detector._ensure_bgr(img)
        assert result.shape == (100, 200, 3)

    def test_grayscale_1d_converts_to_bgr(self, detector):
        img = np.zeros((100, 200), dtype=np.uint8)
        result = detector._ensure_bgr(img)
        assert result.shape == (100, 200, 3)

    def test_hwx1_converts_to_bgr(self, detector):
        img = np.zeros((100, 200, 1), dtype=np.uint8)
        result = detector._ensure_bgr(img)
        assert result.shape == (100, 200, 3)

    def test_bgra_4channels_converts_to_bgr(self, detector):
        img = np.zeros((100, 200, 4), dtype=np.uint8)
        result = detector._ensure_bgr(img)
        assert result.shape == (100, 200, 3)


# ============================================================================
# _parse_yolo_results — parseamento
# ============================================================================

class TestParseYoloResults:
    def test_empty_results_returns_empty(self, detector):
        assert detector._parse_yolo_results([]) == []
        assert detector._parse_yolo_results(None) == []

    def test_single_detection(self, detector):
        box = _make_yolo_box(10, 20, 100, 80, conf=0.85)
        result = _make_yolo_result([box])
        detections = detector._parse_yolo_results([result])
        assert len(detections) == 1
        assert detections[0]['bbox'] == (10, 20, 100, 80)
        assert detections[0]['confidence'] == pytest.approx(0.85)
        assert detections[0]['class_id'] == 0
        assert detections[0]['class_name'] == 'plate'

    def test_multiple_detections(self, detector):
        boxes = [
            _make_yolo_box(10, 20, 100, 80, 0.9),
            _make_yolo_box(200, 50, 300, 120, 0.75),
        ]
        result = _make_yolo_result(boxes)
        detections = detector._parse_yolo_results([result])
        assert len(detections) == 2

    def test_invalid_bbox_filtered(self, detector):
        # x1 >= x2 deve ser ignorado
        boxes = [_make_yolo_box(100, 50, 50, 80, conf=0.9)]
        result = _make_yolo_result(boxes)
        detections = detector._parse_yolo_results([result])
        assert len(detections) == 0

    def test_negative_coords_filtered(self, detector):
        boxes = [_make_yolo_box(-10, -5, 100, 80, conf=0.9)]
        result = _make_yolo_result(boxes)
        detections = detector._parse_yolo_results([result])
        assert len(detections) == 0


# ============================================================================
# _nms_detections — NMS
# ============================================================================

class TestNMS:
    def test_empty_returns_empty(self, detector):
        assert detector._nms_detections([]) == []

    def test_single_detection_unchanged(self, detector):
        dets = [{'bbox': (10, 20, 100, 80), 'confidence': 0.9, 'class_id': 0, 'class_name': 'plate'}]
        result = detector._nms_detections(dets)
        assert len(result) == 1
        assert result[0]['bbox'] == (10, 20, 100, 80)

    def test_overlapping_duplicates_suppressed(self, detector):
        # Dois bboxes quase idênticos — NMS deve manter apenas um
        dets = [
            {'bbox': (10, 20, 100, 80), 'confidence': 0.95, 'class_id': 0, 'class_name': 'plate'},
            {'bbox': (12, 22, 102, 82), 'confidence': 0.80, 'class_id': 0, 'class_name': 'plate'},
        ]
        result = detector._nms_detections(dets, iou_threshold=0.4)
        # Deve manter o de maior confiança
        assert len(result) == 1
        assert result[0]['confidence'] == pytest.approx(0.95)

    def test_non_overlapping_both_kept(self, detector):
        dets = [
            {'bbox': (10, 20, 100, 80), 'confidence': 0.9, 'class_id': 0, 'class_name': 'plate'},
            {'bbox': (300, 200, 400, 260), 'confidence': 0.8, 'class_id': 0, 'class_name': 'plate'},
        ]
        result = detector._nms_detections(dets, iou_threshold=0.4)
        assert len(result) == 2


# ============================================================================
# extract_plate_regions — recorte e upscale
# ============================================================================

class TestExtractPlateRegions:
    def test_empty_image_returns_empty(self, detector):
        result = detector.extract_plate_regions(
            np.array([]), [{'bbox': (0, 0, 10, 10), 'confidence': 0.9}]
        )
        assert result == []

    def test_basic_crop(self, detector):
        img = _make_bgr(300, 400)
        detections = [{'bbox': (100, 100, 300, 180), 'confidence': 0.9}]
        regions = detector.extract_plate_regions(img, detections, add_margin=0.0)
        assert len(regions) == 1
        assert 'image' in regions[0]
        assert 'bbox' in regions[0]
        assert regions[0]['confidence'] == pytest.approx(0.9)

    def test_crop_with_margin(self, detector):
        img = _make_bgr(500, 500)
        detections = [{'bbox': (200, 200, 300, 280), 'confidence': 0.9}]
        regions = detector.extract_plate_regions(img, detections, add_margin=0.10)
        assert len(regions) == 1
        # Com margem, bbox resultante deve ser maior que o original
        x1, y1, x2, y2 = regions[0]['bbox']
        ox1, oy1, ox2, oy2 = regions[0]['original_bbox']
        assert x2 - x1 >= ox2 - ox1

    def test_adaptive_margin_for_small_plate(self, detector):
        # Placa pequena (h < SMALL_PLATE_HEIGHT_THRESHOLD) deve receber margem maior
        img = _make_bgr(500, 500)
        detections = [{'bbox': (200, 200, 240, 230), 'confidence': 0.9}]  # 40x30 placa
        regions = detector.extract_plate_regions(img, detections, add_margin=0.05)
        assert len(regions) == 1
        # Deve ter aplicado margem adaptativa (> 5%)
        x1, y1, x2, y2 = regions[0]['bbox']
        assert (x2 - x1) > 40  # maior que a placa original

    def test_crop_upscale_small(self, detector):
        # Crop menor que MIN_PLATE_HEIGHT_FOR_OCR deve ser upscaled
        img = _make_bgr(500, 500)
        detections = [{'bbox': (200, 200, 300, 250), 'confidence': 0.9}]  # 100x50
        regions = detector.extract_plate_regions(img, detections, add_margin=0.0)
        assert len(regions) == 1
        crop = regions[0]['image']
        # Altura resultante deve ser >= MIN_PLATE_HEIGHT_FOR_OCR (80)
        assert crop.shape[0] >= PlateDetector.MIN_PLATE_HEIGHT_FOR_OCR

    def test_bbox_clamped_to_image(self, detector):
        img = _make_bgr(100, 100)
        # Bbox que extrapola a imagem com margem
        detections = [{'bbox': (80, 80, 120, 120), 'confidence': 0.9}]
        regions = detector.extract_plate_regions(img, detections, add_margin=0.5)
        assert len(regions) == 1
        x1, y1, x2, y2 = regions[0]['bbox']
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 100 and y2 <= 100

    def test_multiple_detections(self, detector):
        img = _make_bgr(500, 500)
        detections = [
            {'bbox': (50, 50, 200, 150), 'confidence': 0.9},
            {'bbox': (250, 250, 400, 350), 'confidence': 0.85},
        ]
        regions = detector.extract_plate_regions(img, detections)
        assert len(regions) == 2


# ============================================================================
# list_available_models — scan de filesystem
# ============================================================================

class TestListAvailableModels:
    def test_missing_dir_returns_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert PlateDetector.list_available_models(str(missing)) == []

    def test_finds_pt_files_non_recursive(self, tmp_path):
        (tmp_path / "model1.pt").touch()
        (tmp_path / "model2.pt").touch()
        (tmp_path / "ignored.txt").touch()
        models = PlateDetector.list_available_models(str(tmp_path), recursive=False)
        assert len(models) == 2
        assert all(m.endswith('.pt') for m in models)

    def test_finds_pt_files_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.pt").touch()
        (sub / "b.pt").touch()
        models = PlateDetector.list_available_models(str(tmp_path), recursive=True)
        assert len(models) == 2

    def test_ignores_non_pt_files(self, tmp_path):
        (tmp_path / "config.yaml").touch()
        (tmp_path / "readme.md").touch()
        models = PlateDetector.list_available_models(str(tmp_path))
        assert models == []


# ============================================================================
# detect() — pipeline completo com model mockado
# ============================================================================

class TestDetect:
    def test_not_loaded_raises(self, mock_yolo):
        d = PlateDetector(model_path=mock_yolo['path'], device='cpu')
        d._is_loaded = False
        d.model = None
        with pytest.raises(RuntimeError, match="não carregado"):
            d.detect(_make_bgr())

    def test_empty_image_returns_empty(self, detector):
        assert detector.detect(np.array([])) == []

    def test_detect_returns_parsed(self, detector):
        # Configurar mock: quando chamado, retorna 1 detecção
        box = _make_yolo_box(10, 20, 100, 80, conf=0.9)
        yolo_result = _make_yolo_result([box])
        detector.model.return_value = [yolo_result]

        img = _make_bgr(300, 400)
        detections = detector.detect(img)
        assert len(detections) == 1
        assert detections[0]['confidence'] == pytest.approx(0.9)

    def test_detect_uses_confidence_override(self, detector):
        detector.model.return_value = []
        img = _make_bgr(300, 400)
        detector.detect(img, confidence=0.7)
        # Verificar que model foi chamado com conf=0.7
        args, kwargs = detector.model.call_args
        assert kwargs.get('conf') == 0.7

    def test_detect_handles_yolo_exception(self, detector):
        detector.model.side_effect = RuntimeError("GPU OOM")
        img = _make_bgr(300, 400)
        result = detector.detect(img)
        # Deve capturar e retornar lista vazia, não propagar
        assert result == []


# ============================================================================
# SAHI — sliced inference
# ============================================================================

class TestSAHI:
    def test_small_image_no_sahi(self, mock_yolo):
        """Imagem menor que slice_size → SAHI retorna vazio (short-circuit)."""
        d = PlateDetector(
            model_path=mock_yolo['path'], device='cpu',
            enable_sahi=True, sahi_slice_size=640,
        )
        img = _make_bgr(200, 300)  # menor que slice_size
        result = d._detect_with_sahi(img, conf=0.25)
        assert result == []

    def test_large_image_slices_processed(self, mock_yolo):
        """Imagem grande → SAHI processa múltiplos slices."""
        d = PlateDetector(
            model_path=mock_yolo['path'], device='cpu',
            enable_sahi=True, sahi_slice_size=320, sahi_overlap_ratio=0.0,
        )
        # Mock de retorno vazio para todas as chamadas
        d.model.return_value = []

        img = _make_bgr(700, 700)
        result = d._detect_with_sahi(img, conf=0.25)
        # Modelo deve ter sido chamado múltiplas vezes (slices)
        assert d.model.call_count >= 2
        assert result == []

    def test_sahi_remaps_bbox_coords(self, mock_yolo):
        """Detecções feitas em slices devem ter coords remapeadas para imagem original."""
        d = PlateDetector(
            model_path=mock_yolo['path'], device='cpu',
            enable_sahi=True, sahi_slice_size=320, sahi_overlap_ratio=0.0,
        )

        # Simula que todo slice retorna 1 detecção em (10, 10, 50, 40)
        box = _make_yolo_box(10, 10, 50, 40, conf=0.9)
        d.model.return_value = [_make_yolo_result([box])]

        img = _make_bgr(700, 700)
        result = d._detect_with_sahi(img, conf=0.25)
        # Deve haver pelo menos 1 detecção com bbox remapeado (não mais 10,10,50,40)
        assert len(result) >= 1

    def test_detect_retries_sahi_for_low_confidence_large_image(self, mock_yolo):
        d = PlateDetector(
            model_path=mock_yolo['path'], device='cpu',
            enable_sahi=True, sahi_slice_size=320, sahi_overlap_ratio=0.0,
        )
        d.sahi_retry_large_image_threshold = 1000
        d.sahi_retry_confidence_threshold = 0.80
        d._detect_standard = MagicMock(return_value=[
            {'bbox': (10, 20, 100, 80), 'confidence': 0.40, 'class_id': 0, 'class_name': 'plate'}
        ])
        d._detect_with_sahi = MagicMock(return_value=[
            {'bbox': (300, 200, 420, 280), 'confidence': 0.91, 'class_id': 0, 'class_name': 'plate', 'source': 'sahi'}
        ])

        detections = d.detect(_make_bgr(1200, 2200))

        assert d._detect_with_sahi.called
        assert len(detections) == 2
        assert any(det.get('source') == 'sahi' for det in detections)
