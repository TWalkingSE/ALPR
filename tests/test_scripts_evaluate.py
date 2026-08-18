# tests/test_scripts_evaluate.py
"""
Testes dos entrypoints de linha de comando em `scripts/`.

Os scripts constroem o pipeline real (YOLO + PaddleOCR), então aqui exercitamos
a lógica que os envolve — resolução de caminhos, dispatch por tipo de mídia,
formatação de saída, códigos de saída e edição do config.yaml — com dublês no
lugar do pipeline.
"""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import alpr_cli  # noqa: E402
import calibrate  # noqa: E402
import evaluate  # noqa: E402
from _fixtures import resolve_fixture_path, run_fixtures  # noqa: E402

from src.v2.config import AppConfig  # noqa: E402
from src.v2.evaluation import FixtureEntry  # noqa: E402
from src.v2.models import LocalPlateResult  # noqa: E402


def _result(plate='ABC1D23', confidence=0.9):
    return LocalPlateResult(
        plate_text=plate,
        confidence=confidence,
        detection_confidence=0.8,
        format_type='mercosul',
        is_valid=True,
        original_crop=np.zeros((4, 8, 3), dtype=np.uint8),
        bbox=(0, 0, 8, 4),
        normalized_text=plate,
        ocr_engine='paddle_ocr',
        quality_score=0.7,
        processing_time_ms=12.5,
    )


# ============================================================================
# _fixtures — resolução de caminho e execução
# ============================================================================

class TestResolveFixturePath:
    def test_absolute_path_is_returned_as_is(self, tmp_path):
        fixture = FixtureEntry(fixture_id='f', path=str(tmp_path / 'a.jpg'))
        assert resolve_fixture_path(fixture, tmp_path / 'm.json', tmp_path) == tmp_path / 'a.jpg'

    def test_resolves_relative_to_manifest_dir(self, tmp_path):
        (tmp_path / 'images').mkdir()
        media = tmp_path / 'images' / 'a.jpg'
        media.write_bytes(b'x')

        fixture = FixtureEntry(fixture_id='f', path='images/a.jpg')
        resolved = resolve_fixture_path(fixture, tmp_path / 'manifest.json', Path('outro'))

        assert resolved == media

    def test_falls_back_to_fixtures_dir(self, tmp_path):
        fixtures_dir = tmp_path / 'fixtures'
        (fixtures_dir / 'images').mkdir(parents=True)
        media = fixtures_dir / 'images' / 'a.jpg'
        media.write_bytes(b'x')

        manifest_dir = tmp_path / 'outro'
        manifest_dir.mkdir()
        fixture = FixtureEntry(fixture_id='f', path='images/a.jpg')

        assert resolve_fixture_path(fixture, manifest_dir / 'm.json', fixtures_dir) == media


class TestRunFixtures:
    def test_missing_media_yields_empty_prediction(self, tmp_path):
        fixtures = [FixtureEntry(fixture_id='sumiu', path='nao-existe.jpg', expected_plate='ABC1D23')]

        records = run_fixtures(MagicMock(), AppConfig(), fixtures, tmp_path / 'm.json')

        assert len(records) == 1
        assert records[0].predicted_plate == ''
        assert records[0].exact_match is False

    def test_pipeline_exception_does_not_abort_the_run(self, tmp_path, monkeypatch):
        media = tmp_path / 'a.jpg'
        media.write_bytes(b'x')
        fixtures = [
            FixtureEntry(fixture_id='quebra', path='a.jpg', expected_plate='ABC1D23'),
            FixtureEntry(fixture_id='ok', path='a.jpg', expected_plate='ABC1D23'),
        ]

        chamadas = {'n': 0}

        def _falha_na_primeira(pipeline, config, path):
            chamadas['n'] += 1
            if chamadas['n'] == 1:
                raise RuntimeError('boom')
            return _result()

        monkeypatch.setattr('_fixtures.run_image_fixture', _falha_na_primeira)

        records = run_fixtures(MagicMock(), AppConfig(), fixtures, tmp_path / 'm.json')

        assert len(records) == 2, 'a execução deve seguir após uma fixture quebrada'
        assert records[0].exact_match is False
        assert records[1].exact_match is True

    def test_dispatches_video_fixtures_to_video_path(self, tmp_path, monkeypatch):
        media = tmp_path / 'v.mp4'
        media.write_bytes(b'x')
        chamou = {'video': False}

        def _video(pipeline, config, path):
            chamou['video'] = True
            return _result()

        monkeypatch.setattr('_fixtures.run_video_fixture', _video)
        fixtures = [FixtureEntry(fixture_id='v', path='v.mp4', media_type='video')]

        run_fixtures(MagicMock(), AppConfig(), fixtures, tmp_path / 'm.json')

        assert chamou['video'] is True


# ============================================================================
# evaluate.py
# ============================================================================

class TestEvaluateCli:
    def test_missing_manifest_exits_with_code_2(self, tmp_path, capsys):
        assert evaluate.main(['--manifest', str(tmp_path / 'nao-existe.json')]) == 2
        assert 'manifesto não encontrado' in capsys.readouterr().err

    def test_empty_manifest_exits_with_code_2(self, tmp_path, capsys):
        manifest = tmp_path / 'm.json'
        manifest.write_text(json.dumps({'version': 1, 'fixtures': []}), encoding='utf-8')

        assert evaluate.main(['--manifest', str(manifest)]) == 2
        assert 'nenhum fixture' in capsys.readouterr().err

    def test_format_delta_reports_percentage_points(self):
        assert '+10.0pp' in evaluate._format_delta(0.9, 0.8)
        assert '-10.0pp' in evaluate._format_delta(0.8, 0.9)
        assert evaluate._format_delta(0.9, None) == ''

    def test_format_delta_absolute_for_latency(self):
        assert '+50.0' in evaluate._format_delta(150.0, 100.0, as_percent=False)
        assert 'pp' not in evaluate._format_delta(150.0, 100.0, as_percent=False)


# ============================================================================
# calibrate.py — edição do config.yaml preservando comentários
# ============================================================================

class TestReplaceScalar:
    SAMPLE = (
        '# comentario de topo\n'
        'models:\n'
        '  detector:\n'
        '    confidence: 0.25  # limiar base\n'
        '\n'
        'pipeline:\n'
        '  # explica o threshold\n'
        '  ocr_confidence_threshold: 0.6\n'
        '  fallback_confidence_threshold: 0.8\n'
    )

    def test_replaces_value_and_keeps_comments(self):
        updated, replaced = calibrate.replace_scalar(
            self.SAMPLE, 'ocr_confidence_threshold', 0.55, section='pipeline'
        )

        assert replaced is True
        assert 'ocr_confidence_threshold: 0.55' in updated
        assert '# comentario de topo' in updated
        assert '# explica o threshold' in updated

    def test_preserves_inline_comment(self):
        updated, _ = calibrate.replace_scalar(self.SAMPLE, 'confidence', 0.3, section='models')
        assert 'confidence: 0.3  # limiar base' in updated

    def test_section_scoping_avoids_wrong_key(self):
        text = 'a:\n  valor: 1\nb:\n  valor: 2\n'
        updated, _ = calibrate.replace_scalar(text, 'valor', 9, section='b')
        assert yaml.safe_load(updated) == {'a': {'valor': 1}, 'b': {'valor': 9}}

    def test_missing_key_reports_not_replaced(self):
        updated, replaced = calibrate.replace_scalar(self.SAMPLE, 'inexistente', 1, section='pipeline')
        assert replaced is False
        assert updated == self.SAMPLE

    def test_result_is_still_valid_yaml(self):
        updated, _ = calibrate.replace_scalar(
            self.SAMPLE, 'fallback_confidence_threshold', 0.75, section='pipeline'
        )
        parsed = yaml.safe_load(updated)
        assert parsed['pipeline']['fallback_confidence_threshold'] == 0.75
        assert parsed['models']['detector']['confidence'] == 0.25


class TestApplyToConfigYaml:
    def test_writes_values_and_backup(self, tmp_path):
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(TestReplaceScalar.SAMPLE, encoding='utf-8')
        candidate = calibrate.ThresholdCandidate(
            detector_confidence=0.30,
            ocr_confidence_threshold=0.55,
            fallback_confidence_threshold=0.75,
        )

        calibrate.apply_to_config_yaml(candidate, config_path)

        parsed = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert parsed['models']['detector']['confidence'] == 0.30
        assert parsed['pipeline']['ocr_confidence_threshold'] == 0.55
        assert parsed['pipeline']['fallback_confidence_threshold'] == 0.75

        backup = config_path.with_suffix('.yaml.bak')
        assert backup.exists()
        assert backup.read_text(encoding='utf-8') == TestReplaceScalar.SAMPLE

    def test_comments_survive_the_round_trip(self, tmp_path):
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(TestReplaceScalar.SAMPLE, encoding='utf-8')

        calibrate.apply_to_config_yaml(
            calibrate.ThresholdCandidate(0.30, 0.55, 0.75), config_path
        )

        conteudo = config_path.read_text(encoding='utf-8')
        assert '# comentario de topo' in conteudo
        assert '# explica o threshold' in conteudo
        assert '# limiar base' in conteudo

    def test_real_config_yaml_keeps_its_comments(self, tmp_path):
        """Exercita o arquivo real do projeto, não só o exemplo sintético."""
        real = PROJECT_ROOT / 'config.yaml'
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(real.read_text(encoding='utf-8'), encoding='utf-8')
        comentarios_antes = real.read_text(encoding='utf-8').count('#')

        calibrate.apply_to_config_yaml(
            calibrate.ThresholdCandidate(0.30, 0.55, 0.75), config_path
        )

        depois = config_path.read_text(encoding='utf-8')
        assert depois.count('#') == comentarios_antes
        parsed = yaml.safe_load(depois)
        assert parsed['pipeline']['ocr_confidence_threshold'] == 0.55
        assert parsed['models']['detector']['confidence'] == 0.30


# ============================================================================
# alpr_cli.py
# ============================================================================

class TestIterInputFiles:
    def test_single_file(self, tmp_path):
        media = tmp_path / 'a.jpg'
        media.write_bytes(b'x')
        assert list(alpr_cli.iter_input_files(media, recursive=False)) == [media]

    def test_directory_filters_unsupported_extensions(self, tmp_path):
        (tmp_path / 'a.jpg').write_bytes(b'x')
        (tmp_path / 'b.mp4').write_bytes(b'x')
        (tmp_path / 'leiame.txt').write_bytes(b'x')

        nomes = {p.name for p in alpr_cli.iter_input_files(tmp_path, recursive=False)}

        assert nomes == {'a.jpg', 'b.mp4'}

    def test_recursive_descends_into_subdirectories(self, tmp_path):
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub' / 'a.jpg').write_bytes(b'x')

        assert list(alpr_cli.iter_input_files(tmp_path, recursive=False)) == []
        assert len(list(alpr_cli.iter_input_files(tmp_path, recursive=True))) == 1


class TestWriteOutput:
    ROWS = [
        {
            'arquivo': 'a.jpg', 'tipo': 'imagem', 'placa': 'ABC1D23', 'formato': 'mercosul',
            'valida': True, 'confianca_ocr': 0.9, 'confianca_deteccao': 0.8, 'qualidade': 0.7,
            'cenarios': 'low_light', 'avisos': '', 'tempo_ms': 12.5, 'laudo': 'r.json',
        }
    ]

    def test_csv_has_header_and_row(self, tmp_path):
        out = tmp_path / 'saida.csv'
        alpr_cli.write_output(self.ROWS, 'csv', out)

        linhas = list(csv.DictReader(out.open(encoding='utf-8')))
        assert len(linhas) == 1
        assert linhas[0]['placa'] == 'ABC1D23'

    def test_json_is_one_object_per_line(self, tmp_path):
        out = tmp_path / 'saida.jsonl'
        alpr_cli.write_output(self.ROWS * 2, 'json', out)

        linhas = out.read_text(encoding='utf-8').strip().splitlines()
        assert len(linhas) == 2
        assert json.loads(linhas[0])['placa'] == 'ABC1D23'

    def test_text_marks_validity(self, tmp_path):
        out = tmp_path / 'saida.txt'
        alpr_cli.write_output(self.ROWS, 'text', out)
        assert out.read_text(encoding='utf-8').startswith('OK')

        invalida = [{**self.ROWS[0], 'valida': False}]
        alpr_cli.write_output(invalida, 'text', out)
        assert out.read_text(encoding='utf-8').startswith('REV')


class TestAlprCliMain:
    def test_missing_input_exits_with_code_2(self, tmp_path, capsys):
        assert alpr_cli.main([str(tmp_path / 'nao-existe')]) == 2
        assert 'entrada não encontrada' in capsys.readouterr().err

    def test_directory_without_media_exits_with_code_2(self, tmp_path, capsys):
        (tmp_path / 'leiame.txt').write_bytes(b'x')
        assert alpr_cli.main([str(tmp_path)]) == 2
        assert 'nenhum arquivo de mídia' in capsys.readouterr().err

    @pytest.mark.parametrize(
        ('nome', 'esperado'),
        [('r.csv', 'csv'), ('r.json', 'json'), ('r.jsonl', 'json'), ('r.txt', 'text')],
    )
    def test_output_format_inferred_from_extension(self, tmp_path, monkeypatch, nome, esperado):
        media = tmp_path / 'a.jpg'
        media.write_bytes(b'x')
        capturado = {}

        monkeypatch.setattr(
            alpr_cli.LocalAnalysisPipeline, 'from_settings',
            classmethod(lambda cls, *a, **k: SimpleNamespace()),
        )
        monkeypatch.setattr(
            alpr_cli, 'process_image_file', lambda *a, **k: list(TestWriteOutput.ROWS)
        )
        monkeypatch.setattr(
            alpr_cli, 'write_output',
            lambda rows, fmt, out: capturado.update(formato=fmt),
        )

        alpr_cli.main([str(media), '--out', str(tmp_path / nome)])

        assert capturado['formato'] == esperado

    def test_exit_code_1_when_nothing_valid_was_read(self, tmp_path, monkeypatch):
        media = tmp_path / 'a.jpg'
        media.write_bytes(b'x')

        monkeypatch.setattr(
            alpr_cli.LocalAnalysisPipeline, 'from_settings',
            classmethod(lambda cls, *a, **k: SimpleNamespace()),
        )
        monkeypatch.setattr(
            alpr_cli, 'process_image_file',
            lambda *a, **k: [{**TestWriteOutput.ROWS[0], 'valida': False}],
        )

        assert alpr_cli.main([str(media), '--out', str(tmp_path / 'r.csv')]) == 1

    def test_exit_code_0_when_something_valid_was_read(self, tmp_path, monkeypatch):
        media = tmp_path / 'a.jpg'
        media.write_bytes(b'x')

        monkeypatch.setattr(
            alpr_cli.LocalAnalysisPipeline, 'from_settings',
            classmethod(lambda cls, *a, **k: SimpleNamespace()),
        )
        monkeypatch.setattr(
            alpr_cli, 'process_image_file', lambda *a, **k: list(TestWriteOutput.ROWS)
        )

        assert alpr_cli.main([str(media), '--out', str(tmp_path / 'r.csv')]) == 0
