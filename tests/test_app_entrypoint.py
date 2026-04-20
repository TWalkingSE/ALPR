from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import app


class _TabContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_app_main_renders_release_entrypoint_without_uploads(monkeypatch):
    config = SimpleNamespace(signature=lambda: ('sig',))
    pipeline = SimpleNamespace(name='pipeline')
    premium_service = SimpleNamespace(available=True)

    render_sidebar = MagicMock(return_value=(config, 'model.pt', False))
    ensure_services = MagicMock(return_value=(pipeline, premium_service))
    display_pipeline = MagicMock()
    render_image = MagicMock()
    render_video = MagicMock()
    file_uploader = MagicMock(side_effect=[None, None])

    monkeypatch.setattr(app, 'load_config', lambda: {'pipeline': {}})
    monkeypatch.setattr(app, 'build_v2_config', lambda raw_config: config)
    monkeypatch.setattr(app, 'render_sidebar', render_sidebar)
    monkeypatch.setattr(app, '_ensure_services', ensure_services)
    monkeypatch.setattr(app, 'display_pipeline_info', display_pipeline)
    monkeypatch.setattr(app, '_render_image_outputs', render_image)
    monkeypatch.setattr(app, '_render_video_outputs', render_video)
    monkeypatch.setattr(app.st, 'title', lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, 'caption', lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, 'tabs', lambda labels: (_TabContext(), _TabContext()))
    monkeypatch.setattr(app.st, 'file_uploader', file_uploader)

    app.main()

    render_sidebar.assert_called_once_with(config, app.PROJECT_DIR)
    ensure_services.assert_called_once_with(config, 'model.pt', force_rebuild=False)
    display_pipeline.assert_called_once_with(pipeline)
    render_image.assert_called_once_with()
    render_video.assert_called_once_with(config)
    assert file_uploader.call_count == 2


def test_app_main_shows_error_when_service_init_fails(monkeypatch):
    config = SimpleNamespace(signature=lambda: ('sig',))
    render_sidebar = MagicMock(return_value=(config, 'model.pt', False))
    show_error = MagicMock()
    log_error = MagicMock()

    monkeypatch.setattr(app, 'load_config', lambda: {'pipeline': {}})
    monkeypatch.setattr(app, 'build_v2_config', lambda raw_config: config)
    monkeypatch.setattr(app, 'render_sidebar', render_sidebar)
    monkeypatch.setattr(app, '_ensure_services', MagicMock(side_effect=RuntimeError('boom')))
    monkeypatch.setattr(app.st, 'title', lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, 'caption', lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, 'error', show_error)
    monkeypatch.setattr(app.logger, 'error', log_error)

    app.main()

    show_error.assert_called_once()
    assert 'Falha ao inicializar a aplicacao' in show_error.call_args.args[0]
    log_error.assert_called_once()