"""Bootstrap compartilhado pelos scripts de linha de comando.

Garante que a raiz do projeto está no ``sys.path`` para que ``import src...``
funcione ao executar ``python scripts/<nome>.py`` de qualquer diretório.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
