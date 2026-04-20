"""
Script de instalação automatizada do ALPR.

Fluxo:
 1. Checa Python >= 3.11
 2. Detecta CUDA (opcional, apenas informativo)
 3. Atualiza pip
 4. Instala PyTorch com CUDA 12.8 se GPU detectada, CPU caso contrário
 5. Instala o projeto via `pip install -e .` (fonte de verdade: pyproject.toml)
 6. Cria diretórios de dados
 7. Verifica .env

Para GPU NVIDIA (CUDA 12.x):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

"""

import os
import sys
import platform
import subprocess


def run(command: str, description: str) -> bool:
    """Executa comando e imprime status."""
    print(f"\n{'='*60}\n🔧 {description}\n{'='*60}")
    try:
        subprocess.run(command, shell=True, check=True)
        print("✅ Sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro: {e}")
        return False


def check_cuda() -> bool:
    """Detecta GPU NVIDIA via nvidia-smi."""
    try:
        result = subprocess.run(
            "nvidia-smi",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            print("✅ NVIDIA GPU detectada")
            return True
    except Exception:
        pass
    print("ℹ️  GPU NVIDIA não detectada — instalando PyTorch CPU")
    return False


def main() -> int:
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║          ALPR - Instalacao de Dependencias                 ║
    ║     ALPR 2.0 com PaddleOCR + YOLOv11 + Streamlit          ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    py = sys.version_info
    print(f"🐍 Python: {py.major}.{py.minor}.{py.micro}")
    if py < (3, 11):
        print("❌ Python 3.11+ é obrigatório. Abortando.")
        return 1

    print(f"💻 Sistema: {platform.system()} {platform.release()}")
    has_cuda = check_cuda()

    # 1) Atualizar pip
    if not run(f"{sys.executable} -m pip install --upgrade pip", "Atualizando pip"):
        print("⚠️  Falha no upgrade do pip — prosseguindo mesmo assim")

    # 2) PyTorch (GPU primeiro; pip install -e . pegaria versão CPU)
    if has_cuda:
        torch_cmd = (
            f"{sys.executable} -m pip install --upgrade "
            f"torch torchvision torchaudio "
            f"--index-url https://download.pytorch.org/whl/cu128"
        )
        if not run(torch_cmd, "Instalando PyTorch com CUDA 12.8"):
            print("⚠️  Falha ao instalar PyTorch CUDA. Seguirá com CPU do pyproject.")

    # 3) Projeto (pyproject.toml como fonte de verdade)
    if not run(
        f"{sys.executable} -m pip install -e .",
        "Instalando dependências do projeto (pyproject.toml)",
    ):
        print("❌ Falha ao instalar o projeto. Abortando.")
        return 1

    # 4) Diretórios
    print("\n📁 Criando estrutura de diretórios...")
    for d in ("models/yolo", "data/results"):
        os.makedirs(d, exist_ok=True)
        print(f"   ✓ {d}")

    # 5) .env
    print("\n🔐 Verificando configuração...")
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("⚠️  Arquivo .env não encontrado")
            print("    cp .env.example .env    # Edite com suas chaves")
        else:
            print("❌ .env.example não encontrado")
    else:
        print("✅ Arquivo .env presente")

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                  Instalação Concluída!                     ║
    ╚════════════════════════════════════════════════════════════╝

    📋 Próximos passos:

        1. Baixe um modelo YOLOv11 para placas em models/yolo/
       (ex: yolo11l-plate.pt — ver README.md)

        2. Configure .env com PLATE_RECOGNIZER_API_KEY, se quiser usar o fluxo Premium

        3. Execute a aplicacao:
            streamlit run app.py

        4. Rode os testes:
         python -m pytest tests/ -v
    """)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Instalação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
