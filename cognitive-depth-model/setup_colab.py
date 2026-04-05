"""
============================================
Script de Configuración para Google Colab
Modelo Cognitivo Artificial - Tesis Doctoral
Jesús Goenaga Peña
============================================

Este script se ejecuta al inicio de cada sesión de Colab
para configurar el entorno de trabajo completo.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Instala todas las dependencias del proyecto."""
    print("=" * 60)
    print("INSTALANDO DEPENDENCIAS DEL PROYECTO")
    print("=" * 60)

    packages = [
        "torch",
        "torchvision",
        "torchaudio",
        "opencv-python",
        "scikit-image",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "grad-cam",
        "captum",
        "tensorboard",
        "tqdm",
        "pyyaml",
        "tabulate",
        "colorama",
    ]

    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg]
        )

    print("\nTodas las dependencias instaladas correctamente.")


def verify_environment():
    """Verifica el entorno y muestra información del sistema."""
    import torch
    import torchvision
    import numpy as np
    import platform

    print("\n" + "=" * 60)
    print("VERIFICACIÓN DEL ENTORNO")
    print("=" * 60)

    # Sistema
    print(f"\nSistema Operativo: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")

    # PyTorch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"NumPy: {np.__version__}")

    # GPU
    print(f"\nCUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Memoria GPU: {gpu_mem:.1f} GB")
        print(f"CUDA versión: {torch.version.cuda}")
    else:
        print("ADVERTENCIA: No se detectó GPU. El entrenamiento será lento.")
        print("Ve a Runtime > Change runtime type > GPU")

    # Verificar librerías clave
    print("\n" + "-" * 40)
    print("Librerías verificadas:")
    libs = {
        "scipy": "scipy",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "yaml": "pyyaml",
        "pytorch_grad_cam": "grad-cam",
        "captum": "captum",
    }

    for import_name, display_name in libs.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "OK")
            print(f"  {display_name}: {version}")
        except ImportError:
            print(f"  {display_name}: NO INSTALADO")

    print("\n" + "=" * 60)
    print("ENTORNO LISTO PARA TRABAJAR")
    print("=" * 60)


def setup_google_drive(project_name="cognitive-depth-model"):
    """Monta Google Drive y configura el directorio del proyecto."""
    from google.colab import drive

    print("\nMontando Google Drive...")
    drive.mount("/content/drive")

    project_path = f"/content/drive/MyDrive/{project_name}"
    os.makedirs(project_path, exist_ok=True)

    # Crear subdirectorios en Drive para persistencia
    subdirs = [
        "checkpoints",
        "results/metrics",
        "results/visualizations",
        "results/grad_cam",
        "data/processed",
        "logs",
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(project_path, subdir), exist_ok=True)

    print(f"Directorio del proyecto: {project_path}")
    print("Subdirectorios creados para persistencia de datos.")

    return project_path


def set_reproducibility(seed=42):
    """Configura semillas para reproducibilidad completa."""
    import torch
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"\nSemillas de reproducibilidad configuradas (seed={seed})")


def generate_environment_report():
    """
    Genera un reporte del entorno para incluir en la tesis.
    Guarda un archivo de texto con todas las versiones.
    """
    import torch
    import torchvision
    import numpy as np
    import scipy
    import cv2
    import sklearn
    import matplotlib
    import pandas as pd
    import platform
    from datetime import datetime

    report = []
    report.append("=" * 60)
    report.append("REPORTE DE ENTORNO COMPUTACIONAL")
    report.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append(f"Sistema Operativo: {platform.system()} {platform.release()}")
    report.append(f"Python: {platform.python_version()}")
    report.append(f"PyTorch: {torch.__version__}")
    report.append(f"TorchVision: {torchvision.__version__}")
    report.append(f"NumPy: {np.__version__}")
    report.append(f"SciPy: {scipy.__version__}")
    report.append(f"OpenCV: {cv2.__version__}")
    report.append(f"scikit-learn: {sklearn.__version__}")
    report.append(f"Matplotlib: {matplotlib.__version__}")
    report.append(f"Pandas: {pd.__version__}")

    if torch.cuda.is_available():
        report.append(f"CUDA: {torch.version.cuda}")
        report.append(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        report.append(f"Memoria GPU: {gpu_mem:.1f} GB")

    report_text = "\n".join(report)
    print(report_text)

    return report_text


# --- Ejecución principal ---
if __name__ == "__main__":
    install_dependencies()
    verify_environment()
    set_reproducibility(seed=42)
    print("\n¡Configuración completada exitosamente!")
