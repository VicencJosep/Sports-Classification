"""Descarga los datasets originales de Roboflow Universe usados por el pipeline
de baloncesto a basketball/datasets/. Requiere ROBOFLOW_API_KEY en el entorno
(o en basketball/.env).

Uso:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --format coco
"""
import argparse
import os
from pathlib import Path

from roboflow import Roboflow

BASKETBALL_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASKETBALL_DIR / ".env"

DATASETS = [
    {
        "workspace": "roboflow-jvuqo",
        "project": "basketball-player-detection-3-ycjdo",
        "version": 4,
        "default_format": "yolov8",
    },
    {
        "workspace": "roboflow-jvuqo",
        "project": "basketball-court-detection-2",
        "version": 14,
        "default_format": "coco",
    },
]


def _load_dotenv() :
    if "ROBOFLOW_API_KEY" in os.environ or not ENV_FILE.exists() :
        return
    for line in ENV_FILE.read_text().splitlines() :
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line :
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() :
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        default=None,
        help="Formato de export de Roboflow (por defecto: yolov8 para detección, "
        "coco para keypoints de cancha). Sobrescribe ambos si se indica.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASKETBALL_DIR / "datasets"),
        help="Carpeta destino (default: basketball/datasets/)",
    )
    args = parser.parse_args()

    _load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key :
        raise SystemExit(
            "ROBOFLOW_API_KEY no está definida. Expórtala o copia .env.example a .env "
            "y rellena tu clave (https://app.roboflow.com/settings/api)."
        )

    rf = Roboflow(api_key=api_key)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS :
        fmt = args.format or ds["default_format"]
        dest = output_root / f"{ds['project']}-v{ds['version']}"
        print(f"Descargando {ds['workspace']}/{ds['project']}/{ds['version']} "
              f"(formato={fmt}) -> {dest}")
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        version = project.version(ds["version"])
        version.download(fmt, location=str(dest))
        print(f"  OK: {dest}")


if __name__ == "__main__" :
    main()
