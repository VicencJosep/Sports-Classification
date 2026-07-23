"""Descarga los datos de SoccerNet necesarios para entrenar el clasificador de
Pass/Drive/Tackle/Background (core/ball_actions.py), vía el paquete pip
``SoccerNet`` (`pip install SoccerNet`, ya listado en requirements.txt).

Dos fuentes de datos DISTINTAS (ver README.md, sección "Ball Action Spotting"):

1. **Ball Action Spotting** (``Labels-ball.json`` + vídeo): 9 partidos de la EFL
   Championship inglesa 2019-2020 (4 train / 1 valid / 2 test con ground truth
   pública + 2 challenge sin GT) — dataset AISLADO, no las 6 ligas ya
   descargadas en ``/mnt/datasets/SoccerNet``. Requiere password de la NDA de
   SoccerNet (https://www.soccer-net.org/data) porque viene empaquetado junto
   al vídeo. Task real: ``downloadDataTask(task="spotting-ball-2024", ...)``.
2. **Labels-v2.json / Labels-cameras.json** (públicas, sin NDA, password
   genérica ``"SoccerNet"`` ya hardcodeada en el SDK): pertenecen al corpus
   original de 500 partidos, es decir, a las 6 ligas que ya tenéis en
   ``/mnt/datasets/SoccerNet``. Se usan aquí solo para la clase Background
   (ver ``scripts/extract_ball_action_clips.py``): ``Labels-v2.json`` para
   descartar cercanía a cualquier evento de Action Spotting, ``Labels-cameras.json``
   (solo disponible para 200 de esos 500 partidos) para diversificar el tipo de
   plano muestreado.

Uso (desde la raíz del proyecto; con -m por consistencia con el resto del repo):
    # Solo lo público (sin password NDA) -> Labels-v2 + Labels-cameras
    python -m scripts.download_ball_action_data --tasks labels-v2,cameras

    # Ball Action Spotting (requiere NDA) -> exporta la password antes:
    export SOCCERNET_NDA_PASSWORD="..."
    python -m scripts.download_ball_action_data --tasks ball

    # Todo, sin descargar nada de verdad (valida los parámetros / imprime plan):
    python -m scripts.download_ball_action_data --tasks all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

DEFAULT_LOCAL_DIR = "/mnt/datasets/SoccerNet"
_PUBLIC_PASSWORD = "SoccerNet"  # password pública de Labels/features, hardcodeada en el propio SDK

_ALL_TASKS = ("ball", "labels-v2", "cameras", "player-bboxes")


def _load_dotenv() -> None:
    if "SOCCERNET_NDA_PASSWORD" in os.environ or not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _download_ball_action_spotting(local_dir: str, split: list[str], password: str, dry_run: bool) -> None:
    """9 partidos EFL Championship 2019-2020 (Pass/Drive/Tackle/... + vídeo). Requiere NDA."""
    logger.info(
        "[ball] downloadDataTask(task='spotting-ball-2024', split=%s) -> %s (NDA)",
        split, local_dir,
    )
    if dry_run:
        return
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.downloadDataTask(task="spotting-ball-2024", split=split, password=password)


def _download_labels_v2(local_dir: str, split: list[str], dry_run: bool) -> None:
    """Labels-v2.json (Action Spotting, 17 clases) para los 500 partidos ya descargados. Público."""
    logger.info("[labels-v2] downloadGames(files=['Labels-v2.json'], split=%s) -> %s", split, local_dir)
    if dry_run:
        return
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.password = _PUBLIC_PASSWORD
    downloader.downloadGames(files=["Labels-v2.json"], split=split)


def _download_cameras(local_dir: str, split: list[str], dry_run: bool) -> None:
    """Labels-cameras.json (tipo de plano), solo disponible en 200/500 partidos. Público."""
    logger.info(
        "[cameras] downloadGames(files=['Labels-cameras.json'], split=%s, task='camera-changes') -> %s",
        split, local_dir,
    )
    if dry_run:
        return
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.password = _PUBLIC_PASSWORD
    downloader.downloadGames(files=["Labels-cameras.json"], split=split, task="camera-changes")


def _download_player_bboxes(local_dir: str, split: list[str], dry_run: bool) -> None:
    """Bounding boxes de jugadores precalculadas (MaskRCNN, 2fps), 500 partidos. Público."""
    files = ["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"]
    logger.info("[player-bboxes] downloadGames(files=%s, split=%s) -> %s", files, split, local_dir)
    if dry_run:
        return
    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=local_dir)
    downloader.password = _PUBLIC_PASSWORD
    downloader.downloadGames(files=files, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tasks", default="all",
        help=f"Coma-separado, subconjunto de {_ALL_TASKS} (o 'all')",
    )
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="Raíz local de SoccerNet")
    parser.add_argument(
        "--split", default="train,valid,test",
        help="Splits a descargar (coma-separado). 'ball' también acepta 'challenge'.",
    )
    parser.add_argument(
        "--password", default=None,
        help="Password NDA para --tasks ball (si no, se lee SOCCERNET_NDA_PASSWORD del entorno/.env)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="No descarga nada: solo valida parámetros e imprime qué se llamaría",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tasks = _ALL_TASKS if args.tasks == "all" else tuple(t.strip() for t in args.tasks.split(","))
    unknown = set(tasks) - set(_ALL_TASKS)
    if unknown:
        parser.error(f"Tarea(s) desconocida(s): {unknown} (válidas: {_ALL_TASKS})")
    split = [s.strip() for s in args.split.split(",")]

    if "ball" in tasks:
        _load_dotenv()
        password = args.password or os.environ.get("SOCCERNET_NDA_PASSWORD")
        if not password and not args.dry_run:
            parser.error(
                "--tasks ball requiere password NDA: pásala con --password o define "
                "SOCCERNET_NDA_PASSWORD (env var o .env en la raíz del proyecto). "
                "La NDA se firma en https://www.soccer-net.org/data"
            )
        _download_ball_action_spotting(args.local_dir, split, password or "<dry-run>", args.dry_run)

    if "labels-v2" in tasks:
        _download_labels_v2(args.local_dir, split, args.dry_run)

    if "cameras" in tasks:
        _download_cameras(args.local_dir, split, args.dry_run)

    if "player-bboxes" in tasks:
        _download_player_bboxes(args.local_dir, split, args.dry_run)

    if args.dry_run:
        logger.info("Dry-run: no se ha descargado nada.")


if __name__ == "__main__":
    main()
