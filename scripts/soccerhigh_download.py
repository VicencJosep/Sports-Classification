"""Descarga las features VideoMAEv2 de SoccerHigh (Hugging Face, CC BY-NC-SA 4.0) a ``datasets/soccerhigh``.

Las anotaciones (``Labels-summary.json``, ``*_intervals.srt``, listas de partición) vienen del repo de GitHub
IPCV/SoccerHigh y ya están en ``datasets/soccerhigh``; aquí solo se bajan los ``.npy`` por partido.

    docker exec -w /app Sports-Classification python scripts/soccerhigh_download.py --split test --backbone giant
    docker exec -w /app Sports-Classification python scripts/soccerhigh_download.py --games "spain_laliga/2016-2017/2017-04-23 - 21-45 Real Madrid 2 - 3 Barcelona"
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path("datasets/soccerhigh")
REPO = "imva-upf/SoccerHigh"
FILES = {
    "giant": "HQ_224p_VideoMAEv2_Giant_K710_1408.npy",
    "small": "HQ_224p_VideoMAEv2_SmallFromGiant_K710_384.npy",
}


def read_split(name: str) -> list[str]:
    return [l.strip() for l in (ROOT / f"{name}.txt").read_text().splitlines() if l.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "validation", "test"], action="append", default=[])
    p.add_argument("--games", nargs="*", default=[], help="rutas liga/temporada/partido (como en test.txt)")
    p.add_argument("--backbone", choices=list(FILES), default="giant")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    import time

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    games = list(args.games)
    for s in args.split:
        games += read_split(s)
    games = games[: args.limit] if args.limit else games
    pending = [g for g in games if not all((ROOT / g / f"{h}_{FILES[args.backbone]}").exists() for h in (1, 2))]
    print(f"{len(games)} partidos, {len(pending)} sin features '{args.backbone}'")
    if not pending:
        return
    # Fichero a fichero (listar el árbol entero del repo agota el límite de peticiones de la API de HF).
    for i, g in enumerate(pending, 1):
        for h in (1, 2):
            rel = f"{g}/{h}_{FILES[args.backbone]}"
            if (ROOT / rel).exists():
                continue
            for attempt in range(6):
                try:
                    hf_hub_download(REPO, rel, repo_type="dataset", local_dir=str(ROOT))
                    break
                except HfHubHTTPError as exc:
                    wait = 60 * (attempt + 1)
                    print(f"  {rel}: {exc}; reintento en {wait} s", flush=True)
                    time.sleep(wait)
        print(f"[{i}/{len(pending)}] {g}", flush=True)
    done = sum(all((ROOT / g / f"{h}_{FILES[args.backbone]}").exists() for h in (1, 2)) for g in games)
    print(f"listo: {done}/{len(games)} partidos con features '{args.backbone}' en {ROOT}")


if __name__ == "__main__":
    main()
