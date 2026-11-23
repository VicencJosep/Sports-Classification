"""
Convierte los datasets de posesión a la caché ``.npz`` de ``core/team_possession``.

    python scripts/benchmark/cache_possession_data.py --dataset idsse
    python scripts/benchmark/cache_possession_data.py --dataset metrica

IDSSE: ``datasets/posesion/idsse/DFL_04_03_positions_raw_observed_*.xml`` (350-420 MB
cada uno, ~1-2 min por partido); Metrica: ``datasets/posesion/metrica_sample``.
La verdad de Metrica se deriva de los eventos (aproximada); la de IDSSE es la del
proveedor por fotograma. Salida: ``output/posesion/cache/<dataset>_<partido>.npz``.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.team_possession import io_metrica, io_sportec  # noqa: E402
from core.team_possession.labels import describe  # noqa: E402

CACHE = ROOT / "output" / "posesion" / "cache"


def cache_idsse(src: Path, force: bool) -> None:
    for pos in sorted(src.glob("DFL_04_03_positions_raw_observed_*.xml")):
        mid = re.search(r"(DFL-MAT-[A-Z0-9]+)", pos.name).group(1)
        info = next(src.glob(f"DFL_02_01_matchinformation_*_{mid}.xml"))
        out = CACHE / f"idsse_{mid[-6:]}.npz"
        if out.exists() and not force:
            print("ya existe", out.name); continue
        t0 = time.time()
        data = io_sportec.load_idsse(pos, info, name=f"idsse_{mid[-6:]}")
        io_sportec.save_npz(data, out)
        print(f"{out.name}: {data.summary()} | {describe(data.labels, data.fps)} | {time.time() - t0:.0f} s",
              flush=True)


def cache_metrica(src: Path, force: bool) -> None:
    d = src / "data"
    for g in (1, 2):
        out = CACHE / f"metrica_game{g}.npz"
        if out.exists() and not force:
            print("ya existe", out.name); continue
        gd = d / f"Sample_Game_{g}"
        data = io_metrica.load_sample_csv(gd / f"Sample_Game_{g}_RawTrackingData_Home_Team.csv",
                                          gd / f"Sample_Game_{g}_RawTrackingData_Away_Team.csv",
                                          name=f"metrica_game{g}")
        events = io_metrica.read_events_csv(gd / f"Sample_Game_{g}_RawEventsData.csv")
        data.labels = io_metrica.labels_from_events(events, data.frame_index, data.period)
        io_sportec.save_npz(data, out)
        print(f"{out.name}: {data.summary()} | {describe(data.labels, data.fps)}", flush=True)
    out = CACHE / "metrica_game3.npz"
    if not out.exists() or force:
        gd = d / "Sample_Game_3"
        meta = io_metrica.read_epts_metadata(gd / "Sample_Game_3_metadata.xml")
        events = io_metrica.read_events_json(gd / "Sample_Game_3_events.json", home_team_id=meta["teams"][0][0])
        # periodos: del primer al último evento de cada periodo (el EPTS de ejemplo no los trae)
        periods = []
        for p in sorted({e["period"] for e in events}):
            fr = [e["start"] for e in events if e["period"] == p] + [e["end"] for e in events if e["period"] == p]
            periods.append((min(fr), max(fr), p))
        # extender el periodo 1 hasta el inicio del 2 y el 2 hasta el final para no dejar fotogramas sin periodo
        if len(periods) >= 2:
            periods[0] = (1, periods[1][0] - 1, 1)
            periods[-1] = (periods[-1][0], 10 ** 9, periods[-1][2])
        data = io_metrica.load_epts(gd / "Sample_Game_3_metadata.xml", gd / "Sample_Game_3_tracking.txt",
                                    name="metrica_game3", periods=periods)
        data.labels = io_metrica.labels_from_events(events, data.frame_index, data.period)
        io_sportec.save_npz(data, out)
        print(f"{out.name}: {data.summary()} | {describe(data.labels, data.fps)} | params {list(meta['params'])[:8]}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dataset", choices=("idsse", "metrica"), required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.dataset == "idsse":
        cache_idsse(ROOT / "datasets" / "posesion" / "idsse", args.force)
    else:
        cache_metrica(ROOT / "datasets" / "posesion" / "metrica_sample", args.force)


if __name__ == "__main__":
    main()
