"""Benchmark E1 (docs/comparativa_benchmark.md §5.2): mAP@1 OFICIAL de Ball
Action Spotting (12 clases, ±0,5 s) a partir de las cachés de T-DEED
(``720p.tdeed.<modelo>.npz``), con el post-proceso y el evaluador del propio
repo de T-DEED (``models/tdeed/T-DEED/util``), para poder poner nuestras cifras
al lado de las publicadas.

Post-proceso, igual que ``util.eval.evaluate(test=True)``: candidatos con
score ≥ 0,01 en cada frame muestreado y clase, luego NMS (ventana 6 frames
muestreados) o NMS suave (ventana 12). T-DEED guarda la que mejor sale en el
propio test; aquí se informan las dos.

Splits (``models/tdeed/T-DEED/data/soccernetball/*.json``): ``test`` = Stoke +
Reading, igual que nuestro eval de EFL (docs/particion_datasets.md).
``SoccerNetBall_challenge2`` se entrenó con ellos (``train_challenge.json``):
su cifra en ``test`` está contaminada.

Otros partidos: ``--dataset st2`` evalúa SoccerTrack v2 (panorámica, cámara virtual
``views/<m>/<parte>_<vista>.mp4``; mismas 12 clases, equipo left/right) por partes.
``--dataset pcbas`` evalúa el split de validación de SN-PCBAS-2026 (FOOTPASS: 3 partidos completos 1080p de 2023/24,
8 clases anotadas por frame de contacto; etiquetas convertidas desde ``val_tactical_data.h5``). Como FOOTPASS no separa
pase alto de pase, ``--merge-high-pass`` funde HIGH PASS en PASS (máximo) antes del NMS; se informa además la media sobre
las 8 clases de FOOTPASS (``mAP_pcbas8``).
Otras métricas: ``--metrics at1 tight`` (tight = media de mAP con tolerancias 1-5 s).

Uso (contenedor, raíz del proyecto):
    python -m scripts.benchmark.eval_bas --model SoccerNetBall_test --split test --metrics at1 tight
    python -m scripts.benchmark.eval_bas --model SNB_pano1 --dataset st2 --matches 128057 132831
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

from core.ball_spotting import read_scores

ROOT = Path(__file__).resolve().parents[2]
TDEED_DIR = ROOT / "models" / "tdeed" / "T-DEED"
LABELS_ROOT = ROOT / "datasets" / "SoccerNet"
OUT_ROOT = ROOT / "output" / "benchmark" / "bas"
ST2_ROOT = ROOT / "datasets" / "soccertrack-v2"
OUT_ROOT_ST2 = ROOT / "output" / "benchmark" / "bas_st2"
PCBAS_ROOT = ROOT / "datasets" / "SoccerNet" / "pcbas-2026"
OUT_ROOT_PCBAS = ROOT / "output" / "benchmark" / "bas_pcbas"
# Ids de clase de ``*_tactical_data.h5`` (columna 14) -> nombres de SN-BAS. Orden de ``class_names`` en ``evaluation.py`` del
# repo oficial (github.com/JeremieOchin/FOOTPASS): background, drive, pass, cross, throw-in, shot, header, tackle, block; coincide
# con la distribución publicada (pase 49,9 %, conducción 39,0 %, ..., entrada 0,3 %) frente a los recuentos del split de validación.
PCBAS_CLASSES = {1: "DRIVE", 2: "PASS", 3: "CROSS", 4: "THROW IN", 5: "SHOT", 6: "HEADER",
                 7: "PLAYER SUCCESSFUL TACKLE", 8: "BALL PLAYER BLOCK"}
PCBAS_FPS = 25.0


def _tdeed_eval():
    sys.path.insert(0, str(TDEED_DIR))
    from util import eval as tdeed_eval  # noqa: E402  (util/ de T-DEED, no utils/ del proyecto)
    return tdeed_eval


def candidates(probs: np.ndarray, classes: tuple[str, ...], thr: float = 0.01) -> list[dict]:
    """Eventos de alto recall: (frame muestreado, clase) con score ≥ thr; columna 0 = fondo."""
    rows, cols = np.nonzero(probs[:, 1:] >= thr)
    return [{"label": classes[c + 1], "frame": int(i), "score": float(probs[i, c + 1])} for i, c in zip(rows, cols)]


def st2_games(matches: list[str], view: str) -> list[tuple[str, Path]]:
    """SoccerTrack v2: cada parte es un "partido" para el evaluador (solo mira la parte 1).

    Genera ``OUT_ROOT_ST2/labels/<m>/<parte>/Labels-ball.json`` desde ``bas/<m>/<m>_12_class_events.json``
    (posición en ms dentro de la parte, equipo left/right)."""
    games = []
    for m in matches:
        actions = json.loads((ST2_ROOT / "bas" / m / f"{m}_12_class_events.json").read_text())["actions"]
        for h, hh in (("1st", "1"), ("2nd", "2")):
            ann = [{"gameTime": "1 - " + a["gameTime"].split(" - ")[1], "label": a["label"], "position": a["position"],
                    "team": a.get("team", ""), "visibility": "visible"}
                   for a in actions if a["gameTime"].startswith(f"{hh} -")]
            path = OUT_ROOT_ST2 / "labels" / m / h / "Labels-ball.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"UrlLocal": f"{m}/{h}", "annotations": ann}))
            games.append((f"{m}/{h}", ST2_ROOT / "views" / m / f"{h}_{view}"))
    return games


def pcbas_games(split: str = "val") -> list[tuple[str, Path]]:
    """SN-PCBAS-2026 (FOOTPASS): un vídeo 1080p por partido (``videos/game_<n>.mp4``, 25 fps, frames numerados de forma
    continua en el partido) y eventos por frame de contacto en ``labels/<split>_tactical_data.h5`` (columna 0 frame, 13 clase).
    Genera ``OUT_ROOT_PCBAS/labels/<game>/Labels-ball.json`` (todo como parte 1, posición en ms)."""
    import h5py

    games = []
    with h5py.File(PCBAS_ROOT / "labels" / f"{split}_tactical_data.h5") as f:
        by_game: dict[str, list[dict]] = {}
        for key in f:
            game = key.rsplit("_H", 1)[0]
            a = f[key][...]
            ev = a[a[:, 13] != 0]
            for row in ev:
                ms = int(round(row[0] / PCBAS_FPS * 1000))
                by_game.setdefault(game, []).append({"gameTime": f"1 - {ms // 60000}:{(ms % 60000) // 1000:02d}",
                                                     "label": PCBAS_CLASSES[int(row[13])], "position": ms,
                                                     "team": "left" if row[2] == 0 else "right", "visibility": "visible"})
    for game, ann in sorted(by_game.items()):
        path = OUT_ROOT_PCBAS / "labels" / game / "Labels-ball.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"UrlLocal": game, "annotations": sorted(ann, key=lambda x: x["position"])}))
        games.append((game, PCBAS_ROOT / "videos" / game))
    return games


def merge_high_pass(probs: np.ndarray, classes: tuple[str, ...]) -> np.ndarray:
    """FOOTPASS no distingue pase alto: PASS := max(PASS, HIGH PASS) y HIGH PASS := 0."""
    p = probs.astype(np.float32).copy()
    i, j = classes.index("PASS"), classes.index("HIGH PASS")
    p[:, i] = np.maximum(p[:, i], p[:, j])
    p[:, j] = 0
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Nombre de la caché: <vídeo>.tdeed.<model>.npz")
    ap.add_argument("--dataset", default="efl", choices=["efl", "st2", "pcbas"])
    ap.add_argument("--merge-high-pass", action="store_true", help="PCBAS: funde HIGH PASS en PASS antes del NMS")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="EFL")
    ap.add_argument("--matches", nargs="+", default=["128057", "132831"], help="SoccerTrack v2 (eval por defecto)")
    ap.add_argument("--view", default="vcam796vl1", help="SoccerTrack v2: vista de cámara virtual")
    ap.add_argument("--metrics", nargs="+", default=["at1"], help="at1 (oficial BAS), tight (1-5 s), at2..at5")
    ap.add_argument("--nms", action="store_true", help="Evalúa también NMS duro además del suave")
    args = ap.parse_args()

    tdeed_eval = _tdeed_eval()
    if args.dataset == "efl":
        names = [g["video"] for g in json.loads((TDEED_DIR / "data" / "soccernetball" / f"{args.split}.json").read_text())]
        games = [(g, LABELS_ROOT / g / "720p") for g in names]
        label_root, out_root, tag = LABELS_ROOT, OUT_ROOT / args.model / args.split, args.split
    elif args.dataset == "st2":
        games = st2_games(args.matches, args.view)
        label_root, out_root, tag = OUT_ROOT_ST2 / "labels", OUT_ROOT_ST2 / args.model / args.view, args.view
    else:
        games = pcbas_games(args.split if args.split != "test" else "val")
        tag = "val_merged" if args.merge_high_pass else "val"
        label_root, out_root = OUT_ROOT_PCBAS / "labels", OUT_ROOT_PCBAS / args.model / tag
    preds, stride = [], None
    for game, stem in games:
        sc = read_scores(stem.with_name(f"{stem.name}.tdeed.{args.model}.npz"))
        stride = sc.stride
        probs = merge_high_pass(sc.probs, sc.classes) if args.merge_high_pass else sc.probs
        preds.append({"video": game, "events": candidates(probs, sc.classes), "fps": sc.fps / sc.stride})

    # Ventanas de T-DEED en filas a 12,5 fps (stride 2); se escalan al stride de la caché.
    w_nms, w_snms = (max(1, round(w * 2 / stride)) for w in tdeed_eval.WINDOWS_SNB)
    variants = {f"snms{w_snms}": tdeed_eval.soft_non_maximum_supression(preds, window=w_snms, threshold=0.01)}
    if args.nms:
        variants[f"nms{w_nms}"] = tdeed_eval.non_maximum_supression(preds, window=w_nms, threshold=0.01)
    counts = {c: 0 for c in tdeed_eval_classes()}
    for game, _ in games:
        for a in json.loads((label_root / game / "Labels-ball.json").read_text())["annotations"]:
            counts[a["label"]] += 1
    summary = {"model": args.model, "dataset": args.dataset, "games": [g for g, _ in games], "gt_counts": counts,
               "results": {}}
    for name, pred in variants.items():
        out = out_root / name
        # store_json_snb escribe en <dir(pred_path)>/preds/<game>/results_spotting.json
        tdeed_eval.store_json_snb(str(out / "pred"), pred, stride=stride)
        for metric in args.metrics:
            with contextlib.redirect_stderr(io.StringIO()):
                res = tdeed_eval.aux_evaluate(str(label_root), str(out / "preds"), list_games=[g for g, _ in games],
                                              prediction_file="results_spotting.json", version=2, metric=metric,
                                              num_classes=12, label_files="Labels-ball.json", dataset="Ball",
                                              framerate=25)
            per_class = {c: round(100 * float(v), 2) for c, v in zip(tdeed_eval_classes(), res["a_mAP_per_class"])}
            # AdaSpot solo tiene 10 clases (sin FREE KICK ni GOAL): media de las comunes. En SoccerTrack v2 hay
            # clases casi vacías (HEADER: 1 en 128057): media también sobre las de >= 10 eventos.
            map10 = float(np.mean([v for c, v in per_class.items() if c not in ("FREE KICK", "GOAL")]))
            map_n10 = float(np.mean([v for c, v in per_class.items() if counts[c] >= 10]))
            map_pcbas8 = float(np.mean([per_class[c] for c in PCBAS_CLASSES.values()]))
            summary["results"][f"{name}/{metric}"] = {
                "mAP": round(100 * float(res["a_mAP"]), 2), "mAP_10cls": round(map10, 2),
                "mAP_n10": round(map_n10, 2), "mAP_pcbas8": round(map_pcbas8, 2), "per_class": per_class}
            print(f"{args.model} [{tag}, {name}, {metric}] mAP {100 * res['a_mAP']:.2f} (10 clases {map10:.2f}, "
                  f"clases con >=10 GT {map_n10:.2f}, 8 FOOTPASS {map_pcbas8:.2f})  "
                  + "  ".join(f"{c} {v:.1f}" for c, v in per_class.items()))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))


def tdeed_eval_classes() -> list[str]:
    """Orden de clases del EVENT_DICTIONARY de ``util.eval.aux_evaluate``."""
    return ["PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT",
            "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"]


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
