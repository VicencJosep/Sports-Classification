"""Las 12 clases de acciones de balón (no solo tiros) sobre las vistas de cámara virtual ya escaneadas por T-DEED:

  1. Cobertura de la cámara (geometría, sin T-DEED): ¿está el balón anotado dentro de la ventana 796x448 en el instante
     de cada evento anotado, y durante [-0,5 s, +2 s]? Responde a "¿qué parte del partido abarca la cámara?".
  2. AP de T-DEED por clase (cabeza de acciones de balón, picos con NMS 0,5 s, sin suavizar), AP ±1 s y ±3 s, y mAP
     sobre las clases con >= 20 eventos. Responde a "¿cómo reconoce T-DEED cada cosa sobre esa cámara?".

Vistas: patrón de sufijo de vídeo (vcam796 = cámara del balón anotado; vcam796<tag> = trayectoria de vcam_path.py
path --tag _<tag>, cuyo npz output/panorama_ball/<match>_<half>_vcam_<tag>.npz da la cobertura; sin npz, solo AP).
Uso (dentro del contenedor, cwd /app):
    python scripts/shot_benchmark/panorama_class_eval.py 117093 132877 [--views vcam796 vcam796vl1]
NOTA: los tiros aquí (SHOT sin GOAL, NMS 0,5 s) no coinciden con la métrica principal de panorama_views_eval.py
(SHOT+GOAL, suavizado 6, NMS 4 s); sirven para comparar vistas entre sí, no con aquella cifra."""
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/shot_benchmark")
import panorama_ball_eval as E, vcam_path as V  # noqa: E402
from core.spotting import find_peaks  # noqa: E402
from scripts.evaluate_spotting import Prediction, average_precision  # noqa: E402

TD = Path(f"output/shot_benchmark/tdeed/{os.environ.get('TDEED_MODEL', 'SoccerNetBall_challenge2')}_heads_views")   # TDEED_MODEL=<afinado> para evaluar otro checkpoint
W, H = 796, 448
PRE, POST = 12, 50          # frames: -0,5 s, +2 s
HALVES = (("1st", "1"), ("2nd", "2"))


def camera_path(view, m, h):
    """Centro de la cámara por frame para una vista, o None si no se puede reconstruir."""
    if view == "vcam796":
        return V.camera("dataset", m, h, "tiles1024_pano")
    p = Path(f"output/panorama_ball/{m}_{h}_vcam_{view[len('vcam796'):]}.npz")
    if not view.startswith("vcam796") or not p.exists():
        return None
    z = np.load(p)
    return np.column_stack([z["x"], z["y"]])


def coverage(matches, views, bas):
    per = {}   # clase -> vista -> [(en t, en ventana [-0,5,+2 s]), ...]
    for m in matches:
        for h, hh in HALVES:
            gt, _ = E.gt_pixels(m, h)
            z = np.load(E.D / "ball" / f"{m}_{h}_ball.npz"); parked = (z["x"] == 0) & (z["y"] == 0)
            fw, fh = V.frame_size(m, h)
            inw = {}
            for v in views:
                cam = camera_path(v, m, h)
                if cam is None:
                    continue
                c = np.column_stack([np.clip(cam[:, 0], W / 2, fw - W / 2), np.clip(cam[:, 1], H / 2, fh - H / 2)])
                n = min(len(c), len(gt))
                inw[v] = (np.abs(gt[:n, 0] - c[:n, 0]) <= W / 2) & (np.abs(gt[:n, 1] - c[:n, 1]) <= H / 2)
            if not inw:
                continue
            n = min(len(gt), *[len(x) for x in inw.values()])
            valid = np.isfinite(gt[:n]).all(1) & ~parked[:n]
            for a in bas[m]:
                f = int(a["frame"]) - 1
                if not a["gameTime"].startswith(f"{hh} -") or f < 0 or f >= n or not valid[f]:
                    continue
                lo, hi = max(f - PRE, 0), min(f + POST, n); vv = valid[lo:hi]
                for v, x in inw.items():
                    per.setdefault(a["label"], {}).setdefault(v, []).append((x[f], x[lo:hi][vv].mean()))
    if not per:
        print("  (ninguna vista con trayectoria reconstruible)"); return
    vs = [v for v in views if any(v in d for d in per.values())]
    print(f"{'clase':<26}{'n':>6}" + "".join(f"{v + ': en t':>20}{'[-0,5,+2 s]':>13}" for v in vs))
    rows = sorted(per.items(), key=lambda kv: -len(next(iter(kv[1].values()))))
    for cls, d in rows + [("TODOS", {v: [r for _, d in rows for r in d.get(v, [])] for v in vs})]:
        n = len(next(iter(d.values())))
        print(f"{cls:<26}{n:>6}" + "".join(f"{100 * np.mean([r[0] for r in d[v]]):19.1f}%{100 * np.mean([r[1] for r in d[v]]):12.1f}%"
                                           if d.get(v) else f"{'-':>20}{'-':>13}" for v in vs))


def curve(zs, cls):
    per = [z["probs"][:, list(z["classes"]).index(cls)].astype(np.float32) for z in zs]
    n = min(len(p) for p in per)
    return np.arange(n) * int(zs[0]["stride"]) / E.FPS, np.max([p[:n] for p in per], axis=0)


def ap_by_class(matches, views, bas):
    data, classes = {}, None
    for v in views:
        for m in matches:
            for h, _ in HALVES:
                zs = [np.load(f) for f in sorted(TD.glob(f"{m} [[]{h}_{v}].npz"))]
                if not zs:
                    print(f"  falta el escaneo de {m} {h} {v}"); return
                data[(v, m, h)] = zs; classes = classes or list(zs[0]["classes"])
    cnt = {c: sum(a["label"] == c for m in matches for a in bas[m]) for c in classes}
    missing = {a["label"] for m in matches for a in bas[m]} - set(classes)
    if missing:
        print(f"  etiquetas sin clase en T-DEED: {', '.join(sorted(missing))}")
    print(f"{'clase':<26}{'n':>6}" + "".join(f"{v:>31}" for v in views))
    print(f"{'':<32}" + "".join(f"{'AP±1':>15}{'AP±3':>16}" for _ in views))
    maps = {v: [] for v in views}
    for cls in sorted([c for c in classes if cnt[c]], key=lambda c: -cnt[c]):
        cells = []
        for v in views:
            pc = []
            for m in matches:
                for h, hh in HALVES:
                    t, s = curve(data[(v, m, h)], cls)
                    gt = sorted(int(a["position"]) / 1000 for a in bas[m] if a["gameTime"].startswith(f"{hh} -") and a["label"] == cls)
                    pc.append((gt, [Prediction(tt, tt, tt, float(sc)) for tt, sc in find_peaks(t.tolist(), s.tolist(), 0.01, 0.5)]))
            ap = (average_precision(pc, 1.0), average_precision(pc, 3.0))
            if cnt[cls] >= 20:
                maps[v].append(ap)
            cells.append(f"{100 * ap[0]:14.1f}%{100 * ap[1]:15.1f}%")
        print(f"{cls:<26}{cnt[cls]:>6}" + "".join(cells))
    print(f"{'mAP (clases con n>=20)':<32}" + "".join(f"{100 * np.mean([x[0] for x in maps[v]]):14.1f}%{100 * np.mean([x[1] for x in maps[v]]):15.1f}%" for v in views))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matches", nargs="+"); ap.add_argument("--views", nargs="+", default=["vcam796", "vcam796vl1"])
    a = ap.parse_args()
    bas = {m: json.load(open(f"datasets/soccertrack-v2/bas/{m}/{m}_12_class_events.json"))["actions"] for m in a.matches}
    print(f"Partidos: {' '.join(a.matches)} — {sum(len(b) for b in bas.values())} eventos anotados")
    print("\n=== Cobertura: balón anotado dentro de la ventana 796x448 de la cámara ===")
    coverage(a.matches, a.views, bas)
    print("\n=== T-DEED: AP por clase (cabeza de acciones de balón, NMS 0,5 s) ===")
    ap_by_class(a.matches, a.views, bas)
