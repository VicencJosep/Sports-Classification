"""Tiros en partidos COMPLETOS de SoccerTrack v2 con T-DEED (SoccerNetBall_challenge2, sin entrenar con
SoccerTrack) sobre varias vistas de la panorámica, generadas por run_panorama_views.sh: cámara virtual a
tres escalas y recortes fijos de las dos zonas de tiro.

Tiro = SHOT o GOAL (convención gol ⊂ tiro). Puntuación por frame: P(SHOT) + P(GOAL) de la cabeza de acciones
de balón ("bas") o Shots on target + Shots off target + Goal de la cabeza SoccerNet-v2 ("sn"). Las vistas con
varias ventanas (una por área, o varias nativas) se fusionan con el máximo por frame. Picos con suavizado de
6 frames muestreados (~0,5 s) y NMS de 4 s, como core/ball_spotting.py en el pipeline. Con varios partidos,
AP y P/R se calculan agrupando todas las mitades.

Uso: python panorama_views_eval.py [MATCH ...] [--views vcam796 goaltile ...]   (por defecto 117093, todas)"""
import os
import argparse, json
from pathlib import Path
import numpy as np
from core.spotting import find_peaks
from scripts.evaluate_spotting import Prediction, count, Counts, average_precision

FPS, SMOOTH, NMS_S, PROD_TH = 25.0, 6, 4.0, 0.3
TD = Path(f"output/shot_benchmark/tdeed/{os.environ.get('TDEED_MODEL', 'SoccerNetBall_challenge2')}_heads_views")   # TDEED_MODEL=<afinado> para evaluar otro checkpoint
HALVES = {"1st": "1", "2nd": "2"}
VIEWS = {  # nombre -> patrón de sufijo de vídeo (glob) dentro de "<MATCH> [<half>_<patrón>].npz"
    "cámara virtual 1280x720 (x0,62)": "vcam1280",
    "cámara virtual 796x448 (x1)": "vcam796",
    "cámara virtual 532x300 (x1,5)": "vcam532",
    "cám. virtual 796x448 guiada por YOLO": "vcam796yolo",   # pista de nuestro detector (panorama_ball_eval.py track)
    "recorte fijo 16:9 por área (~x0,55)": "goalcrop_*",
    "ventanas fijas 796x448 por área (x1)": "goaltile_*",
}
HEADS = {"bas": ("probs", "classes", ["SHOT", "GOAL"]),
         "sn": ("probs_sn", "classes_sn", ["Shots on target", "Shots off target", "Goal"])}

ap = argparse.ArgumentParser()
ap.add_argument("matches", nargs="*", default=["117093"])
ap.add_argument("--views", nargs="*", default=None, help="prefijos de vista, p. ej. vcam796 goaltile")
a = ap.parse_args()
MATCHES = a.matches or ["117093"]
views = {k: v for k, v in VIEWS.items() if a.views is None or v in a.views}
for p in a.views or []:
    if p not in VIEWS.values():
        views[f"vista {p}"] = p; VIEWS[f"vista {p}"] = p   # variantes sin nombre propio (p. ej. vcam796yolo_t03)
bas = {m: json.load(open(f"datasets/soccertrack-v2/bas/{m}/{m}_12_class_events.json"))["actions"] for m in MATCHES}
SPLITS = [(m, h) for m in MATCHES for h in HALVES]   # cada mitad es una "secuencia" para el AP


def events(m, half, labels):
    return sorted((a for a in bas[m] if a["gameTime"].startswith(f"{HALVES[half]} -") and a["label"] in labels),
                  key=lambda a: int(a["position"]))


def gt_times(m, half, labels):
    return [int(a["position"]) / 1000 for a in events(m, half, labels)]


def load_view(m, half, pattern):
    return [np.load(f) for f in sorted(TD.glob(f"{m} [[]{half}_{pattern}].npz"))]


def curve(zs, head, classes=None):
    """Puntuación por frame fusionada (máximo entre ventanas) y tiempos en segundos."""
    key, ckey, default = HEADS[head]
    cls = classes or default
    per = [z[key][:, [list(z[ckey]).index(c) for c in cls]].astype(np.float32).sum(1) for z in zs]
    n = min(len(p) for p in per)
    return np.arange(n) * int(zs[0]["stride"]) / FPS, np.max([p[:n] for p in per], axis=0)


def peaks(t, s, smooth=SMOOTH, nms_s=NMS_S, min_score=0.001):
    if smooth > 1:
        s = np.convolve(s, np.ones(smooth) / smooth, mode="same")
    return [Prediction(tt, tt, tt, float(v)) for tt, v in find_peaks(t.tolist(), s.tolist(), min_score, nms_s)]


def at(per, th, tol=3.0):
    c = Counts()
    for g, p in per:
        c.add(count(g, p, tol, th))
    return c


def summary(per, tol=3.0):
    """AP a ±0,5/±1/±3 s, techo de recall, umbral de producción y mejor F1 (con nº de tiros emitidos)."""
    aps = {t: average_precision(per, t) for t in (0.5, 1.0, 3.0)}
    n_gt = sum(len(g) for g, _ in per)
    ceil = sum(count(g, p, tol, 0.0).tp for g, p in per) / max(n_gt, 1)
    best_th, best = 0.0, at(per, 0.0)
    # rejilla como evaluate_spotting._best_threshold, más fina por debajo de 0,05 (vistas con puntuaciones bajas)
    for th in np.concatenate([np.arange(0.001, 0.05, 0.001), np.arange(0.05, 1.0, 0.005)]):
        c = at(per, float(th))
        if c.f1 > best.f1:
            best_th, best = float(th), c
    return aps, ceil, at(per, PROD_TH), best_th, best


gts = {(m, h): gt_times(m, h, ("SHOT", "GOAL")) for m, h in SPLITS}
n_gt = sum(len(v) for v in gts.values())
print(f"Partidos: {' '.join(MATCHES)} — {n_gt} tiros (SHOT+GOAL)")
print("Métrica principal: AP con tolerancia ±3 s; también ±1 s y ±0,5 s (= τ 1 s del paper de SoccerTrack v2)\n")

results = {}
for head in HEADS:
    print(f"=== Cabeza {'acciones de balón: SHOT+GOAL' if head == 'bas' else 'SoccerNet-v2: Shots on/off target + Goal'} ===")
    print(f"{'vista':<38}{'AP±0,5':>8}{'AP±1':>7}{'AP±3':>7}{'techo R':>9}   {'θ=0,3: P / R (emitidos)':<26}{'mejor F1: θ, P / R, emitidos'}")
    for view, pattern in views.items():
        per, missing = {}, []
        for m, h in SPLITS:
            zs = load_view(m, h, pattern)
            if not zs:
                missing.append(f"{m} {h}"); continue
            per[(m, h)] = (gts[(m, h)], peaks(*curve(zs, head)))
        if missing:
            print(f"{view:<38}  (falta el escaneo de: {', '.join(missing)})"); continue
        aps, ceil, prod, th, best = summary(list(per.values()))
        results[(head, view)] = (aps[3.0], per)
        print(f"{view:<38}{100*aps[0.5]:7.1f}%{100*aps[1.0]:6.1f}%{100*aps[3.0]:6.1f}%{100*ceil:8.0f}%   "
              f"{100*prod.precision:3.0f}% / {100*prod.recall:3.0f}% ({prod.n_pred:3d})       "
              f"{th:.3f}, {100*best.precision:3.0f}% / {100*best.recall:3.0f}%, {best.n_pred} de {n_gt}")
    print()

if not results:
    raise SystemExit("Sin escaneos todavía")
head, view = max(results, key=lambda k: results[k][0])
pattern = VIEWS[view]
per = results[(head, view)][1]

if len(MATCHES) > 1:
    # Por partido: AP y, con el umbral de producción, cuántos tiros emite frente a los reales
    print(f"=== Por partido, mejor vista ({view}, cabeza {head}) ===")
    print(f"{'partido':<9}{'tiros':>6}{'AP±3':>8}   θ=0,3: emitidos / aciertos / P / R")
    for m in MATCHES:
        pm = [per[(m, h)] for h in HALVES]
        c = at(pm, PROD_TH)
        print(f"{m:<9}{sum(len(g) for g, _ in pm):>6}{100 * average_precision(pm, 3.0):7.1f}%   "
              f"{c.n_pred:3d} / {c.tp:3d} / {100*c.precision:3.0f}% / {100*c.recall:3.0f}%")
    print(f"Umbral θ -> P / R agrupados (±3 s): " + "   ".join(
        f"{th:g}: {100*at(list(per.values()), th).precision:.0f}% / {100*at(list(per.values()), th).recall:.0f}%"
        for th in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5)))
    print()

# Diagnóstico: ¿T-DEED ve el golpeo y lo llama de otra forma?
print(f"=== Diagnóstico ({view}, cabeza {head}): clase ganadora de T-DEED en ±1 s de cada tiro anotado ===")
if len(MATCHES) == 1:
    print("instante       etiqueta  máx P(tiro) ±1 s  rango entre picos   clase ganadora (sin fondo)")
labels_of, invisible = {}, 0
for m, half in SPLITS:
    zs = load_view(m, half, pattern)
    t, s = curve(zs, head)
    scores = sorted((p.score for p in peaks(t, s)), reverse=True)
    s = np.convolve(s, np.ones(SMOOTH) / SMOOTH, mode="same")  # la misma curva sobre la que se buscan los picos
    names = list(zs[0]["classes"])
    n = min(len(z["probs"]) for z in zs)
    P = np.max([z["probs"][:n].astype(np.float32) for z in zs], axis=0)
    for a in events(m, half, ("SHOT", "GOAL")):
        tg = int(a["position"]) / 1000
        win = (t >= tg - 1) & (t <= tg + 1)
        if not win.any():
            continue
        local = float(s[win].max())
        fg = P[win][:, 1:]
        winner = names[1 + int(np.unravel_index(fg.argmax(), fg.shape)[1])]
        labels_of[winner] = labels_of.get(winner, 0) + 1
        invisible += local < 0.05
        if len(MATCHES) == 1:
            rank = 1 + sum(sc > local for sc in scores)
            print(f"{half} {int(tg // 60):2d}:{tg % 60:04.1f}  {a['label']:<8}  {local:15.3f}  {rank:>17}   {winner} ({fg.max():.2f})")
print("Clase ganadora en los tiros:", {str(k): v for k, v in sorted(labels_of.items(), key=lambda kv: -kv[1])},
      f"| tiros casi invisibles (P(tiro) < 0,05): {invisible} de {n_gt}")

# Control: pases y conducciones (en 0-630 s de 117093 la cámara virtual 1280 daba AP ±1 s de 78 % y 76 %)
print("\n=== Control: PASS y DRIVE, AP ±1 s (cabeza de acciones de balón, NMS 0,5 s) ===")
for view, pattern in views.items():
    cells = []
    for cls in ("PASS", "DRIVE"):
        pc = []
        for m, h in SPLITS:
            zs = load_view(m, h, pattern)
            if not zs:
                break
            t, s = curve(zs, "bas", [cls])
            pc.append((gt_times(m, h, (cls,)), peaks(t, s, smooth=1, nms_s=0.5, min_score=0.01)))
        else:
            cells.append(f"{cls} {100 * average_precision(pc, 1.0):5.1f}%")
    if cells:
        print(f"  {view:<38} " + "   ".join(cells))
