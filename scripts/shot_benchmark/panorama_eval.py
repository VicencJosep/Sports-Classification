"""Viabilidad en panorámica: T-DEED (challenge2) sobre el tramo 0-630 s de la 1ª parte de SoccerTrack v2
117093, panorámica tal cual vs cámara virtual, contra las anotaciones BAS del dataset. También evalúa el
telemetry.json del pipeline completo sobre la cámara virtual. Uso: cat common.py panorama_eval.py | python -"""
import json
from pathlib import Path
import numpy as np
from core.spotting import find_peaks
from scripts.evaluate_spotting import Prediction, match_1to1, count, Counts, average_precision

END_S, FPS = 630.0, 25.0
bas = [a for a in json.load(open("datasets/soccertrack-v2/bas/117093/117093_12_class_events.json"))["actions"]
       if a["gameTime"].startswith("1 -") and int(a["position"]) / 1000 <= END_S]
gt = {}
for a in bas:
    gt.setdefault(a["label"], []).append(int(a["position"]) / 1000)
gt = {k: sorted(v) for k, v in gt.items()}
print("GT 0-630 s:", {k: len(v) for k, v in sorted(gt.items(), key=lambda kv: -len(kv[1]))})

TD = Path("output/shot_benchmark/tdeed/SoccerNetBall_challenge2_pano")
def peaks_of(npz, cls, nms_s, smooth):
    z = np.load(npz); names = list(z["classes"]); p = z["probs"][:, names.index(cls)].astype(np.float32)
    if smooth > 1: p = np.convolve(p, np.ones(smooth) / smooth, mode="same")
    t = (np.arange(len(p)) * int(z["stride"]) / FPS).tolist()
    return [Prediction(tt, tt, tt, float(s)) for tt, s in find_peaks(t, p.tolist(), 0.01, nms_s)]

def row(label, g, preds, tol):
    ap = average_precision([(g, preds)], tol)
    cells = []
    for th in (0.3, 0.5):
        c = count(g, preds, tol, th); cells.append(f"θ{th}: P {100*c.precision:3.0f}% R {100*c.recall:3.0f}% ({c.n_pred:3d} alarmas)")
    ceil = count(g, preds, tol, 0.0).recall
    print(f"  {label:<34} AP {100*ap:5.1f}%  techo R {100*ceil:3.0f}%  " + "  ".join(cells))

for cls, nms, smooth in (("SHOT", 4.0, 6), ("PASS", 0.5, 1), ("DRIVE", 0.5, 1), ("HIGH PASS", 0.5, 1), ("CROSS", 2.0, 1), ("THROW IN", 2.0, 1), ("OUT", 2.0, 1)):
    if cls not in gt: continue
    for tol in ((1.0, 3.0) if cls == "SHOT" else (1.0,)):
        print(f"\n=== {cls} ({len(gt[cls])} anotados), tolerancia ±{tol:g} s ===")
        for name, f in (("panorámica entera (4096x1080→796x448)", "clips [117093_1st_0-630].npz"), ("cámara virtual 1280x720 → 796x448", "clips [117093_1st_0-630_vcam].npz")):
            row(name, gt[cls], peaks_of(TD / f, cls, nms, smooth), tol)
# referencia: ¿qué hacía T-DEED con estas mismas clases en TV (3 EFL limpios)? -> ver docs/estudio_deteccion_tiros.md

# pipeline completo sobre la cámara virtual (telemetry.json): acciones de balón y tiros
tel = json.load(open("output/panorama_vcam/telemetry.json"))
MAP = {"PASS": "pass", "DRIVE": "drive", "HIGH PASS": "high_pass", "HEADER": "header", "THROW IN": "throw_in"}
print(f"\n=== pipeline completo sobre la cámara virtual: {len(tel['ball_actions'])} acciones, {len(tel['events'])} tramos de evento ===")
tot = Counts()
for raw, ours in MAP.items():
    g = gt.get(raw, []); preds = [Prediction(a["frame_idx"] / FPS, a["frame_idx"] / FPS, a["frame_idx"] / FPS, a["confidence"]) for a in tel["ball_actions"] if a["label"] == ours]
    c = count(g, preds, 1.0); tot.add(c)
    print(f"  {ours:<10} GT {c.n_gt:3d}  pred {c.n_pred:3d}  TP {c.tp:3d}  P {100*c.precision:3.0f}%  R {100*c.recall:3.0f}%")
print(f"  {'TOTAL':<10} GT {tot.n_gt:3d}  pred {tot.n_pred:3d}  TP {tot.tp:3d}  P {100*tot.precision:3.0f}%  R {100*tot.recall:3.0f}%  F1 {100*tot.f1:3.0f}%   (TV, ground_truth/: P 87% R 88% F1 88%)")
shots = [e for e in tel["events"] if e["label"] in ("shot", "goal")]
g = gt.get("SHOT", []) + gt.get("GOAL", [])
preds = [Prediction((e["start_s"] + e["end_s"]) / 2, e["start_s"], e["end_s"], e["peak_confidence"]) for e in shots]
c = count(sorted(g), preds, 1.0)
print(f"  tiros/goles: GT {c.n_gt}  tramos emitidos {c.n_pred}  aciertos {c.tp}  -> " + ", ".join(f"{int(e['start_s']//60)}:{(e['start_s']+1)%60:04.1f} ({e['peak_confidence']:.2f})" for e in shots))
print("  tiros GT:", ", ".join(f"{int(t//60)}:{t%60:04.1f}" for t in sorted(g)))
