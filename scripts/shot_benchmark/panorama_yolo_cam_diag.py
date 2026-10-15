"""Tiro a tiro: ¿dónde apuntaba la cámara virtual guiada por nuestro detector cuando se produjo cada tiro, y qué
vio T-DEED? Compara con la cámara guiada por el balón del dataset. Separa dos causas de fallo: la cámara no estaba
sobre el balón (pista mala en ese instante) o sí lo estaba y aun así T-DEED no lo reconoció (encuadre/movimiento).

Uso (cwd /app): python panorama_yolo_cam_diag.py MATCH [--tag _t03] [--mode tiles1024]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts/shot_benchmark")
from panorama_ball_eval import gt_pixels, shot_frames, smooth_track, load_dets, FPS, D, OUT

TD = Path("output/shot_benchmark/tdeed/SoccerNetBall_challenge2_heads_views")
W, H = 796, 448
ap = argparse.ArgumentParser(); ap.add_argument("match"); ap.add_argument("--tag", default=""); ap.add_argument("--mode", default="tiles1024")
a = ap.parse_args()


def p_shot(npz, t, tol=1.0):
    z = np.load(npz); names = list(z["classes"]); p = z["probs"][:, [names.index("SHOT"), names.index("GOAL")]].astype(np.float32).sum(1)
    p = np.convolve(p, np.ones(6) / 6, mode="same")
    tt = np.arange(len(p)) * int(z["stride"]) / FPS
    w = (tt >= t - tol) & (tt <= t + tol)
    return float(p[w].max()) if w.any() else float("nan")


bas = json.load(open(D / "bas" / a.match / f"{a.match}_12_class_events.json"))["actions"]
import cv2
print(f"{'instante':<13}{'etiq.':<6}{'det ±2s':>8}{'balón-cámara en t (dx, dy) px':>32}{'en t-1s':>18}{'dentro':>8}{'P(tiro) YOLO':>14}{'P(tiro) GT':>12}")
rows = []
for h in ("1st", "2nd"):
    gt, status = gt_pixels(a.match, h)
    cam = np.load(OUT / f"{a.match}_{h}_{a.mode}_cam{a.tag}.npz")
    FW, FH = (lambda c: (int(c.get(3)), int(c.get(4))))(cv2.VideoCapture(str(D / "videos" / a.match / f"{a.match}_panorama_{h}_half.mp4")))
    n = len(cam["frame"])
    _, c = smooth_track(cam["frame"], np.column_stack([cam["x"], cam["y"]]), n, 1)   # el mismo suavizado que panorama_vcam.py
    # la ventana se recorta a los bordes del frame: centro efectivo
    cx = np.clip(c[:, 0] - W / 2, 0, FW - W) + W / 2; cy = np.clip(c[:, 1] - H / 2, 0, FH - H) + H / 2
    has = cam["has_det"]
    yolo = TD / f"{a.match} [{h}_vcam796yolo{a.tag}].npz"; ref = TD / f"{a.match} [{h}_vcam796].npz"
    for ev in sorted((e for e in bas if e["gameTime"].startswith(("1" if h == "1st" else "2") + " -") and e["label"] in ("SHOT", "GOAL")), key=lambda e: int(e["frame"])):
        f = int(ev["frame"]) - 1; t = f / FPS
        if f >= n or not np.isfinite(gt[f]).all():
            continue
        dx, dy = gt[f, 0] - cx[f], gt[f, 1] - cy[f]
        f1 = max(f - 25, 0); dx1, dy1 = gt[f1, 0] - cx[f1], gt[f1, 1] - cy[f1]
        inside = abs(dx) <= W / 2 and abs(dy) <= H / 2
        det = bool(has[max(f - 50, 0): f + 50].any())
        py, pg = p_shot(yolo, t), p_shot(ref, t)
        rows.append((inside, py, pg))
        print(f"{h} {int(t // 60):2d}:{t % 60:04.1f}  {ev['label']:<6}{'sí' if det else 'NO':>8}{dx:+16.0f}, {dy:+6.0f}{dx1:+11.0f}, {dy1:+5.0f}{'sí' if inside else 'NO':>8}{py:14.2f}{pg:12.2f}")
ins = np.array([r[0] for r in rows]); py = np.array([r[1] for r in rows]); pg = np.array([r[2] for r in rows])
print(f"\nTiros con el balón dentro de la ventana en el instante del tiro: {ins.sum()} de {len(rows)}")
for name, m in (("cámara dentro", ins), ("cámara fuera", ~ins)):
    if m.any():
        print(f"  {name:<14}: P(tiro) mediana YOLO {np.median(py[m]):.2f} vs GT {np.median(pg[m]):.2f}; "
              f"tiros con P>=0,3: YOLO {(py[m] >= 0.3).sum()} / GT {(pg[m] >= 0.3).sum()} de {m.sum()}")
