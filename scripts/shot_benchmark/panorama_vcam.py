"""Cámara virtual sobre la panorámica de SoccerTrack v2: recorta una ventana 16:9 que sigue al balón
(posición en metros del dataset -> píxeles del vídeo original vía spline sobre los 65 keypoints de
calibración de raw/<match>_keypoints.json; residuo 0,1 px). Trayectoria suavizada y velocidad de
cámara limitada para que parezca una realización. Sin reescalado: los jugadores conservan su tamaño
original (~46 px de alto en 4096x1080; en TV 720p son ~84 px).

Uso: python panorama_vcam.py MATCH HALF START_S END_S [--size 1280x720]
Salida: datasets/soccertrack-v2/clips/<match>_<half>_<start>-<end>_vcam.mp4"""
import argparse, json, subprocess, tempfile
from pathlib import Path
import cv2, numpy as np
from scipy.interpolate import RBFInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("match"); ap.add_argument("half", choices=["1st", "2nd"]); ap.add_argument("start_s", type=float); ap.add_argument("end_s", type=float)
ap.add_argument("--size", default="1280x720"); ap.add_argument("--max-speed", type=float, default=40.0, help="px/frame")
a = ap.parse_args()
W, H = map(int, a.size.split("x"))
D = Path("datasets/soccertrack-v2"); R = D / "raw" / a.match / f"{a.match}_"
kp = json.load(open(str(R) + "keypoints.json"))
P = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp]); I = np.array(list(kp.values()), dtype=float)
to_px = RBFInterpolator(P, I, kernel="thin_plate_spline", smoothing=1.0)
z = np.load(D / "ball" / f"{a.match}_{a.half}_ball.npz")
frames, bx, by = z["frame"].astype(int), z["x"].astype(float), z["y"].astype(float)

video = D / "videos" / a.match / f"{a.match}_panorama_{a.half}_half.mp4"
cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS); FW, FH = int(cap.get(3)), int(cap.get(4))
f0, f1 = int(a.start_s * fps), int(a.end_s * fps)
# posición del balón por frame (interpolando huecos), en píxeles, suavizada
idx = np.arange(f0, f1)
pos = {f: (x, y) for f, x, y in zip(frames, bx, by) if np.isfinite(x) and np.isfinite(y)}
known = np.array(sorted(pos)); kx = np.array([pos[f][0] for f in known]); ky = np.array([pos[f][1] for f in known])
mx, my = np.interp(idx, known, kx), np.interp(idx, known, ky)
px = to_px(np.column_stack([mx + 52.5, my + 34]))
k = int(fps)  # ~1 s de suavizado
sm = lambda v: np.convolve(np.pad(v, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
cx, cy = sm(px[:, 0]), sm(px[:, 1])
# limitar la velocidad de la cámara
for i in range(1, len(cx)):
    for c in (cx, cy):
        d = c[i] - c[i - 1]
        if abs(d) > a.max_speed: c[i] = c[i - 1] + np.sign(d) * a.max_speed

out = D / "clips" / f"{a.match}_{a.half}_{int(a.start_s)}-{int(a.end_s)}_vcam.mp4"; out.parent.mkdir(exist_ok=True)
tmp = Path(tempfile.mkdtemp()) / "vcam_raw.mp4"
w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
for i, f in enumerate(idx):
    ok, fr = cap.read()
    if not ok: break
    x0 = int(np.clip(cx[i] - W / 2, 0, FW - W)); y0 = int(np.clip(cy[i] - H / 2, 0, FH - H))
    w.write(fr[y0:y0 + H, x0:x0 + W])
    if i % 2500 == 0: print(f"  {i}/{len(idx)}", flush=True)
w.release(); cap.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
tmp.unlink()
print("FIN", out, flush=True)
