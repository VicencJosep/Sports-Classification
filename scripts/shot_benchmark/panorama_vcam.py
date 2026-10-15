"""Cámara virtual sobre la panorámica de SoccerTrack v2: recorta una ventana 16:9 que sigue al balón
(posición en metros del dataset -> píxeles del vídeo original vía spline sobre los 65 keypoints de
calibración de raw/<match>_keypoints.json; residuo 0,1 px). Trayectoria suavizada y velocidad de
cámara limitada para que parezca una realización. Sin reescalado: los jugadores conservan su tamaño
original (~46 px de alto en 4096x1080; en TV 720p son ~84 px).

Uso: python panorama_vcam.py MATCH HALF START_S END_S [--size 1280x720] [--out RUTA] [--camera NPZ]
     (END_S <= 0: hasta el final)
Salida: datasets/soccertrack-v2/clips/<match>_<half>_<start>-<end>_vcam.mp4 (o --out)

OJO: la pista del balón del dataset está reconstruida entre los eventos anotados, así que el movimiento
de esta cámara lleva implícito cuándo ocurren. Vale para probar modelos ya entrenados, no para entrenar.
Con --camera NPZ (salida de panorama_ball_eval.py track: x, y en píxeles por frame) la cámara sigue esa pista
-- la de nuestro detector -- en vez del balón del dataset; el suavizado y el límite de velocidad son los mismos."""
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np
from scipy.interpolate import RBFInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("match"); ap.add_argument("half", choices=["1st", "2nd"]); ap.add_argument("start_s", type=float); ap.add_argument("end_s", type=float)
ap.add_argument("--size", default="1280x720"); ap.add_argument("--max-speed", type=float, default=40.0, help="px/frame")
ap.add_argument("--out", type=Path, default=None); ap.add_argument("--camera", type=Path, default=None)
ap.add_argument("--smooth-s", type=float, default=1.0, help="ventana de la media móvil de la cámara, en segundos")
ap.add_argument("--enhance", action="store_true", help="partidos nocturnos: gamma automática (luminancia media -> 0,45, "
                "estimada con media exponencial para que no parpadee) + contraste local CLAHE sobre la luminancia")
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
n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
f0, f1 = int(a.start_s * fps), (n_video if a.end_s <= 0 else min(int(a.end_s * fps), n_video))
# posición del balón por frame (interpolando huecos), en píxeles, suavizada
idx = np.arange(f0, f1)
pos = {f: (x, y) for f, x, y in zip(frames, bx, by) if np.isfinite(x) and np.isfinite(y)}
known = np.array(sorted(pos)); kx = np.array([pos[f][0] for f in known]); ky = np.array([pos[f][1] for f in known])
mx, my = np.interp(idx, known, kx), np.interp(idx, known, ky)
px = to_px(np.column_stack([mx + 52.5, my + 34]))
if a.camera:
    c = np.load(a.camera)
    px = np.column_stack([np.interp(idx, c["frame"], c["x"]), np.interp(idx, c["frame"], c["y"])])
k = max(1, int(round(a.smooth_s * fps)))  # media móvil centrada (sin retardo: el análisis es offline)
sm = lambda v: np.convolve(np.pad(v, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
cx, cy = sm(px[:, 0]), sm(px[:, 1])
# limitar la velocidad de la cámara
for i in range(1, len(cx)):
    for c in (cx, cy):
        d = c[i] - c[i - 1]
        if abs(d) > a.max_speed: c[i] = c[i - 1] + np.sign(d) * a.max_speed

out = a.out or D / "clips" / f"{a.match}_{a.half}_{int(a.start_s)}-{int(a.end_s)}_vcam.mp4"; out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_name(out.stem + ".tmp.mp4")
# frames en crudo directos a libx264: una sola compresión (antes pasaba por un mp4v intermedio)
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                        "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        str(tmp)], stdin=subprocess.PIPE)
clahe, mean_l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)), [None]


def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    m = lab[..., 0].mean() / 255
    mean_l[0] = m if mean_l[0] is None else 0.98 * mean_l[0] + 0.02 * m
    gamma = float(np.clip(np.log(0.45) / np.log(max(mean_l[0], 1e-3)), 0.35, 1.0))
    lab[..., 0] = clahe.apply(cv2.LUT(lab[..., 0], ((np.arange(256) / 255) ** gamma * 255).astype(np.uint8)))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
for i, f in enumerate(idx):
    ok, fr = cap.read()
    if not ok: break
    x0 = int(np.clip(cx[i] - W / 2, 0, FW - W)); y0 = int(np.clip(cy[i] - H / 2, 0, FH - H))
    crop = np.ascontiguousarray(fr[y0:y0 + H, x0:x0 + W])
    enc.stdin.write((enhance(crop) if a.enhance else crop).tobytes())
    if i % 2500 == 0: print(f"  {i}/{len(idx)}", flush=True)
enc.stdin.close(); cap.release()
if enc.wait() != 0:
    raise SystemExit(f"ffmpeg falló escribiendo {tmp}")
tmp.rename(out)
print("FIN", out, flush=True)
