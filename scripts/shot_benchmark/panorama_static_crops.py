"""Recortes FIJOS de las dos zonas de tiro de la panorámica de SoccerTrack v2. La cámara no se mueve, así
que no hace falta seguir el balón (al contrario que panorama_vcam.py) y sirve tal cual en producción.
Dos vistas por lado:
  goalcrop  un recorte 16:9 que cubre la zona de tiro; T-DEED lo reescala a 796x448
  goaltile  ventanas de 796x448 a resolución nativa (T-DEED no reescala), con solape entre ellas
Zona de tiro: hasta 30 m de cada línea de fondo, sin las franjas de 4 m pegadas a las bandas. Las ventanas
salen de la calibración (65 keypoints -> spline, igual que panorama_vcam.py).

Uso: python panorama_static_crops.py MATCH HALF [--out-dir DIR] [--dry-run]
Salida: DIR/<half>_goalcrop_{L,R}.mp4, DIR/<half>_goaltile_{L,R}<i>.mp4 y DIR/crops.json"""
import argparse, json, math, subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator

TW, TH = 796, 448            # entrada de T-DEED
MIN_OVERLAP = 64             # px de solape mínimo entre ventanas nativas vecinas
MARGIN_X = 32                # px de margen horizontal alrededor de la zona
ZONES = {"L": (0.0, 30.0), "R": (75.0, 105.0)}   # metros desde la línea de fondo izquierda
Y_RANGE = (4.0, 64.0)

ap = argparse.ArgumentParser()
ap.add_argument("match"); ap.add_argument("half", choices=["1st", "2nd"])
ap.add_argument("--out-dir", type=Path, default=None); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
D = Path("datasets/soccertrack-v2")
out_dir = a.out_dir or D / "views" / a.match
video = D / "videos" / a.match / f"{a.match}_panorama_{a.half}_half.mp4"
kp = json.load(open(D / "raw" / a.match / f"{a.match}_keypoints.json"))
P = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp]); I = np.array(list(kp.values()), dtype=float)
to_px = RBFInterpolator(P, I, kernel="thin_plate_spline", smoothing=1.0)
probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True)
FW, FH = map(int, probe.stdout.strip().split(","))


def even(v):
    return int(round(v)) // 2 * 2


def spread(lo, hi, size, limit):
    """Orígenes de ventanas de ``size`` que cubren [lo, hi] con solape >= MIN_OVERLAP, dentro de [0, limit]."""
    span = hi - lo
    n = 1 if span <= size else math.ceil((span - MIN_OVERLAP) / (size - MIN_OVERLAP))
    starts = [(lo + hi - size) / 2] if n == 1 else np.linspace(lo, hi - size, n)
    return [even(min(max(s, 0), limit - size)) for s in starts]


windows = {}
for side, (x0, x1) in ZONES.items():
    gx, gy = np.meshgrid(np.linspace(x0, x1, 31), np.linspace(*Y_RANGE, 31))
    px = to_px(np.column_stack([gx.ravel(), gy.ravel()]))
    bx0, by0 = px.min(0); bx1, by1 = px.max(0)
    bx0, bx1 = max(bx0 - MARGIN_X, 0), min(bx1 + MARGIN_X, FW)
    # 16:9 que cubre la zona, centrado en ella
    w = min(max(bx1 - bx0, (by1 - by0) * 16 / 9), FW, FH * 16 / 9); h = w * 9 / 16
    w, h = even(w), even(h)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    windows[f"goalcrop_{side}"] = (w, h, even(min(max(cx - w / 2, 0), FW - w)), even(min(max(cy - h / 2, 0), FH - h)))
    # ventanas nativas
    for i, (tx, ty) in enumerate((tx, ty) for ty in spread(by0, by1, TH, FH) for tx in spread(bx0, bx1, TW, FW)):
        windows[f"goaltile_{side}{i}"] = (TW, TH, tx, ty)

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "crops.json").write_text(json.dumps({"frame": [FW, FH], "zones_m": ZONES, "y_range_m": Y_RANGE,
                                               "windows_whxy": windows}, indent=1))
for name, (w, h, x, y) in windows.items():
    print(f"{name:<14} {w}x{h} en ({x}, {y})", flush=True)
if a.dry_run:
    raise SystemExit

outs = {name: out_dir / f"{a.half}_{name}.mp4" for name in windows}
todo = [n for n, p in outs.items() if not p.exists()]
if not todo:
    print("ya hecho", flush=True); raise SystemExit
graph = f"[0:v]split={len(todo)}" + "".join(f"[s{i}]" for i in range(len(todo))) + ";" + ";".join(
    f"[s{i}]crop={windows[n][0]}:{windows[n][1]}:{windows[n][2]}:{windows[n][3]}[o{i}]" for i, n in enumerate(todo))
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-filter_complex", graph]
tmps = {n: outs[n].with_name(outs[n].stem + ".tmp.mp4") for n in todo}
for i, n in enumerate(todo):
    cmd += ["-map", f"[o{i}]", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(tmps[n])]
subprocess.run(cmd, check=True)
for n in todo:
    tmps[n].rename(outs[n])
print("FIN", a.half, ", ".join(todo), flush=True)
