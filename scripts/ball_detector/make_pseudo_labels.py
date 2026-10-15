"""Conjunto de entrenamiento (formato YOLO) para el detector de balón en panorámica, a partir de SoccerTrack v2.

Etiquetas automáticas con verificación cruzada, sin revisión manual:
  1. La pista de balón del dataset (ball/<partido>_<mitad>_ball.npz, metros) se proyecta a píxeles con la
     calibración (65 keypoints -> spline). Solo frames con el balón en juego (status != 0).
  2. YOLO11x COCO ("sports ball") a resolución nativa, por ventanas de 1024 px, sobre cada frame muestreado.
  3. Un frame se acepta si alguna detección cae a < GATE_X px en horizontal y entre -GATE_UP (por encima:
     balón en el aire, la pista es su sombra) y +GATE_DOWN px en vertical de la proyección. La CAJA de la
     etiqueta es la de esa detección (posición y tamaño reales), no la proyección. Sin confirmación se descarta.
  4. Por frame aceptado, un recorte de CROP x CROP con el balón en posición aleatoria (positivo) y, uno de cada
     NEG_EVERY, un recorte dentro del campo lejos del balón sin etiqueta (negativo: césped, jugadores, líneas).

Partición (docs/particion_datasets.md): train = 117092 118575 118576 118577 118578 128058; val = 117093 132877.
NUNCA 128057 ni 132831 (evaluación).

Uso (dentro del contenedor, cwd /app):
    python scripts/ball_detector/make_pseudo_labels.py MATCH HALF [--stride 10] [--max-frames N]
Salida: datasets/ball_panorama/{images,labels}/{train,val}/<match>_<half>_f<frame>[_neg].{jpg,txt},
        datasets/ball_panorama/stats_<match>_<half>.json (resumen) y data.yaml. Reanudable por mitad."""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts/shot_benchmark")
from panorama_ball_eval import gt_pixels, pitch_polygon, tile_starts, merge_close, run_yolo, video_path, TILE_W  # noqa: E402

SPLIT = {"train": ("117092", "118575", "118576", "118577", "118578", "128058"), "val": ("117093", "132877")}
OUT = Path("datasets/ball_panorama")
CROP = 1024
GATE_X, GATE_UP, GATE_DOWN = 30.0, 90.0, 30.0
NEG_EVERY, NEG_MIN_DIST = 3, 200.0
JPEG_Q = 92


def split_of(match):
    for s, ms in SPLIT.items():
        if match in ms:
            return s
    raise SystemExit(f"{match} no está en train ni val (¿es de evaluación? no se etiqueta)")


def crop_around(frame, cx, cy, rng, FW, FH):
    """Ventana CROP x CROP que contiene (cx, cy) en una posición aleatoria (margen de 48 px), dentro del frame."""
    x0 = int(np.clip(cx - rng.uniform(48, CROP - 48), 0, FW - CROP)); y0 = int(np.clip(cy - rng.uniform(48, CROP - 48), 0, max(FH - CROP, 0)))
    return x0, y0


def main():
    import cv2
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("match"); ap.add_argument("half", choices=("1st", "2nd"))
    ap.add_argument("--stride", type=int, default=10); ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    split = split_of(a.match)
    done = OUT / f"stats_{a.match}_{a.half}.json"
    if done.exists() and a.max_frames is None:
        print("ya hecho:", done); return
    for s in SPLIT:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True); (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "data.yaml").write_text(f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")

    gt, status = gt_pixels(a.match, a.half)
    poly = pitch_polygon(a.match)
    model = YOLO("/app/models/yolo11x.pt")
    cap = cv2.VideoCapture(str(video_path(a.match, a.half)))
    FW, FH, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_h = min(CROP, FH)
    starts = tile_starts(FW); imgsz = int(np.ceil(FH / 32) * 32)
    rng = random.Random(hash((a.match, a.half)) & 0xffff)
    st = dict(frames=0, in_play=0, confirmed=0, negatives=0, dists=[])
    t0 = time.perf_counter(); idx = 0
    while idx < n:
        if idx % a.stride:
            cap.grab(); idx += 1; continue
        ok, frame = cap.read()
        if not ok:
            break
        st["frames"] += 1
        g = gt[idx]
        if np.isfinite(g).all() and status[idx] != 0:
            st["in_play"] += 1
            tiles = [np.ascontiguousarray(frame[:, x0:x0 + TILE_W]) for x0 in starts]
            dets = merge_close([[d[0] + x0, d[1], d[2], d[3], d[4]] for x0, ds in zip(starts, run_yolo(model, tiles, imgsz)) for d in ds])
            ok_d = [d for d in dets if abs(d[0] - g[0]) <= GATE_X and -GATE_UP <= d[1] - g[1] <= GATE_DOWN]
            if ok_d:
                d = max(ok_d, key=lambda d: d[4])
                st["confirmed"] += 1; st["dists"].append(float(np.hypot(d[0] - g[0], d[1] - g[1])))
                x0, y0 = crop_around(frame, d[0], d[1], rng, FW, FH)
                crop = frame[y0:y0 + crop_h, x0:x0 + CROP]
                w, h = max(d[2], 6.0), max(d[3], 6.0)
                name = f"{a.match}_{a.half}_f{idx:06d}"
                cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                (OUT / "labels" / split / f"{name}.txt").write_text(f"0 {(d[0] - x0) / CROP:.6f} {(d[1] - y0) / crop_h:.6f} {w / CROP:.6f} {h / crop_h:.6f}\n")
                if st["confirmed"] % NEG_EVERY == 0:
                    # negativo: ventana con su centro dentro del campo y el balón (real y detectado) lejos
                    for _ in range(20):
                        nx = rng.uniform(0, FW - CROP); ny = rng.uniform(0, max(FH - crop_h, 0))
                        ccx, ccy = nx + CROP / 2, ny + crop_h / 2
                        if cv2.pointPolygonTest(poly, (ccx, ccy), False) < 0:
                            continue
                        if abs(g[0] - ccx) < CROP / 2 + NEG_MIN_DIST and abs(g[1] - ccy) < crop_h / 2 + NEG_MIN_DIST:
                            continue
                        if any(nx - 20 <= dd[0] <= nx + CROP + 20 and ny - 20 <= dd[1] <= ny + crop_h + 20 and dd[4] >= 0.3 for dd in dets):
                            continue  # otra "pelota" confiable dentro (balón de reserva...): mejor no enseñarla como fondo
                        nx, ny = int(nx), int(ny)
                        cv2.imwrite(str(OUT / "images" / split / f"{name}_neg.jpg"), frame[ny:ny + crop_h, nx:nx + CROP], [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                        (OUT / "labels" / split / f"{name}_neg.txt").write_text("")
                        st["negatives"] += 1
                        break
        idx += 1
        if st["frames"] % 500 == 0:
            el = time.perf_counter() - t0
            print(f"  {a.match} {a.half} frame {idx}/{n}: {st['confirmed']} confirmados de {st['in_play']} en juego, "
                  f"{el / st['frames']:.2f} s/frame, ~{(n - idx) / a.stride * el / st['frames'] / 60:.0f} min restantes", flush=True)
        if a.max_frames and st["frames"] >= a.max_frames:
            break
    summary = {k: v for k, v in st.items() if k != "dists"}
    summary.update(split=split, stride=a.stride, dist_median_px=float(np.median(st["dists"])) if st["dists"] else None,
                   confirm_rate=st["confirmed"] / max(st["in_play"], 1), seconds=round(time.perf_counter() - t0))
    print("FIN", json.dumps(summary), flush=True)
    if a.max_frames is None:
        done.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
