"""¿Cómo de bien detecta el balón nuestro detector (YOLO11x COCO, clase "sports ball", el de
core/tracking.py::YoloDetector) en la panorámica de SoccerTrack v2? Verdad de referencia: la pista de balón del
dataset (ball/<partido>_<mitad>_ball.npz, metros) proyectada a píxeles con los 65 keypoints de calibración.
Es la posición en el suelo: con el balón en el aire la imagen queda más arriba, de ahí las tolerancias anchas.
Solo cuentan los frames con el balón en juego (status != 0; 0 = BALLOUT del proveedor).

Tres formas de pasar el frame por YOLO:
  pipeline640  el frame entero reducido a 640 px de ancho: lo que hace hoy el pipeline (imgsz por defecto)
  full         el frame entero a resolución nativa (imgsz = ancho del vídeo)
  tiles1024    ventanas de 1024 px de ancho y alto completo a resolución nativa, con 256 px de solape (SAHI)
"detect" guarda las 10 detecciones más confiables de cada frame muestreado. "report" las evalúa crudas y
filtradas (solo dentro del campo + margen, y sin las que llevan paradas > 60 % de ±30 s: balones de reserva,
conos...), y mide:
  * por frame: ¿hay detección?, ¿la más confiable está a < R px del balón real? (R = 40 y 120), precisión
  * para la cámara virtual: con la pista que saldría de esas detecciones (top-1, o seguimiento por cercanía;
    interpolada + mismo suavizado que panorama_vcam.py), ¿qué fracción del tiempo cae el balón real dentro de
    la ventana de 796x448? También en ±2 s de cada tiro, que es lo que importa para T-DEED

Uso (dentro del contenedor, cwd /app):
    python panorama_ball_eval.py detect MATCH HALF [--stride 5] [--modes pipeline640 full tiles1024] [--max-frames N]
                                   [--weights PESOS.pt --tag _pano]
    python panorama_ball_eval.py report MATCH [MATCH...] [--modes tiles1024 tiles1024_pano]
Salida: output/panorama_ball/<match>_<half>_<modo><tag>.jsonl (reanudable) y el informe por stdout.
--weights cambia el detector (por defecto YOLO11x COCO); un modelo de una sola clase (el de balón en panorámica,
scripts/ball_detector/) se usa sin filtrar clases. --tag separa sus detecciones de las del detector COCO."""
import argparse, json, time
from pathlib import Path
import numpy as np

D = Path("datasets/soccertrack-v2")
OUT = Path("output/panorama_ball")
HALVES = ("1st", "2nd")
FPS = 25.0
CONF_MIN = 0.05                      # se guarda todo lo que supere esto; el barrido de umbral va en report
TILE_W, TILE_OVERLAP = 1024, 256
WIN = {"796x448": (796, 448), "1280x720": (1280, 720)}
RADII = (40, 120)                    # px: el balón mide ~8 px, un jugador ~46 px de alto
MAX_SPEED = 40.0                     # px/frame, como panorama_vcam.py
PITCH_MARGIN_M = 3.0                 # el balón en juego puede salir un poco del rectángulo (saques, bandas)
STATIC_WINDOW_S, STATIC_MIN_FRAC, STATIC_CELL = 30.0, 0.6, 16
PROX_MAX_JUMP_PX_S, PROX_MAX_GAP_S = 750.0, 2.0   # 30 m/s ≈ 500 px/s en el área; se pierde la pista tras 2 s


def calib(match):
    from scipy.interpolate import RBFInterpolator
    kp = json.load(open(D / "raw" / match / f"{match}_keypoints.json"))
    P = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp])
    I = np.array(list(kp.values()), dtype=float)
    return RBFInterpolator(P, I, kernel="thin_plate_spline", smoothing=1.0)


def gt_pixels(match, half):
    """Balón real en píxeles por frame de vídeo (fila i = frame i) y su status. Los partidos 1328xx no marcan nunca
    BALLOUT (status siempre != 0): con el balón fuera aparcan la pista exactamente en (0, 0) m, el punto central
    (~20 % de los frames). Ese (0, 0) exacto se trata como balón fuera; en el resto de partidos no aparece nunca."""
    z = np.load(D / "ball" / f"{match}_{half}_ball.npz")
    x, y = z["x"].astype(float), z["y"].astype(float)
    xy = np.column_stack([x + 52.5, y + 34.0])
    ok = np.isfinite(xy).all(1)
    px = np.full_like(xy, np.nan)
    px[ok] = calib(match)(xy[ok])
    status = z["status"].astype(int)
    status[(x == 0) & (y == 0)] = 0
    return px, status


def pitch_polygon(match):
    """Contorno del campo (+ margen) en píxeles, para descartar balones de reserva, conos, etc."""
    m = PITCH_MARGIN_M
    xs, ys = np.linspace(-m, 105 + m, 60), np.linspace(-m, 68 + m, 40)
    ring = np.concatenate([np.column_stack([xs, np.full_like(xs, -m)]), np.column_stack([np.full_like(ys, 105 + m), ys]),
                           np.column_stack([xs[::-1], np.full_like(xs, 68 + m)]), np.column_stack([np.full_like(ys, -m), ys[::-1]])])
    return calib(match)(ring).astype(np.float32)


def shot_frames(match, half):
    bas = json.load(open(D / "bas" / match / f"{match}_12_class_events.json"))["actions"]
    h = "1" if half == "1st" else "2"
    return [int(a["frame"]) - 1 for a in bas if a["gameTime"].startswith(f"{h} -") and a["label"] in ("SHOT", "GOAL")]


def video_path(match, half):
    return D / "videos" / match / f"{match}_panorama_{half}_half.mp4"


def out_path(match, half, mode):
    return OUT / f"{match}_{half}_{mode}.jsonl"


def read_rows(path):
    """Filas del JSONL; si la última quedó a medias (proceso matado), se descarta y se recorta el fichero."""
    rows, good = [], 0
    with open(path, "rb") as f:
        data = f.read()
    for line in data.splitlines(keepends=True):
        try:
            rows.append(json.loads(line)); good += len(line)
        except json.JSONDecodeError:
            break
    if good < len(data):
        with open(path, "wb") as f:
            f.write(data[:good])
    return rows


# --------------------------------------------------------------------------- detect

def tile_starts(width):
    n = 1 if width <= TILE_W else int(np.ceil((width - TILE_OVERLAP) / (TILE_W - TILE_OVERLAP)))
    return [int(round(s)) for s in np.linspace(0, width - TILE_W, n)] if n > 1 else [0]


def merge_close(dets, radius=20.0):
    """Quita duplicados de los solapes entre ventanas: dos detecciones a < radius px son la misma (gana la confianza)."""
    dets = sorted(dets, key=lambda d: -d[4])
    kept = []
    for d in dets:
        if all(np.hypot(d[0] - k[0], d[1] - k[1]) > radius for k in kept):
            kept.append(d)
    return kept


def run_yolo(model, images, imgsz, classes=(32,)):
    res = model(images, imgsz=imgsz, conf=CONF_MIN, classes=list(classes) if classes else None, half=True, verbose=False)
    out = []
    for r in res:
        b = r.boxes
        if b is None or len(b) == 0:
            out.append([]); continue
        xywh = b.xywh.cpu().numpy(); conf = b.conf.cpu().numpy()
        out.append([[float(x), float(y), float(w), float(h), float(c)] for (x, y, w, h), c in zip(xywh, conf)])
    return out


def detect(a):
    import cv2
    from ultralytics import YOLO
    model = YOLO(a.weights)
    classes = None if len(model.names) == 1 else (32,)   # COCO: 32 = "sports ball"
    cap = cv2.VideoCapture(str(video_path(a.match, a.half)))
    W, H, n = int(cap.get(3)), int(cap.get(4)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    OUT.mkdir(parents=True, exist_ok=True)
    imgsz_full = int(np.ceil(W / 32) * 32); imgsz_tile = int(np.ceil(H / 32) * 32)
    starts = tile_starts(W)
    print(f"{a.match} {a.half}: {W}x{H}, {n} frames, paso {a.stride}; full imgsz {imgsz_full}; "
          f"{len(starts)} ventanas de {TILE_W}x{H} (imgsz {imgsz_tile}) en x={starts}; pesos {a.weights}", flush=True)

    files, done = {}, {}
    for mode in a.modes:
        p = out_path(a.match, a.half, mode + a.tag)
        done[mode] = max([r["frame"] for r in read_rows(p)], default=-1) if p.exists() else -1
        files[mode] = open(p, "a")
    resume_from = min(done.values()) + 1
    if resume_from > 0:
        print(f"reanudando desde el frame {resume_from}", flush=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, resume_from)

    t0, n_done = time.perf_counter(), 0
    idx = resume_from
    while idx < n:
        if idx % a.stride:
            cap.grab(); idx += 1; continue
        ok, frame = cap.read()
        if not ok:
            break
        for mode in a.modes:
            if idx <= done[mode]:
                continue
            if mode == "pipeline640":
                dets = run_yolo(model, [frame], 640, classes)[0]
            elif mode == "full":
                dets = run_yolo(model, [frame], imgsz_full, classes)[0]
            else:
                tiles = [np.ascontiguousarray(frame[:, x0:x0 + TILE_W]) for x0 in starts]
                dets = merge_close([[d[0] + x0, d[1], d[2], d[3], d[4]] for x0, ds in zip(starts, run_yolo(model, tiles, imgsz_tile, classes)) for d in ds])
            dets = sorted(dets, key=lambda d: -d[4])[:10]
            files[mode].write(json.dumps({"frame": idx, "dets": [[round(v, 1) for v in d[:4]] + [round(d[4], 3)] for d in dets]}) + "\n")
        n_done += 1; idx += 1
        if n_done % 500 == 0:
            for f in files.values():
                f.flush()
            el = time.perf_counter() - t0
            print(f"  frame {idx}/{n}  {el / n_done:.2f} s/frame muestreado, ~{(n - idx) / a.stride * el / n_done / 60:.0f} min restantes", flush=True)
        if a.max_frames and n_done >= a.max_frames:
            break
    for f in files.values():
        f.close()
    print(f"FIN {a.match} {a.half}: {n_done} frames en {time.perf_counter() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------- report

def load_dets(match, half, mode):
    p = out_path(match, half, mode)
    if not p.exists():
        return None
    rows = read_rows(p)
    return np.array([r["frame"] for r in rows]), [r["dets"] for r in rows]


def static_mask(dets_per_frame, window):
    """True para las detecciones que llevan paradas (misma celda de 16 px) más de STATIC_MIN_FRAC de las ±window
    muestras: balones de reserva, conos, el soporte de la cámara. Un balón parado en una falta no llega."""
    return [[f >= STATIC_MIN_FRAC for f in fr] for fr in static_fraction(dets_per_frame, window)]


def static_fraction(dets_per_frame, window):
    """Por detección, fracción de las ±window muestras con alguna detección en su celda de 16 px (o vecinas)."""
    occ = {}
    for i, ds in enumerate(dets_per_frame):
        for x, y, *_ in ds:
            occ.setdefault((int(x // STATIC_CELL), int(y // STATIC_CELL)), set()).add(i)
    occ = {k: np.array(sorted(v)) for k, v in occ.items()}
    out = []
    for i, ds in enumerate(dets_per_frame):
        flags = []
        for x, y, *_ in ds:
            cx, cy = int(x // STATIC_CELL), int(y // STATIC_CELL)
            seen = set()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    arr = occ.get((cx + dx, cy + dy))
                    if arr is not None:
                        lo, hi = np.searchsorted(arr, i - window), np.searchsorted(arr, i + window, side="right")
                        seen.update(arr[lo:hi].tolist())
            flags.append(len(seen) / (2 * window + 1))
        out.append(flags)
    return out


def filter_dets(dets_per_frame, poly, stride, with_static=False):
    """Quita lo de fuera del campo y lo estático. with_static=True añade a cada detección que queda su fracción de
    estático (6º valor), para penalizar lo que aparece a ratos en el mismo sitio sin llegar al umbral."""
    import cv2
    window = int(round(STATIC_WINDOW_S * FPS / stride))
    static = static_fraction(dets_per_frame, window)
    out = []
    for ds, st in zip(dets_per_frame, static):
        out.append([list(d[:5]) + ([s] if with_static else []) for d, s in zip(ds, st)
                    if s < STATIC_MIN_FRAC and cv2.pointPolygonTest(poly, (float(d[0]), float(d[1])), False) >= 0])
    return out


def top1(dets_per_frame, th):
    sel = [(i, max(ds, key=lambda d: d[4])) for i, ds in enumerate(dets_per_frame) if any(d[4] >= th for d in ds)]
    sel = [(i, d) for i, d in sel if d[4] >= th]
    return np.array([i for i, _ in sel], int), np.array([[d[0], d[1]] for _, d in sel]).reshape(-1, 2)


def proximity_track(dets_per_frame, th, stride):
    """Por frame, la detección más cercana a la última posición (si el salto es plausible); si no hay pista
    reciente, la más confiable."""
    max_jump = PROX_MAX_JUMP_PX_S * stride / FPS
    max_gap = PROX_MAX_GAP_S * FPS / stride
    sel_i, sel_xy, last, last_i = [], [], None, None
    for i, ds in enumerate(dets_per_frame):
        c = [d for d in ds if d[4] >= th]
        if not c:
            continue
        if last is not None and i - last_i <= max_gap:
            d = min(c, key=lambda d: np.hypot(d[0] - last[0], d[1] - last[1]))
            if np.hypot(d[0] - last[0], d[1] - last[1]) > max_jump * (i - last_i):
                d = max(c, key=lambda d: d[4])
        else:
            d = max(c, key=lambda d: d[4])
        last, last_i = (d[0], d[1]), i
        sel_i.append(i); sel_xy.append(last)
    return np.array(sel_i, int), np.array(sel_xy).reshape(-1, 2)


KALMAN = dict(gate_px=120.0, gate_grow_px=60.0, alpha=0.6, beta=0.3, coast_decay=0.85, reinit_th=0.3, reinit_px=100.0)


def kalman_track(dets_per_frame, th, stride):
    """Seguimiento alfa-beta (posición + velocidad) sobre las detecciones filtradas, con PUERTA: solo se acepta la
    detección más cercana a la predicción si está a menos de gate_px (+ gate_grow_px por muestra sin detección).
    Sin detección aceptada la pista avanza por inercia (velocidad que decae), así la cámara no salta a un falso
    positivo lejano justo tras un golpeo. Perdida la pista > PROX_MAX_GAP_S, se reinicia con una detección >=
    reinit_th que se repita en dos muestras seguidas (a < reinit_px). Devuelve TODAS las muestras con pista
    (aceptadas o por inercia) y cuáles tuvieron detección."""
    dt = stride / FPS
    max_gap = PROX_MAX_GAP_S * FPS / stride
    out_i, out_xy, out_has = [], [], []
    pos = vel = None; miss = 0; prev_cand = None
    for i, ds in enumerate(dets_per_frame):
        c = [d for d in ds if d[4] >= th]
        if pos is not None:
            pred = pos + vel * dt
            best = min(c, key=lambda d: np.hypot(d[0] - pred[0], d[1] - pred[1]), default=None)
            if best is not None and np.hypot(best[0] - pred[0], best[1] - pred[1]) <= KALMAN["gate_px"] + KALMAN["gate_grow_px"] * miss:
                z = np.array(best[:2]); r = z - pred
                pos = pred + KALMAN["alpha"] * r; vel = vel + KALMAN["beta"] * r / dt; miss = 0
                out_i.append(i); out_xy.append(pos.copy()); out_has.append(True)
                continue
            miss += 1
            if miss <= max_gap:
                pos = pred; vel = vel * KALMAN["coast_decay"]
                out_i.append(i); out_xy.append(pos.copy()); out_has.append(False)
                continue
            pos = vel = None   # pista perdida
        # (re)inicio: una detección confiable que se repita en la muestra siguiente
        strong = [d for d in c if d[4] >= KALMAN["reinit_th"]]
        if strong:
            top = max(strong, key=lambda d: d[4])
            if prev_cand is not None and np.hypot(top[0] - prev_cand[0], top[1] - prev_cand[1]) <= KALMAN["reinit_px"]:
                pos = np.array(top[:2]); vel = (pos - np.array(prev_cand[:2])) / dt; miss = 0
                out_i.append(i); out_xy.append(pos.copy()); out_has.append(True)
            prev_cand = top
        else:
            prev_cand = None
    return np.array(out_i, int), np.array(out_xy).reshape(-1, 2), np.array(out_has, bool)


def smooth_track(frames, xy, n_frames, stride):
    """Pista de cámara como panorama_vcam.py: interpolación lineal de los huecos, media móvil de 1 s y velocidad
    máxima. Devuelve el centro de cámara en cada frame muestreado (múltiplos de stride)."""
    idx = np.arange(0, n_frames, stride)
    if len(frames) == 0:
        return idx, None
    c = np.column_stack([np.interp(idx, frames, xy[:, 0]), np.interp(idx, frames, xy[:, 1])])
    k = max(1, int(round(FPS / stride)))
    pad = (k // 2, k - 1 - k // 2)
    for j in range(2):
        c[:, j] = np.convolve(np.pad(c[:, j], pad, mode="edge"), np.ones(k) / k, mode="valid")
    lim = MAX_SPEED * stride
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        c[i] = c[i - 1] + np.clip(d, -lim, lim)
    return idx, c


def in_window(gt, cam, win):
    w, h = win
    return (np.abs(gt[:, 0] - cam[:, 0]) <= w / 2) & (np.abs(gt[:, 1] - cam[:, 1]) <= h / 2)


def frame_table(rows, title):
    """rows: lista de (ok, near, g, dets) por mitad. Tabla por umbral: cobertura, top-1 y precisión."""
    ok = np.concatenate([r[0] for r in rows]); near = np.concatenate([r[1] for r in rows])
    g = np.concatenate([r[2] for r in rows]); dets = [d for r in rows for d in r[3]]
    print(f"  {title}: {ok.sum()} frames con balón en juego, {near[ok].sum()} en ±2 s de un tiro")
    print(f"  {'umbral':>7}{'frames con det.':>17}" + "".join(f"{'top-1 <' + str(r) + 'px':>14}" for r in RADII)
          + "".join(f"{'alguna <' + str(r):>13}" for r in RADII) + "".join(f"{'precisión@' + str(r):>14}" for r in RADII)
          + "   | ±2 s de tiro: top-1 <120px")
    for th in (0.05, 0.1, 0.2, 0.3, 0.5):
        has = np.zeros(len(dets), bool); d_top = np.full(len(dets), np.inf); d_min = np.full(len(dets), np.inf)
        tp = {r: 0 for r in RADII}; n_det = 0
        for i, ds in enumerate(dets):
            c = [d for d in ds if d[4] >= th]
            if not c or not ok[i]:
                continue
            has[i] = True
            t = max(c, key=lambda d: d[4])
            d_top[i] = np.hypot(t[0] - g[i, 0], t[1] - g[i, 1])
            dist = [np.hypot(d[0] - g[i, 0], d[1] - g[i, 1]) for d in c]
            d_min[i] = min(dist); n_det += len(dist)
            for r in RADII:
                tp[r] += sum(x <= r for x in dist)
        cells = [f"{100 * has[ok].mean():16.1f}%"] + [f"{100 * (has & (d_top <= r))[ok].mean():13.1f}%" for r in RADII]
        cells += [f"{100 * (d_min <= r)[ok].mean():12.1f}%" for r in RADII] + [f"{100 * tp[r] / max(n_det, 1):13.1f}%" for r in RADII]
        sh = (has & (d_top <= 120))[ok & near].mean() if (ok & near).any() else float("nan")
        print(f"  {th:>7.2f}" + "".join(cells) + f"   | {100 * sh:5.1f}%")


def camera_table(tracks, title):
    """tracks: {etiqueta: [(idx, cam, gt_idx, status_idx, near_idx, sel_frames), ...]}."""
    print(f"\n  {title}: ¿cae el balón real en la ventana de la cámara virtual? (pista interpolada + suavizado de 1 s)")
    print(f"  {'pista':<40}" + "".join(f"{w + ' total':>15}{w + ' ±2s tiro':>18}" for w in WIN) + "   sin detección > 2 s")
    for label, halves in tracks.items():
        cells = []
        for w in WIN:
            tot = tot_n = sh = sh_n = 0
            for idx, cam, g, st, nr, _ in halves:
                if cam is None:
                    continue
                okg = np.isfinite(g).all(1) & (st != 0)
                inw = in_window(g[okg], cam[okg], WIN[w])
                tot += inw.sum(); tot_n += okg.sum(); sh += inw[nr[okg]].sum(); sh_n += nr[okg].sum()
            cells.append(f"{100 * tot / max(tot_n, 1):14.1f}%{100 * sh / max(sh_n, 1):17.1f}%")
        gap = gap_n = 0
        for idx, cam, g, st, nr, sel in halves:
            okg = np.isfinite(g).all(1) & (st != 0)
            if len(sel) == 0:
                gap += okg.sum(); gap_n += okg.sum(); continue
            pos = np.searchsorted(sel, idx[okg])
            near_prev = np.abs(idx[okg] - sel[np.clip(pos - 1, 0, len(sel) - 1)])
            near_next = np.abs(sel[np.clip(pos, 0, len(sel) - 1)] - idx[okg])
            gap += (np.minimum(near_prev, near_next) > 2 * FPS).sum(); gap_n += okg.sum()
        print(f"  {label:<40}" + "".join(cells) + f"{100 * gap / max(gap_n, 1):20.1f}%")


def report(a):
    print(f"Partidos: {' '.join(a.matches)} | GT: pista del balón del dataset proyectada a píxeles (posición en el suelo)")
    print(f"Solo frames con balón en juego. Tolerancias: R=40 px (el propio balón) y R=120 px (vale para encuadrar).")
    print(f"Filtrado = dentro del campo (+{PITCH_MARGIN_M:g} m) y no estático (> {STATIC_MIN_FRAC:.0%} de ±{STATIC_WINDOW_S:g} s en la misma celda)\n")
    for mode in a.modes:
        raw_rows, filt_rows, tracks = [], [], {}
        for m in a.matches:
            poly = pitch_polygon(m)
            for h in HALVES:
                loaded = load_dets(m, h, mode)
                if loaded is None:
                    continue
                frames, dets = loaded
                stride = int(np.min(np.diff(frames))) if len(frames) > 1 else 5
                gt, status = gt_pixels(m, h)
                shots = shot_frames(m, h)
                g = gt[frames]
                ok = np.isfinite(g).all(1) & (status[frames] != 0)
                near = np.zeros(len(frames), bool)
                for s in shots:
                    near |= np.abs(frames - s) <= 2 * FPS
                filt = filter_dets(dets, poly, stride)
                raw_rows.append((ok, near, g, dets)); filt_rows.append((ok, near, g, filt))
                n_frames = int(frames.max()) + 1
                idx = np.arange(0, n_frames, stride)
                near_idx = np.zeros(len(idx), bool)
                for s in shots:
                    near_idx |= np.abs(idx - s) <= 2 * FPS
                variants = {"balón del dataset (techo)": (frames[ok], g[ok])}
                for th in (0.1, 0.3):
                    i_sel, xy = top1(dets, th); variants[f"YOLO top-1 crudo, θ≥{th:g}"] = (frames[i_sel], xy)
                    i_sel, xy = top1(filt, th); variants[f"YOLO top-1 filtrado, θ≥{th:g}"] = (frames[i_sel], xy)
                    i_sel, xy = proximity_track(filt, th, stride); variants[f"YOLO filtrado + cercanía, θ≥{th:g}"] = (frames[i_sel], xy)
                for label, (sel_frames, xy) in variants.items():
                    _, cam = smooth_track(sel_frames, xy, n_frames, stride)
                    tracks.setdefault(label, []).append((idx, cam, gt[idx], status[idx], near_idx, np.sort(sel_frames)))
        if not raw_rows:
            print(f"=== {mode}: sin detecciones guardadas ===\n"); continue
        print(f"=== {mode} ({len(raw_rows)} mitades) ===")
        frame_table(raw_rows, "crudo")
        if mode != "pipeline640":
            frame_table(filt_rows, "filtrado")
        camera_table(tracks, "cámara virtual")
        print()


def track(a):
    """Pista de balón en píxeles por frame de vídeo a partir de las detecciones guardadas (filtradas + seguimiento
    por cercanía, huecos interpolados), para mover la cámara virtual sin el balón del dataset:
    panorama_vcam.py --camera <npz>. El suavizado y el límite de velocidad los aplica panorama_vcam.py."""
    frames, dets = load_dets(a.match, a.half, a.mode)
    stride = int(np.min(np.diff(frames))) if len(frames) > 1 else 5
    filt = filter_dets(dets, pitch_polygon(a.match), stride)
    if a.method == "kalman":
        i_sel, xy, has_i = kalman_track(filt, a.th, stride)
    else:
        i_sel, xy = proximity_track(filt, a.th, stride) if a.method == "prox" else top1(filt, a.th)
        has_i = np.ones(len(i_sel), bool)
    n = int(frames.max()) + 1
    all_frames = np.arange(n)
    x = np.interp(all_frames, frames[i_sel], xy[:, 0]); y = np.interp(all_frames, frames[i_sel], xy[:, 1])
    has = np.zeros(n, bool); has[frames[i_sel[has_i]]] = True
    out = OUT / f"{a.match}_{a.half}_{a.mode}_cam{a.tag}.npz"
    np.savez(out, frame=all_frames, x=x, y=y, has_det=has, stride=stride, th=a.th, method=a.method)
    print(f"{out}: {int(has_i.sum())} detecciones usadas en {len(frames)} frames muestreados ({100 * has_i.sum() / len(frames):.0f}%), "
          f"hueco máximo {np.max(np.diff(frames[i_sel[has_i]])) / FPS if has_i.sum() > 1 else float('inf'):.1f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect"); d.add_argument("match"); d.add_argument("half", choices=HALVES)
    d.add_argument("--stride", type=int, default=5); d.add_argument("--modes", nargs="+", default=["pipeline640", "full", "tiles1024"])
    d.add_argument("--max-frames", type=int, default=None)
    d.add_argument("--weights", default="/app/models/yolo11x.pt"); d.add_argument("--tag", default="", help="sufijo del modo en los JSONL")
    r = sub.add_parser("report"); r.add_argument("matches", nargs="+")
    r.add_argument("--modes", nargs="+", default=["pipeline640", "full", "tiles1024"])
    t = sub.add_parser("track"); t.add_argument("match"); t.add_argument("half", choices=HALVES); t.add_argument("mode")
    t.add_argument("--th", type=float, default=0.1); t.add_argument("--method", choices=["prox", "top1", "kalman"], default="prox")
    t.add_argument("--tag", default="", help="sufijo del npz de salida, para distinguir variantes")
    a = ap.parse_args()
    {"detect": detect, "report": report, "track": track}[a.cmd](a)
