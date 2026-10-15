"""Trayectoria de la cámara virtual sobre la panorámica a partir de las detecciones de balón guardadas
(panorama_ball_eval.py detect), en dos pasos, ambos offline (se conoce el partido entero):

  1. Pista (Viterbi): en vez de la detección más confiable de cada muestra, la secuencia de detecciones más
     plausible de toda la mitad. Coste = -log(confianza) + penalización por "estático a ratos" + coste de
     movimiento (cuadrático por encima de una velocidad libre, imposible por encima de VMAX) + coste por muestra
     sin balón; saltar a otro sitio sin trayectoria física cuesta RESTART. Un falso positivo aislado lejos del
     balón ya no arrastra la cámara.
  2. Cámara (L1 óptima, Grundmann et al. 2011): minimiza |velocidad| + |aceleración| + |tirón| de la cámara con el
     balón dentro de una zona central (penalización lineal si sale, menor cuanto menos se sabe del balón). Da
     tramos quietos, paneos a velocidad constante y transiciones suaves, como un operador; la zona muerta sale sola.
     --lead N desplaza el objetivo hacia la portería con el balón en el último tercio (margen para el vuelo del tiro).

Variantes de cámara (todas por frame, en píxeles de la panorámica):
  dataset     balón del dataset + media móvil 1 s + tope 40 px/frame (= la vista vcam796 ya escaneada)
  actual      top-1 filtrado θ≥0,3 + media móvil 1 s + tope (= la vista vcam796yolo_pano)
  viterbi     pista Viterbi + media móvil 1 s + tope
  viterbi_l1  pista Viterbi + cámara L1
  dataset_l1  balón del dataset + cámara L1 (control: ¿qué hace la dinámica L1 sola?)

Uso (dentro del contenedor, cwd /app):
    python vcam_path.py metrics MATCH [MATCH...] [--variants actual viterbi viterbi_l1 dataset] [--mode tiles1024_pano]
    python vcam_path.py path MATCH HALF --variant viterbi_l1 --tag _vl1
El npz de "path" es la trayectoria FINAL; se renderiza sin volver a suavizar:
    panorama_vcam.py MATCH HALF 0 -1 --size 796x448 --camera NPZ --smooth-s 0 --max-speed 1e9 --out ..."""
import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import panorama_ball_eval as E

WIN_W, WIN_H = 796, 448
OUT = Path("output/panorama_ball")


@dataclass
class TrackParams:
    cmin: float = 0.05       # confianza mínima de un candidato
    static_w: float = 3.0    # coste extra por fracción de "estático" (0-0,6; por encima ya se filtra)
    miss: float = 1.5        # coste por muestra sin balón seleccionado (≈ -log 0,22)
    restart: float = 4.0     # coste de empezar una pista nueva (saltar sin trayectoria física)
    gap: int = 10            # muestras máximas sin balón dentro de una misma pista (10 × 0,2 s = 2 s)
    vfree: float = 10.0      # px/frame de movimiento sin coste (p90 del balón real: ~9)
    vscale: float = 10.0     # coste ((v - vfree) / vscale)^2
    vmax: float = 80.0       # px/frame imposibles (el balón detectado más rápido: ~32)


@dataclass
class CamParams:
    rx: float = 120.0        # semiancho de la zona central en la que el balón puede moverse sin mover la cámara
    ry: float = 60.0
    w1: float = 1.0          # pesos de |velocidad|, |aceleración|, |tirón| (por muestra de 0,2 s)
    w2: float = 4.0
    w3: float = 10.0
    lam_det: float = 1.0     # coste por px fuera de zona: muestra con detección / hueco corto / pista perdida
    lam_gap: float = 0.5
    lam_lost: float = 0.05
    lead: float = 0.0        # px de margen hacia la portería con el balón en el último tercio (rampa hasta la línea de gol)


def candidates(match, half, mode):
    frames, dets = E.load_dets(match, half, mode)
    stride = int(np.min(np.diff(frames)))
    return frames, E.filter_dets(dets, E.pitch_polygon(match), stride, with_static=True), stride


def viterbi_track(dets, stride, P=TrackParams()):
    """Camino de coste mínimo sobre las detecciones (DAG con saltos de hasta P.gap muestras). Devuelve por muestra
    la posición elegida (NaN si ninguna) y su confianza."""
    n = len(dets)
    start = np.zeros(n + 1, int); xy, e = [], []
    for i, ds in enumerate(dets):
        start[i] = len(xy)
        for d in ds:
            if d[4] >= P.cmin:
                xy.append(d[:2]); e.append(-np.log(d[4]) + P.static_w * d[5])
    start[n] = len(xy)
    xy = np.array(xy, float).reshape(-1, 2); e = np.array(e, float)
    node_i = np.repeat(np.arange(n), np.diff(start))
    cost = np.full(len(xy), np.inf); pred = np.full(len(xy), -1)
    Pc = np.zeros(n + 1); Pfrom = np.full(n + 1, -2)   # mejor camino hasta la muestra i-1; -2 = esa muestra sin balón
    for i in range(n):
        a, b = start[i], start[i + 1]
        if b > a:
            best = Pc[i] + P.restart + e[a:b]; arg = np.full(b - a, -1)
            for g in range(1, P.gap + 1):
                j = i - g
                if j < 0:
                    break
                ja, jb = start[j], start[j + 1]
                if jb == ja:
                    continue
                d = np.hypot(*(xy[ja:jb, None, :] - xy[None, a:b, :]).transpose(2, 0, 1))
                v = d / (stride * g)
                t = np.where(v <= P.vmax, (np.maximum(0.0, v - P.vfree) / P.vscale) ** 2, np.inf)
                tot = cost[ja:jb, None] + P.miss * (g - 1) + t + e[None, a:b]
                k = tot.argmin(0); val = tot[k, np.arange(b - a)]
                upd = val < best; best[upd] = val[upd]; arg[upd] = ja + k[upd]
            cost[a:b] = best; pred[a:b] = arg
        Pc[i + 1], Pfrom[i + 1] = Pc[i] + P.miss, -2
        if b > a:
            k = a + int(cost[a:b].argmin())
            if cost[k] < Pc[i + 1]:
                Pc[i + 1], Pfrom[i + 1] = cost[k], k
    sel = np.full(n, -1); i = n
    while i > 0:
        node = Pfrom[i]
        if node == -2:
            i -= 1; continue
        while True:
            sel[node_i[node]] = node
            if pred[node] == -1:
                i = node_i[node]; break
            node = pred[node]
    out = np.full((n, 2), np.nan); has = sel >= 0
    out[has] = xy[sel[has]]
    return out, has


def track_from_top1(dets, th=0.3):
    """La pista de la cámara actual (panorama_ball_eval.py track --method top1 sobre las filtradas)."""
    out = np.full((len(dets), 2), np.nan)
    for i, ds in enumerate(dets):
        c = [d for d in ds if d[4] >= th]
        if c:
            out[i] = max(c, key=lambda d: d[4])[:2]
    return out, np.isfinite(out[:, 0])


def per_frame(frames, xy, has, n_frames):
    f = np.arange(n_frames)
    return np.column_stack([np.interp(f, frames[has], xy[has, j]) for j in range(2)])


def ma_clamp(px, fps=E.FPS, smooth_s=1.0, max_speed=40.0):
    """Lo mismo que panorama_vcam.py: media móvil centrada y tope de velocidad por eje."""
    k = max(1, int(round(smooth_s * fps)))
    sm = lambda v: np.convolve(np.pad(v, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    c = np.column_stack([sm(px[:, 0]), sm(px[:, 1])])
    for i in range(1, len(c)):   # secuencial: el tope se aplica respecto a la posición ya limitada
        c[i] = c[i - 1] + np.clip(c[i] - c[i - 1], -max_speed, max_speed)
    return c


def l1_axis(b, lam, r, lo, hi, C):
    """min w1|D1 p| + w2|D2 p| + w3|D3 p| + Σ lam_i u_i  s.a. |p_i - b_i| <= r + u_i, lo <= p <= hi."""
    n = len(b)
    D = [sparse.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n)), sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)),
         sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(n - 3, n))]
    m = [n - 1, n - 2, n - 3]
    nv = n + sum(m) + n          # variables: p, |D1 p|, |D2 p|, |D3 p|, salida de zona u

    def at(block, col):          # bloque colocado en la columna col de una fila de nv columnas
        block = sparse.coo_matrix(block)
        return sparse.csr_matrix((block.data, (block.row, block.col + col)), shape=(block.shape[0], nv))

    rows, rhs, off = [], [], n
    for Dk, mk in zip(D, m):     # ±Dk p - ak <= 0
        for sgn in (1, -1):
            rows.append(at(sgn * Dk, 0) + at(-sparse.eye(mk), off)); rhs.append(np.zeros(mk))
        off += mk
    I = sparse.eye(n)             # ±(p - b) - u <= r
    rows.append(at(I, 0) + at(-I, off)); rhs.append(b + r)
    rows.append(at(-I, 0) + at(-I, off)); rhs.append(-b + r)
    A = sparse.vstack(rows).tocsr(); rhs = np.concatenate(rhs)
    c = np.concatenate([np.zeros(n), np.full(m[0], C.w1), np.full(m[1], C.w2), np.full(m[2], C.w3), lam])
    bounds = [(lo, hi)] * n + [(0, None)] * (nv - n)
    res = linprog(c, A_ub=A, b_ub=rhs, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"L1: {res.message}")
    return res.x[:n]


def goal_lead(x, match, lead):
    """Desplazamiento horizontal del objetivo hacia la portería más cercana: 0 hasta el último tercio, `lead` px en
    la línea de gol (para que el vuelo de un tiro quede dentro de la ventana)."""
    if not lead:
        return np.zeros_like(x)
    gl, tl, tr, gr = np.sort(E.calib(match)(np.array([[0, 34], [35, 34], [70, 34], [105, 34]], float))[:, 0])
    return lead * (np.clip((x - tr) / (gr - tr), 0, 1) - np.clip((tl - x) / (tl - gl), 0, 1))


def l1_camera(frames, xy, has, n_frames, frame_w, frame_h, stride, gap, C=CamParams(), match=None):
    """Cámara L1 sobre la rejilla de muestras (0,2 s) e interpolada a cada frame."""
    tgt = np.column_stack([np.interp(np.arange(len(frames)), np.where(has)[0], xy[has, j]) for j in range(2)])
    tgt[:, 0] += goal_lead(tgt[:, 0], match, C.lead)
    idx = np.where(has)[0]
    pos = np.searchsorted(idx, np.arange(len(frames)))
    dist = np.minimum(np.abs(np.arange(len(frames)) - idx[np.clip(pos - 1, 0, len(idx) - 1)]),
                      np.abs(idx[np.clip(pos, 0, len(idx) - 1)] - np.arange(len(frames))))
    lam = np.where(has, C.lam_det, np.where(dist <= gap, C.lam_gap, C.lam_lost))
    cx = l1_axis(tgt[:, 0], lam, C.rx, WIN_W / 2, frame_w - WIN_W / 2, C)
    cy = l1_axis(tgt[:, 1], lam, C.ry, WIN_H / 2, frame_h - WIN_H / 2, C)
    f = np.arange(n_frames)
    return np.column_stack([np.interp(f, frames, cx), np.interp(f, frames, cy)])


def frame_size(match, half):
    import cv2
    cap = cv2.VideoCapture(str(E.video_path(match, half)))
    w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
    return w, h


def camera(variant, match, half, mode, TP=TrackParams(), C=CamParams()):
    """Centro de la cámara por frame (n_frames x 2) para una variante. Las del dataset no necesitan detecciones."""
    gt, _ = E.gt_pixels(match, half)
    n_frames = len(gt)
    if variant.startswith("dataset"):
        ok = np.isfinite(gt).all(1)
        if variant == "dataset":
            return ma_clamp(per_frame(np.arange(n_frames), gt, ok, n_frames))
        stride = 5; frames = np.arange(0, n_frames, stride)
        g = gt[frames]; okf = np.isfinite(g).all(1)
        return l1_camera(frames, g, okf, n_frames, *frame_size(match, half), stride, TP.gap, C, match)
    frames, dets, stride = candidates(match, half, mode)
    xy, has =track_from_top1(dets) if variant == "actual" else viterbi_track(dets, stride, TP)
    if variant.endswith("_l1"):
        return l1_camera(frames, xy, has, n_frames, *frame_size(match, half), stride, TP.gap, C, match)
    return ma_clamp(per_frame(frames, xy, has, n_frames))


# --------------------------------------------------------------------------- indicadores

def cam_metrics(cam, gt, status, shots, frame_w, frame_h):
    """Lo que ve el render (centro recortado a los bordes de la panorámica) frente al balón del dataset."""
    c = np.column_stack([np.clip(cam[:, 0], WIN_W / 2, frame_w - WIN_W / 2), np.clip(cam[:, 1], WIN_H / 2, frame_h - WIN_H / 2)])
    n = min(len(c), len(gt)); c, g, st = c[:n], gt[:n], status[:n]
    ok = np.isfinite(g).all(1) & (st != 0)
    inw = (np.abs(g[:, 0] - c[:, 0]) <= WIN_W / 2) & (np.abs(g[:, 1] - c[:, 1]) <= WIN_H / 2)
    around = np.zeros(n, bool); after = np.zeros(n, bool)
    for s in shots:
        around[max(s - 50, 0):s + 51] = True; after[s:s + 51] = True
    v = np.hypot(*np.diff(c, axis=0).T); acc = np.hypot(*np.diff(c, 2, axis=0).T)
    onsets = np.sum((v[1:] > 25) & (v[:-1] <= 25))
    return dict(n_ok=ok.sum(), inw=(inw & ok).sum(), n_around=(ok & around).sum(), inw_around=(inw & ok & around).sum(),
                n_after=(ok & after).sum(), inw_after=(inw & ok & after).sum(), v=v, acc=acc, onsets=onsets, minutes=n / E.FPS / 60)


def metrics(a):
    TP, C = TrackParams(**params(a, TrackParams)), CamParams(**params(a, CamParams))
    print(f"Partidos {' '.join(a.matches)} | modo {a.mode} | {TP} | {C}")
    print(f"{'cámara':<12}{'balón en ventana':>17}{'±2 s tiro':>11}{'+0..2 s tiro':>14}{'vel p50':>9}{'p95':>6}{'p99':>6}"
          f"{'acel p99':>10}{'quieta':>8}{'barridos/10min':>16}")
    for variant in a.variants:
        agg = None
        for m in a.matches:
            for h in E.HALVES:
                cam = camera(variant, m, h, a.mode, TP, C)
                gt, st = E.gt_pixels(m, h)
                r = cam_metrics(cam, gt, st, E.shot_frames(m, h), *frame_size(m, h))
                if agg is None:
                    agg = r
                else:
                    for k in r:
                        agg[k] = np.concatenate([agg[k], r[k]]) if isinstance(r[k], np.ndarray) else agg[k] + r[k]
        v, acc = agg["v"], agg["acc"]
        print(f"{variant:<12}{100 * agg['inw'] / agg['n_ok']:16.1f}%{100 * agg['inw_around'] / max(agg['n_around'], 1):10.1f}%"
              f"{100 * agg['inw_after'] / max(agg['n_after'], 1):13.1f}%{np.median(v):9.1f}{np.percentile(v, 95):6.1f}"
              f"{np.percentile(v, 99):6.1f}{np.percentile(acc, 99):10.2f}{100 * (v < 0.5).mean():7.0f}%"
              f"{10 * agg['onsets'] / agg['minutes']:16.1f}", flush=True)


def path(a):
    TP, C = TrackParams(**params(a, TrackParams)), CamParams(**params(a, CamParams))
    cam = camera(a.variant, a.match, a.half, a.mode, TP, C)
    out = OUT / f"{a.match}_{a.half}_vcam{a.tag}.npz"
    np.savez(out, frame=np.arange(len(cam)), x=cam[:, 0], y=cam[:, 1], variant=a.variant, track=str(TP), cam=str(C))
    print(f"{out}: {len(cam)} frames ({a.variant})", flush=True)


def params(a, cls):
    return {f.name: getattr(a, f.name) for f in fields(cls) if getattr(a, f.name, None) is not None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mt = sub.add_parser("metrics"); mt.add_argument("matches", nargs="+")
    mt.add_argument("--variants", nargs="+", default=["dataset", "actual", "viterbi", "viterbi_l1", "dataset_l1"])
    pt = sub.add_parser("path"); pt.add_argument("match"); pt.add_argument("half", choices=E.HALVES)
    pt.add_argument("--variant", default="viterbi_l1"); pt.add_argument("--tag", required=True)
    for p in (mt, pt):
        p.add_argument("--mode", default="tiles1024_pano")
        for f in fields(TrackParams) + fields(CamParams):
            p.add_argument(f"--{f.name}", type=type(f.default), default=None)
    a = ap.parse_args()
    {"metrics": metrics, "path": path}[a.cmd](a)
