"""RoI «perfecta» para AdaSpot: sustituye las regiones del selector no supervisado por cajas centradas en
coordenadas anotadas, con los pesos congelados (docs/comparativa_benchmark/7.14_adaspot_roi_gt.md).

AdaSpot elige una región por frame con un selector SIN entrenar (``ROISelector`` en
``models/adaspot/repo/model/modules.py``: saliencia = media por canales del último mapa de la rama de baja
resolución, suavizado espacio-temporal y máximo) y la recorta de la imagen de alta resolución para su segunda rama.
En SN-BAS el config usa ``threshold = 0``: el recorte es siempre de 112x112 px sobre 796x448. Aquí se cambia solo
DÓNDE cae ese recorte, para medir (a) cuántas veces el selector deja fuera la acción y (b) cuánto sube el modelo si
la región es la correcta. Es el techo del SELECTOR, no del modelo: no se reentrena nada.

No se edita el repo de AdaSpot (``models/`` está fuera de git): el modo se inyecta cambiando
``model._model.roi_selector`` por ``RoIOverride`` y envolviendo su ``ActionSpotInferenceDataset`` con ``with_rois``.

Fuentes del centro de la RoI (por frame de vídeo, (y, x) normalizado a [0, 1]; NaN = la fuente no sabe y se usa la
saliencia en ese frame):
  st2     SoccerTrack v2: caja (game state) del jugador que ejecuta cada evento, interpolada entre ejecutores
          consecutivos y llevada a la ventana de la cámara virtual. NO se usa la pista de balón del dataset: está
          anclada en la posición de evento del proveedor y cae a más de 56 px del ejecutor en el 31 % de los eventos.
  pcbas   FOOTPASS no trae balón: misma construcción con las cajas de sus jugadores. En los dos, sin dato si el
          jugador no se ve o a más de ``HOLD_S`` de un evento.
  det     pseudo-GT, NO es anotación: balón de nuestro detector (``<vídeo>.tracks.jsonl``). Para SN-BAS, que no trae
          coordenadas de nada (solo instante, clase y equipo).
  center  control: recorte fijo en el centro de la imagen.

Dos variantes de sensibilidad sobre la misma anotación (modos ``gtfeet`` y ``fix``):
  gtfeet  FOOTPASS: centrada en los pies del ejecutor (donde está el balón) en vez de en el centro de su caja. En
          SoccerTrack v2 la caja del game state ya está a la altura de los pies.
  fix     saliencia corregida: el recorte que elige el modelo, desplazado lo mínimo para que el ejecutor quede a
          ``FIX_PX`` o menos de su centro. Cura la ceguera cambiando lo menos posible lo que ve la rama de alta resolución.
"""

from __future__ import annotations

import json
import mmap
import re
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ST2 = ROOT / "datasets" / "soccertrack-v2"
PCBAS = ROOT / "datasets" / "SoccerNet" / "pcbas-2026"
VCAM_PATHS = ROOT / "output" / "panorama_ball"
PLAYERS_CACHE = ROOT / "output" / "benchmark" / "adaspot_gt_roi" / "st2_players"

ROI = 112            # lado del recorte con el que se entrenó la rama de alta resolución (roi_size del config)
HOLD_S = 3.0         # lejos de cualquier evento no se sabe dónde está el balón
SMOOTH = 9           # frames de la media móvil de la pista (0,36 s): quita el temblor de las cajas
FIX_PX = ROI // 4    # modo fix: distancia máxima del ejecutor al centro del recorte (queda entero dentro)
TAGS = {"saliency": "", "gt": "_gtroi", "center": "_ctrroi", "det": "_detroi", "gtfeet": "_gtfeetroi", "fix": "_fixroi",
        "salcheck": "_salroi"}  # salcheck = saliency con otro nombre de caché: mide el ruido de re-puntuar (tamaño de lote)
TRACKED = ("gt", "det", "gtfeet", "fix")  # modos que necesitan la pista de coordenadas del vídeo


def _smooth(track: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    """Media móvil que ignora los NaN y no rellena huecos."""
    v = np.isfinite(track).all(1)
    w = np.convolve(v.astype(float), np.ones(k), "same")
    out = np.full_like(track, np.nan)
    for j in range(track.shape[1]):
        out[v, j] = np.convolve(np.where(v, track[:, j], 0.0), np.ones(k), "same")[v] / w[v]
    return out


def st2_window(video: Path, n_frames: int, size: tuple[int, int] = (796, 448)) -> tuple[np.ndarray, np.ndarray]:
    """Esquina (x0, y0) de la ventana de la cámara virtual en cada frame, en píxeles de la panorámica, reconstruida
    igual que ``scripts/shot_benchmark/panorama_vcam.py`` (coincide con los vídeos renderizados a ±1 px)."""
    match, (half, view) = video.parent.name, video.stem.split("_", 1)
    w, h = size
    cap = cv2.VideoCapture(str(ST2 / "videos" / match / f"{match}_panorama_{half}_half.mp4"))
    fw, fh = int(cap.get(3)), int(cap.get(4))
    cap.release()
    idx = np.arange(n_frames)
    if view == "vcam796":  # balón del dataset (metros -> píxeles) + media móvil de 1 s + tope de 40 px/frame
        from scipy.interpolate import RBFInterpolator

        kp = json.loads((ST2 / "raw" / match / f"{match}_keypoints.json").read_text())
        to_px = RBFInterpolator(np.array([[float(v) for v in k.strip("()").split(",")] for k in kp]),
                                np.array(list(kp.values()), float), kernel="thin_plate_spline", smoothing=1.0)
        z = np.load(ST2 / "ball" / f"{match}_{half}_ball.npz")
        x, y = z["x"].astype(float), z["y"].astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        f = z["frame"][ok].astype(int)
        cam = to_px(np.column_stack([np.interp(idx, f, x[ok]) + 52.5, np.interp(idx, f, y[ok]) + 34.0]))
        k = 25
        cx, cy = (np.convolve(np.pad(c, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
                  for c in (cam[:, 0], cam[:, 1]))
        for c in (cx, cy):
            for i in range(1, len(c)):
                d = c[i] - c[i - 1]
                if abs(d) > 40.0:
                    c[i] = c[i - 1] + np.sign(d) * 40.0
    else:  # trayectoria final de vcam_path.py (p. ej. vcam796vl1 -> <m>_<parte>_vcam_vl1.npz), renderizada sin suavizar
        c = np.load(VCAM_PATHS / f"{match}_{half}_vcam_{view.removeprefix('vcam796')}.npz")
        cx, cy = np.interp(idx, c["frame"], c["x"]), np.interp(idx, c["frame"], c["y"])
    return np.clip(cx - w / 2, 0, fw - w).astype(int), np.clip(cy - h / 2, 0, fh - h).astype(int)


def st2_players(match: str, half: str) -> dict[str, np.ndarray]:
    """Cajas de los 22 jugadores por frame (``frame`` numerado desde 1, píxeles de la panorámica) del game state del
    dataset. El JSON pesa 2,5 GB por parte: se recorre una vez con una expresión regular y se guarda en un npz."""
    cache = PLAYERS_CACHE / f"{match}_{half}.npz"
    if not cache.exists():
        ann = re.compile(rb'"image_id": "(\d+)",\s*"track_id": \d+,\s*"supercategory": "object",\s*"category_id": \d+,\s*'
                         rb'"attributes": \{([^}]*)\},\s*"bbox_image": \{\s*"x": ([-\d.e]+),\s*"y": ([-\d.e]+),\s*'
                         rb'"x_center": [-\d.e]+,\s*"y_center": [-\d.e]+,\s*"w": ([-\d.e]+),\s*"h": ([-\d.e]+)')
        pid = re.compile(rb'"player_id": (\d+)')
        rows = []
        with open(ST2 / "gsr" / match / f"{match}_{half}.json", "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for g in ann.finditer(mm):
                p = pid.search(g.group(2))
                if p:
                    rows.append((int(g.group(1)[-6:]), int(p.group(1)), *(float(g.group(i)) for i in (3, 4, 5, 6))))
        r = np.array(rows)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez_compressed(tmp, frame=r[:, 0].astype(np.int32), player_id=r[:, 1].astype(np.int64),
                            **{k: r[:, i].astype(np.float32) for i, k in enumerate("xywh", start=2)})
        tmp.rename(cache)
    return dict(np.load(cache))


def _between_executors(n: int, pos: dict, t_ev: np.ndarray, p_ev: list, hold: int) -> np.ndarray:
    """Centro (y, x) por frame a partir de los eventos (frame ``t_ev``, jugador ``p_ev``) y de ``pos`` (jugador -> centro
    de su caja por frame, NaN si no se ve). Entre dos eventos consecutivos, de ejecutores A y B, el centro va de la caja
    de A a la de B, las dos tomadas en el frame actual (así el movimiento de cámara ya está descontado); si solo se ve
    uno, vale en su mitad del tramo. A más de ``hold`` frames de cualquier evento no hay dato."""
    track = np.full((n, 2), np.nan)
    if not len(t_ev):
        return track
    order = np.argsort(t_ev, kind="stable")
    t_ev, p_ev = np.asarray(t_ev)[order], [p_ev[i] for i in order]
    bounds = [(max(t_ev[0] - hold, 0), t_ev[0], None, p_ev[0])]
    bounds += [(t_ev[i], t_ev[i + 1], p_ev[i], p_ev[i + 1]) for i in range(len(t_ev) - 1)]
    bounds += [(t_ev[-1], min(t_ev[-1] + hold, n - 1), p_ev[-1], None)]
    nowhere = np.full((n, 2), np.nan)
    for t0, t1, pa, pb in bounds:
        t = np.arange(t0, t1 + 1)
        if pa is None or pb is None:
            alpha = np.full(len(t), 1.0 if pa is None else 0.0)
        elif t1 - t0 <= 2 * hold:
            alpha = (t - t0) / max(t1 - t0, 1)
        else:
            alpha = np.where(t <= t0 + hold, 0.0, np.where(t >= t1 - hold, 1.0, np.nan))
        a_xy, b_xy = pos.get(pa, nowhere)[t], pos.get(pb, nowhere)[t]
        va, vb = np.isfinite(a_xy).all(1), np.isfinite(b_xy).all(1)
        al = alpha[:, None]
        with np.errstate(invalid="ignore"):
            track[t] = np.where((va & vb)[:, None], (1 - al) * a_xy + al * b_xy,
                                np.where((va & (alpha < 0.5))[:, None], a_xy,
                                         np.where((vb & (alpha >= 0.5))[:, None], b_xy, np.nan)))
    return track


def _st2(video: Path, n_frames: int, fps: float = 25.0) -> np.ndarray:
    match, half = video.parent.name, video.stem.split("_", 1)[0]
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.release()
    x0, y0 = st2_window(video, n_frames, (w, h))
    z = st2_players(match, half)
    f = z["frame"] - 1  # el frame k del game state es el k-ésimo del vídeo
    ok = (f >= 0) & (f < n_frames)
    f = np.clip(f, 0, n_frames - 1)
    u, v = (z["x"] + z["w"] / 2 - x0[f]) / w, (z["y"] + z["h"] / 2 - y0[f]) / h
    ok &= (u >= 0) & (u < 1) & (v >= 0) & (v < 1)  # fuera de la ventana de la cámara el jugador no se ve
    pos = {}
    for pid in np.unique(z["player_id"][ok]):
        sel = ok & (z["player_id"] == pid)
        pos[int(pid)] = np.full((n_frames, 2), np.nan)
        pos[int(pid)][f[sel]] = np.column_stack([v[sel], u[sel]])
    acts = [a for a in json.loads((ST2 / "bas" / match / f"{match}_12_class_events.json").read_text())["actions"]
            if a["gameTime"].startswith({"1st": "1", "2nd": "2"}[half] + " -")]
    t_ev = np.clip(np.array([int(a["frame"]) - 1 for a in acts]), 0, n_frames - 1)
    p_ev = [int(a["player_id"]) if a.get("player_id") else -1 for a in acts]
    return _smooth(_between_executors(n_frames, pos, t_ev, p_ev, int(round(HOLD_S * fps))))


def _pcbas(video: Path, n_frames: int, fps: float = 25.0, feet: bool = False) -> np.ndarray:
    import h5py

    cap = cv2.VideoCapture(str(video))
    w, h = cap.get(3), cap.get(4)
    cap.release()
    track = np.full((n_frames, 2), np.nan)
    with h5py.File(PCBAS / "labels" / "val_tactical_data.h5") as f:
        for key in sorted(k for k in f if k.rsplit("_H", 1)[0] == video.stem):
            a = f[key][...]
            fr = a[:, 0].astype(int)
            f0, n = fr.min(), fr.max() - fr.min() + 1
            pos = {}  # jugador -> centro de su caja (y, x) normalizado por frame de la parte; NaN si no se ve
            for pid in np.unique(a[:, 1]):
                r = a[a[:, 1] == pid]
                pos[pid] = np.full((n, 2), np.nan)
                pos[pid][r[:, 0].astype(int) - f0] = np.column_stack([(r[:, 10] + r[:, 12] * (1.0 if feet else 0.5)) / h,
                                                                      (r[:, 9] + r[:, 11] / 2) / w])
            ev = a[a[:, 13] != 0]
            part = _between_executors(n, pos, ev[:, 0].astype(int) - f0, list(ev[:, 1]), int(round(HOLD_S * fps)))
            end = min(f0 + n, n_frames)
            track[f0:end] = part[: end - f0]
    return _smooth(track)


def _det(video: Path, n_frames: int, conf: float = 0.3, max_gap: int = 25) -> np.ndarray:
    """Balón de nuestro detector, filtrado: confianza >= conf, caja pequeña, coherente con un vecino a <= 5 frames
    (velocidad plausible) y huecos de hasta 1 s interpolados. Lo demás queda sin dato."""
    cap = cv2.VideoCapture(str(video))
    w, h = cap.get(3), cap.get(4)
    cap.release()
    speed, slack = 40.0 * w / 1280, 20.0 * w / 1280  # px/frame
    xy = np.full((n_frames, 2), np.nan)
    with open(video.with_name(video.stem + ".tracks.jsonl")) as fh:
        for line in fh:
            if '"ball": null' in line:
                continue
            r = json.loads(line)
            b = r["ball"]
            if b["confidence"] >= conf and b["xyxy"][2] - b["xyxy"][0] <= 0.04 * w and r["frame_idx"] < n_frames:
                xy[r["frame_idx"]] = b["center_xy"]
    keep = np.zeros(n_frames, bool)
    for d in range(1, 6):
        with np.errstate(invalid="ignore"):
            near = np.hypot(*(xy[d:] - xy[:-d]).T) <= speed * d + slack
        keep[d:] |= near
        keep[:-d] |= near
    xy[~keep] = np.nan
    good = np.flatnonzero(keep)
    for a, b in zip(good[:-1], good[1:]):
        if 1 < b - a <= max_gap and np.hypot(*(xy[b] - xy[a])) <= speed * (b - a) + slack:
            xy[a + 1:b] = xy[a] + (xy[b] - xy[a]) * (np.arange(1, b - a) / (b - a))[:, None]
    return _smooth(np.column_stack([xy[:, 1] / h, xy[:, 0] / w]))


def roi_track(video: Path, mode: str, n_frames: int) -> np.ndarray:
    """Centro de la RoI por frame de vídeo: anotación del dataset del vídeo (``gt``, ``gtfeet``, ``fix``) o ``det``."""
    if mode == "det":
        return _det(video, n_frames)
    p = video.resolve().as_posix()
    if "/soccertrack-v2/views/" in p:
        return _st2(video, n_frames)
    if "/pcbas-2026/videos/" in p:
        return _pcbas(video, n_frames, feet=mode == "gtfeet")
    raise SystemExit(f"{video}: este dataset no trae coordenadas anotadas (SN-BAS solo da instante, clase y equipo). "
                     "No hay RoI GT posible: usa --roi det (pseudo-GT de nuestro detector) o --roi center.")


class RoIOverride(torch.nn.Module):
    """Ocupa el sitio de ``model._model.roi_selector``. En modo ``saliency`` delega en el selector original; en los
    demás devuelve los centros dados (``given``, del dataset) con tamaño fijo ``ROI`` y solo recurre a la saliencia en los
    frames sin dato. Con ``log_saliency=False`` y todos los frames con dato, los mapas de saliencia no se calculan."""

    def __init__(self, base: torch.nn.Module, log_saliency: bool = True) -> None:
        super().__init__()
        self.base, self.sizes, self.log_saliency = base, base.sizes, log_saliency
        self.mode, self.given, self.saliency, self.used = "saliency", None, None, None

    def new_batch(self) -> None:
        self.saliency = None  # depende solo de la rama de baja resolución: se calcula una vez por lote, no por modo

    def _saliency(self, maps: torch.Tensor):
        if self.saliency is None:
            self.saliency = self.base(maps)
        return self.saliency

    def forward(self, maps: torch.Tensor):
        if self.mode in ("saliency", "salcheck"):
            centers, sizes = self._saliency(maps)
        elif self.mode == "fix":  # saliencia desplazada lo justo para que el punto anotado quede a <= FIX_PX del centro
            c, sizes = self._saliency(maps)
            given = self.given.to(c.device, c.dtype)
            valid = torch.isfinite(given).all(-1, keepdim=True)
            margin = c.new_tensor([FIX_PX / self.base.original_size[0], FIX_PX / self.base.original_size[1]])
            centers = torch.where(valid, torch.minimum(torch.maximum(c, given - margin), given + margin), c)
        else:
            b, t = maps.shape[:2]
            given = (torch.full((b, t, 2), 0.5) if self.mode == "center" else self.given).to(maps.device, torch.float32)
            valid = torch.isfinite(given).all(-1, keepdim=True)
            if bool(valid.all()) and not self.log_saliency:
                centers, sizes = given, torch.full_like(given, float(ROI))
            else:
                c, s = self._saliency(maps)
                centers, sizes = torch.where(valid, given, c), torch.where(valid, torch.full_like(s, float(ROI)), s)
        self.used = centers
        return centers, sizes


def with_rois(dataset_cls):
    """``ActionSpotInferenceDataset`` que carga las coordenadas de su vídeo y entrega, con cada clip, los centros de
    RoI de sus frames para cada modo que las necesita (``TRACKED``)."""

    class RoIDataset(dataset_cls):
        def __init__(self, *args, modes=(), **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.tracks = {m: roi_track(Path(self.video_path), m, self._video_len)[::self._stride].astype(np.float32)
                           for m in modes if m in TRACKED}  # modo -> (frames muestreados, 2)

        def __iter__(self):
            for clip, start in super().__iter__():
                idx = np.arange(start, start + self._clip_len)
                rois = {}
                for mode, track in self.tracks.items():
                    c = np.full((self._clip_len, 2), np.nan, np.float32)
                    ok = (idx >= 0) & (idx < len(track))  # el relleno inicial (frames negros) no tiene dato
                    c[ok] = track[idx[ok]]
                    rois[mode] = torch.from_numpy(c)
                yield clip, start, rois

    return RoIDataset
