"""
core/homography.py
===================

Calibración de campo y proyección del tracking a coordenadas 2D de campo
(metros), usando los dos HRNet de PnLCalib (keypoints + líneas).

Repo vendorizado en ``PnLCalib/`` (clonado de https://github.com/mguti97/PnLCalib);
pesos ``SV_kp`` / ``SV_lines`` en ``models/calibration/`` (releases v1.0.0 del
mismo repo). Ninguno de los dos se versiona — ver .gitignore.

Sistema de coordenadas
----------------------
⚠️  PnLCalib trabaja en coordenadas de campo **centradas**: su
``keypoint_world_coords_2D`` aplica ``[x - 52.5, y - 34]``, así que su
homografía devuelve X ∈ [-52.5, 52.5], Y ∈ [-34, 34] con el origen en el
centro del campo.

Este módulo NO propaga esa convención: aplica una traslación
(``_CORNER_ORIGIN_SHIFT``) para devolver siempre **origen en la esquina**,
X ∈ [0, 105], Y ∈ [0, 68]. Es la convención que asumen ``PitchRenderer`` y
todo lo que consuma ``field_xy`` aguas abajo (zonas, áreas, tercios), y la
que hace que un número suelto sea legible sin recordar el signo.

Diseño
------
- ``FieldCalibrator``: carga los dos HRNet una vez y resuelve homografías por
  lotes (``estimate_batch``) o de una en una (``estimate``). GPU + fp16 por
  defecto: es la etapa más cara del pipeline y es GPU-bound (ver tabla de
  medidas en ``config.HomographyConfig``).
- ``ensure_homography(video)``: cachea las homografías en
  ``<video>.homography.jsonl``, con el mismo patrón tmp+rename que
  ``core.tracking.ensure_tracking`` (una caché a medias por un crash nunca
  debe aparentar estar completa). Es una caché independiente del tracking a
  propósito: es lo caro, no depende del detector, y se reutiliza entre
  ejecuciones que cambien cualquier otra cosa del pipeline.
- ``HomographySequence``: carga esa caché y resuelve, para cualquier frame,
  la homografía del frame calibrado más cercano (dentro de ``max_gap_frames``).
  Eso es lo que permite ``stride > 1`` sin arrastrar una H obsoleta a través
  de un corte de plano.
- ``project_tracks``: une ``tracks.jsonl`` + caché de homografías y escribe
  ``tracks_2d.jsonl``, añadiendo ``field_xy`` a cada track y al balón.
- ``PitchRenderer`` / ``render_radar_video``: vista cenital ("radar") para
  poder VER que la calibración es correcta, que es la única forma seria de
  validar esto.

Uso desde el pipeline: ``main.py`` lo invoca entre tracking y analytics
(``--skip-homography`` para saltarlo). Como CLI independiente:

    python -m core.homography --source partido.mp4 --tracks partido.tracks.jsonl \\
        --output tracks_2d.jsonl --radar-video radar.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
from bisect import bisect_left
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
import torch

import config
from core.tracking import _repo_import_context

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Convención de coordenadas
# --------------------------------------------------------------------------- #

# Denominador proyectivo mínimo para dar un punto por proyectable. Medido en
# el partido de ground_truth/: los puntos sanos tienen |w| ~ 2.1 (mediana) y
# los que se iban a >200 m, ~0.07. El corte va holgadamente por debajo de los
# sanos para no descartar jugadores legítimamente lejanos.
_MIN_PROJECTIVE_SCALE: float = 1e-3

# Margen alrededor del campo dentro del cual una proyección se considera
# creíble. Un jugador puede estar legítimamente fuera de banda (saque, córner,
# banquillo), pero no a 30 m del lateral.
_FIELD_SANITY_MARGIN_M: float = 20.0


def _corner_origin_shift(pitch_length_m: float, pitch_width_m: float) -> np.ndarray:
    """Traslación centro-de-campo -> esquina, para pre-multiplicar por la H
    de PnLCalib (ver nota de "Sistema de coordenadas" en el docstring)."""
    return np.array([
        [1.0, 0.0, pitch_length_m / 2.0],
        [0.0, 1.0, pitch_width_m / 2.0],
        [0.0, 0.0, 1.0],
    ])


@dataclass
class CalibrationResult:
    """Homografía de UN frame: [x_img, y_img, 1] -> [X_m, Y_m, w] (proyectivo),
    ya en coordenadas de esquina (0<=X<=105, 0<=Y<=68)."""

    homography: Optional[np.ndarray]
    num_keypoints: int
    reprojection_error: Optional[float]

    @property
    def ok(self) -> bool:
        return self.homography is not None


# --------------------------------------------------------------------------- #
#  Calibrador
# --------------------------------------------------------------------------- #

class FieldCalibrator:
    """Envuelve los dos HRNet de PnLCalib y resuelve la homografía
    imagen -> campo (metros, origen en la esquina)."""

    def __init__(self, cfg: Optional[config.HomographyConfig] = None,
                 device: Optional[str] = None):
        self.config = cfg or config.HomographyConfig()
        self.device: str = (
            device or self.config.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # fp16 solo tiene sentido (y solo está soportado por autocast) en CUDA.
        self.half: bool = self.config.half and self.device.startswith("cuda")
        self._shift = _corner_origin_shift(
            self.config.pitch_length_m, self.config.pitch_width_m
        )
        self._load_pnlcalib(self.config.pnlcalib_repo_path)
        self._load_models()

    # -- carga ------------------------------------------------------------- #

    def _load_pnlcalib(self, repo_path: Path) -> None:
        """Importa PnLCalib de forma aislada.

        Reutiliza ``core.tracking._repo_import_context``, que ya resuelve
        exactamente este problema: PnLCalib define un paquete namespace
        ``utils`` que choca con el ``utils/`` regular del proyecto (y los
        regulares ganan independientemente del orden de sys.path).
        """
        import yaml

        repo_path = Path(repo_path)
        if not (repo_path / "model" / "cls_hrnet.py").exists():
            raise FileNotFoundError(
                f"No encuentro el código de PnLCalib en {repo_path}. "
                "Clónalo con: git clone https://github.com/mguti97/PnLCalib.git"
            )

        with _repo_import_context(repo_path):
            from model.cls_hrnet import get_cls_net
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            from utils.utils_calib import FramebyFrameCalib
            from utils.utils_heatmap import (
                get_keypoints_from_heatmap_batch_maxpool,
                get_keypoints_from_heatmap_batch_maxpool_l,
                complete_keypoints,
                coords_to_dict,
            )

            with open(repo_path / "config" / "hrnetv2_w48.yaml") as f:
                cfg_kp = yaml.safe_load(f)
            with open(repo_path / "config" / "hrnetv2_w48_l.yaml") as f:
                cfg_line = yaml.safe_load(f)

        self._get_cls_net = get_cls_net
        self._get_cls_net_l = get_cls_net_l
        self._FramebyFrameCalib = FramebyFrameCalib
        self._get_kp_from_heatmap = get_keypoints_from_heatmap_batch_maxpool
        self._get_kp_from_heatmap_l = get_keypoints_from_heatmap_batch_maxpool_l
        self._complete_keypoints = complete_keypoints
        self._coords_to_dict = coords_to_dict
        self._cfg_kp = cfg_kp
        self._cfg_line = cfg_line

    def _load_models(self) -> None:
        for path in (self.config.weights_kp, self.config.weights_line):
            if not Path(path).exists():
                raise FileNotFoundError(
                    f"Faltan los pesos de calibración: {path}. Descárgalos de "
                    "https://github.com/mguti97/PnLCalib/releases/tag/v1.0.0 "
                    "(SV_kp y SV_lines) a models/calibration/."
                )

        self.model_kp = self._get_cls_net(self._cfg_kp)
        self.model_kp.load_state_dict(
            torch.load(self.config.weights_kp, map_location=self.device)
        )
        self.model_kp.to(self.device).eval()

        self.model_line = self._get_cls_net_l(self._cfg_line)
        self.model_line.load_state_dict(
            torch.load(self.config.weights_line, map_location=self.device)
        )
        self.model_line.to(self.device).eval()

        logger.info(
            "PnLCalib cargado (kp+lines) en device=%s, fp16=%s", self.device, self.half
        )

    # -- inferencia -------------------------------------------------------- #

    def _preprocess(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """Lote de frames BGR (resolución original) -> tensor (B,3,540,960)."""
        rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div(255.0)
        w_target, h_target = self.config.model_input_size
        if tensor.shape[-1] != w_target or tensor.shape[-2] != h_target:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(h_target, w_target), mode="bilinear", align_corners=False
            )
        return tensor.to(self.device)

    def estimate_batch(self, frames_bgr: list[np.ndarray]) -> list[CalibrationResult]:
        """Calibra un lote de frames (todos de la misma resolución).

        La parte GPU (los dos HRNet) se hace en lote porque es lo caro; el
        solve por frame es de 0.5 ms y va en bucle, no compensa complicarlo.
        """
        if not frames_bgr:
            return []

        h_orig, w_orig = frames_bgr[0].shape[:2]
        tensor = self._preprocess(frames_bgr)
        _, _, h_in, w_in = tensor.shape

        autocast = (
            torch.autocast("cuda", dtype=torch.float16) if self.half else nullcontext()
        )
        with torch.no_grad(), autocast:
            heatmaps = self.model_kp(tensor)
            heatmaps_l = self.model_line(tensor)

        # La decodificación de heatmaps hace comparaciones contra umbrales
        # calibrados en fp32; volvemos a fp32 antes de decodificar.
        kp_coords = self._get_kp_from_heatmap(heatmaps[:, :-1, :, :].float())
        line_coords = self._get_kp_from_heatmap_l(heatmaps_l[:, :-1, :, :].float())
        kp_dicts = self._coords_to_dict(kp_coords, threshold=self.config.kp_threshold)
        line_dicts = self._coords_to_dict(line_coords, threshold=self.config.line_threshold)

        results: list[CalibrationResult] = []
        for kp_raw, lines_raw in zip(kp_dicts, line_dicts):
            results.append(self._solve(kp_raw, lines_raw, w_in, h_in, w_orig, h_orig))
        return results

    def estimate(self, frame_bgr: np.ndarray) -> CalibrationResult:
        """Calibra un único frame BGR (resolución original)."""
        return self.estimate_batch([frame_bgr])[0]

    def _solve(self, kp_raw, lines_raw, w_in: int, h_in: int,
               w_orig: int, h_orig: int) -> CalibrationResult:
        kp_dict, lines_dict = self._complete_keypoints(
            kp_raw, lines_raw, w=w_in, h=h_in, normalize=True
        )
        num_kp = len(kp_dict)
        if num_kp < 4:
            # Sin 4 correspondencias no hay homografía posible.
            return CalibrationResult(None, num_kp, None)

        # denormalize=True + iwidth/iheight originales: los keypoints vienen
        # normalizados a [0,1] y se reescalan a la resolución REAL del frame,
        # así que la H resultante consume píxeles de la imagen original.
        cam = self._FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
        cam.update(kp_dict, lines_dict)
        try:
            H, err = cam.get_homography_from_ground_plane(
                use_ransac=self.config.use_ransac,
                inverse=True,          # inverse=True -> imagen a campo (lo que queremos)
                refine_lines=self.config.pnl_refine,
            )
        except np.linalg.LinAlgError:
            # Correspondencias degeneradas (p.ej. todos los keypoints casi
            # colineales): la H sale singular y no se puede invertir. Pasa en
            # bastantes frames de un partido real, no es excepcional.
            logger.debug("Homografía singular, frame descartado")
            return CalibrationResult(None, num_kp, None)
        except Exception:
            logger.exception("Fallo resolviendo la homografía")
            return CalibrationResult(None, num_kp, None)

        if H is None:
            return CalibrationResult(None, num_kp, None)

        if err is not None and err > self.config.max_reprojection_error_px:
            logger.debug("Homografía descartada por error de reproyección %.1f px", err)
            return CalibrationResult(None, num_kp, float(err))

        # PnLCalib devuelve coordenadas centradas; pasamos a origen-esquina.
        H_corner = self._shift @ np.asarray(H)
        return CalibrationResult(
            homography=H_corner,
            num_keypoints=num_kp,
            reprojection_error=float(err) if err is not None else None,
        )

    # -- proyección -------------------------------------------------------- #

    @staticmethod
    def project_point(xy: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
        """Punto de imagen (px) -> campo (m, origen en la esquina).

        Devuelve ``(nan, nan)`` si el punto no es proyectable (ver
        ``project_points``).
        """
        result = FieldCalibrator.project_points(np.array([xy], dtype=float), H)
        return float(result[0, 0]), float(result[0, 1])

    @staticmethod
    def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Vectorizada: points (N,2) en px -> (N,2) en metros.

        Las filas no proyectables salen como ``nan``. Son dos casos, y ambos
        aparecen de verdad en un partido real (medido: 4.8% de los puntos):

        - ``w < 0``: el punto cae DETRÁS del plano de la cámara. La división
          proyectiva no avisa, simplemente devuelve un punto reflejado que
          parece plausible. Sin este filtro se colaban como posiciones buenas.
        - ``w ≈ 0``: el punto está sobre la línea del horizonte, donde la
          homografía diverge. Son los que proyectaban a cientos de km.
        """
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return points.reshape(0, 2).copy()

        homog = np.hstack([points.reshape(-1, 2), np.ones((points.shape[0], 1))])
        proj = (H @ homog.T).T
        w = proj[:, 2:3]

        out = np.full((points.shape[0], 2), np.nan)
        valid = (w[:, 0] > _MIN_PROJECTIVE_SCALE)
        out[valid] = proj[valid, :2] / w[valid]
        return out


# --------------------------------------------------------------------------- #
#  Caché de homografías
# --------------------------------------------------------------------------- #

class HomographySequence:
    """Homografías de un vídeo, indexadas por frame.

    ``at(frame_idx)`` devuelve la H del frame calibrado más cercano, siempre
    que esté a menos de ``max_gap_frames``. Frames sin homografía válida
    (repetición, plano corto, pocas líneas visibles) devuelven ``None``: es
    preferible no proyectar a proyectar con una H de otro plano.
    """

    def __init__(self, frame_indices: list[int], homographies: list[np.ndarray],
                 max_gap_frames: int):
        self._frames = frame_indices          # ordenados, solo frames con H válida
        self._homographies = homographies
        self._max_gap = max_gap_frames

    def __len__(self) -> int:
        return len(self._frames)

    def at(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        pos = bisect_left(self._frames, frame_idx)
        candidates = []
        if pos < len(self._frames):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best = min(candidates, key=lambda i: abs(self._frames[i] - frame_idx))
        if abs(self._frames[best] - frame_idx) > self._max_gap:
            return None
        return self._homographies[best]

    @classmethod
    def load(cls, path: Path, max_gap_frames: int = 25) -> "HomographySequence":
        frames, matrices = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("homography") is None:
                    continue
                frames.append(rec["frame_idx"])
                matrices.append(np.asarray(rec["homography"], dtype=float))
        order = np.argsort(frames)
        return cls([frames[i] for i in order], [matrices[i] for i in order],
                   max_gap_frames)


def iter_calibration(source_video: Path, calibrator: FieldCalibrator,
                     stride: int = 1, batch_size: int = 4) -> Iterator[dict]:
    """Recorre el vídeo calibrando 1 de cada ``stride`` frames.

    Emite un registro por frame calibrado (no por frame del vídeo), con la
    homografía en coordenadas de esquina o ``None`` si no se pudo resolver.
    """
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")

    pending_frames: list[np.ndarray] = []
    pending_indices: list[int] = []
    frame_idx = 0

    def _flush() -> Iterator[dict]:
        if not pending_frames:
            return
        for idx, result in zip(pending_indices, calibrator.estimate_batch(pending_frames)):
            yield {
                "frame_idx": idx,
                "homography": result.homography.tolist() if result.ok else None,
                "num_keypoints": result.num_keypoints,
                "reprojection_error": result.reprojection_error,
            }
        pending_frames.clear()
        pending_indices.clear()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                pending_frames.append(frame)
                pending_indices.append(frame_idx)
                if len(pending_frames) >= batch_size:
                    yield from _flush()
            frame_idx += 1
        yield from _flush()
    finally:
        cap.release()


def ensure_homography(video_path: Path, cfg: Optional[config.HomographyConfig] = None,
                      device: Optional[str] = None,
                      calibrator: Optional[FieldCalibrator] = None) -> Path:
    """Corre la calibración sobre ``video_path`` si la caché no existe ya.

    Mismo patrón tmp+rename que ``core.tracking.ensure_tracking``: una caché
    truncada por un crash nunca debe aparentar estar completa.
    """
    cfg = cfg or config.HomographyConfig()
    homography_path = video_path.with_suffix(".homography.jsonl")
    if homography_path.exists():
        logger.info("homography.jsonl ya existe, reutilizando: %s", homography_path)
        return homography_path

    logger.info("Calibrando campo (stride=%d) sobre %s ...", cfg.stride, video_path)
    calibrator = calibrator or FieldCalibrator(cfg, device=device)

    tmp_path = homography_path.with_suffix(homography_path.suffix + ".tmp")
    n_ok = n_total = 0
    with open(tmp_path, "w") as f:
        for record in iter_calibration(video_path, calibrator,
                                       stride=cfg.stride, batch_size=cfg.batch_size):
            n_total += 1
            n_ok += record["homography"] is not None
            f.write(json.dumps(record) + "\n")
    tmp_path.rename(homography_path)

    pct = 100.0 * n_ok / n_total if n_total else 0.0
    logger.info("Calibración: %d/%d frames con homografía válida (%.1f%%)",
                n_ok, n_total, pct)
    if n_total and pct < 50.0:
        logger.warning(
            "Menos de la mitad de los frames se han calibrado. Revisa el radar "
            "antes de fiarte de cualquier métrica en metros."
        )
    return homography_path


# --------------------------------------------------------------------------- #
#  Proyección del tracking
# --------------------------------------------------------------------------- #

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def project_tracks(tracks_jsonl_path: Path, homography_jsonl_path: Path,
                   output_path: Path, max_gap_frames: int = 25) -> Path:
    """``tracks.jsonl`` + caché de homografías -> ``tracks_2d.jsonl``.

    Añade ``field_xy`` (metros) a cada track y al balón. Los tracks usan
    ``anchor_xy`` (bottom-center de la caja = punto de apoyo en el suelo),
    que es el único punto del jugador que está REALMENTE sobre el plano del
    campo y por tanto el único que la homografía proyecta bien.

    El balón se proyecta desde ``center_xy`` asumiendo z=0, así que cuando
    está en el aire la proyección es incorrecta por construcción; se marca
    con ``ball.field_xy_assumes_ground = true`` para que aguas abajo se sepa.
    """
    homographies = HomographySequence.load(homography_jsonl_path, max_gap_frames)
    logger.info("Cargadas %d homografías válidas de %s",
                len(homographies), homography_jsonl_path)

    cfg = config.HomographyConfig()
    n_projected = n_missing = 0
    n_points = n_rejected = 0

    def _plausible(xy: np.ndarray) -> bool:
        """Descarta proyecciones no calculables (nan) o absurdamente lejanas."""
        if not np.all(np.isfinite(xy)):
            return False
        m = _FIELD_SANITY_MARGIN_M
        return (-m <= xy[0] <= cfg.pitch_length_m + m
                and -m <= xy[1] <= cfg.pitch_width_m + m)

    with open(output_path, "w") as out_f:
        for record in _iter_jsonl(tracks_jsonl_path):
            H = homographies.at(record["frame_idx"])
            out_rec = dict(record)
            out_rec["homography_ok"] = H is not None

            if H is None:
                n_missing += 1
            else:
                n_projected += 1
                tracks = out_rec.get("tracks", [])
                anchors = np.array([t["anchor_xy"] for t in tracks], dtype=float)
                if anchors.size:
                    field_xy = FieldCalibrator.project_points(anchors.reshape(-1, 2), H)
                    for track, xy in zip(tracks, field_xy):
                        n_points += 1
                        # Sin field_xy es una ausencia explícita: aguas abajo se
                        # comprueba con `"field_xy" in track`, así que un punto
                        # no proyectable nunca se confunde con uno bueno.
                        if _plausible(xy):
                            track["field_xy"] = [float(xy[0]), float(xy[1])]
                        else:
                            n_rejected += 1

                ball = out_rec.get("ball")
                if ball is not None:
                    ball_xy = FieldCalibrator.project_points(
                        np.array([ball["center_xy"]], dtype=float), H
                    )[0]
                    if _plausible(ball_xy):
                        ball["field_xy"] = [float(ball_xy[0]), float(ball_xy[1])]
                        ball["field_xy_assumes_ground"] = True

            out_f.write(json.dumps(out_rec) + "\n")

    pct = 100.0 * n_rejected / n_points if n_points else 0.0
    logger.info("Proyectados %d frames, %d sin homografía -> %s",
                n_projected, n_missing, output_path)
    logger.info("Puntos: %d proyectados, %d descartados por no plausibles (%.1f%%)",
                n_points - n_rejected, n_rejected, pct)
    return output_path


# --------------------------------------------------------------------------- #
#  Render cenital ("radar")
# --------------------------------------------------------------------------- #

class PitchRenderer:
    """Plantilla de campo en planta (105x68 m) y puntos proyectados encima.

    Asume la convención de este módulo: origen en la esquina, X ∈ [0, 105],
    Y ∈ [0, 68].
    """

    LINE_COLOR = (220, 220, 220)
    BG_COLOR = (40, 110, 40)
    TEAM_COLORS = ((0, 0, 235), (235, 120, 0))   # BGR: cluster 0 / cluster 1
    NEUTRAL_COLOR = (200, 200, 200)
    BALL_COLOR = (0, 255, 255)

    def __init__(self, pixels_per_meter: int = 8, pitch_length_m: float = 105.0,
                 pitch_width_m: float = 68.0, margin_px: int = 20):
        self.ppm = pixels_per_meter
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.margin = margin_px
        self.width_px = int(pitch_length_m * pixels_per_meter) + 2 * margin_px
        self.height_px = int(pitch_width_m * pixels_per_meter) + 2 * margin_px

    def _m2px(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (int(x_m * self.ppm) + self.margin, int(y_m * self.ppm) + self.margin)

    def blank_pitch(self) -> np.ndarray:
        img = np.full((self.height_px, self.width_px, 3), self.BG_COLOR, dtype=np.uint8)
        L, W = self.pitch_length_m, self.pitch_width_m

        def line(p1, p2):
            cv2.line(img, self._m2px(*p1), self._m2px(*p2), self.LINE_COLOR, 2)

        line((0, 0), (L, 0)); line((L, 0), (L, W))
        line((L, W), (0, W)); line((0, W), (0, 0))
        line((L / 2, 0), (L / 2, W))
        # Áreas grandes
        line((0, 13.84), (16.5, 13.84)); line((16.5, 13.84), (16.5, 54.16)); line((16.5, 54.16), (0, 54.16))
        line((L, 13.84), (L - 16.5, 13.84)); line((L - 16.5, 13.84), (L - 16.5, 54.16)); line((L - 16.5, 54.16), (L, 54.16))
        # Áreas pequeñas
        line((0, 24.84), (5.5, 24.84)); line((5.5, 24.84), (5.5, 43.16)); line((5.5, 43.16), (0, 43.16))
        line((L, 24.84), (L - 5.5, 24.84)); line((L - 5.5, 24.84), (L - 5.5, 43.16)); line((L - 5.5, 43.16), (L, 43.16))
        # Círculo central y puntos de penalti
        cv2.circle(img, self._m2px(L / 2, W / 2), int(9.15 * self.ppm), self.LINE_COLOR, 2)
        cv2.circle(img, self._m2px(L / 2, W / 2), 3, self.LINE_COLOR, -1)
        cv2.circle(img, self._m2px(11.0, W / 2), 3, self.LINE_COLOR, -1)
        cv2.circle(img, self._m2px(L - 11.0, W / 2), 3, self.LINE_COLOR, -1)
        return img

    def draw_points(self, img: np.ndarray, points_m: Iterable[tuple[float, float]],
                    color=(0, 0, 255), radius: int = 5,
                    labels: Optional[list[str]] = None) -> np.ndarray:
        for i, (x_m, y_m) in enumerate(points_m):
            px, py = self._m2px(x_m, y_m)
            cv2.circle(img, (px, py), radius, color, -1)
            cv2.circle(img, (px, py), radius, (20, 20, 20), 1)
            if labels is not None and i < len(labels):
                cv2.putText(img, str(labels[i]), (px + 6, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def render_radar_video(tracks_2d_path: Path, output_path: Path, fps: float = 25.0,
                       team_by_tracker_id: Optional[dict[int, int]] = None,
                       renderer: Optional[PitchRenderer] = None,
                       show_ids: bool = False) -> Path:
    """Vídeo cenital a partir de ``tracks_2d.jsonl``.

    Es la herramienta de validación del módulo: si los jugadores no caen
    dentro del campo y no se mueven de forma coherente aquí, la calibración
    está mal y ninguna métrica en metros vale nada.
    """
    renderer = renderer or PitchRenderer()
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (renderer.width_px, renderer.height_px),
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo abrir el writer de vídeo: {output_path}")

    n_frames = 0
    try:
        for record in _iter_jsonl(tracks_2d_path):
            img = renderer.blank_pitch()
            if not record.get("homography_ok"):
                cv2.putText(img, "sin homografia", (renderer.margin + 8, renderer.margin + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                by_color: dict[tuple[int, int, int], list] = {}
                labels: dict[tuple[int, int, int], list[str]] = {}
                for track in record.get("tracks", []):
                    if "field_xy" not in track:
                        continue
                    cluster = (team_by_tracker_id or {}).get(track.get("tracker_id"))
                    color = (renderer.TEAM_COLORS[cluster]
                             if cluster in (0, 1) else renderer.NEUTRAL_COLOR)
                    by_color.setdefault(color, []).append(track["field_xy"])
                    labels.setdefault(color, []).append(str(track.get("tracker_id", "")))
                for color, points in by_color.items():
                    renderer.draw_points(img, points, color=color, radius=5,
                                         labels=labels[color] if show_ids else None)

                ball = record.get("ball")
                if ball and "field_xy" in ball:
                    renderer.draw_points(img, [ball["field_xy"]],
                                         color=renderer.BALL_COLOR, radius=4)

            writer.write(img)
            n_frames += 1
    finally:
        writer.release()

    logger.info("Radar escrito en %s (%d frames)", output_path, n_frames)
    return output_path


def video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibración de campo (PnLCalib) + proyección 2D del tracking"
    )
    parser.add_argument("--source", required=True, type=Path, help="Vídeo original")
    parser.add_argument("--tracks", type=Path, default=None,
                        help="tracks.jsonl de core/tracking.py (por defecto, <source>.tracks.jsonl)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Salida tracks_2d.jsonl (por defecto, junto al vídeo)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=None,
                        help="Calibra 1 de cada N frames (por defecto, config.HomographyConfig.stride)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-half", action="store_true", help="Desactiva fp16")
    parser.add_argument("--pnl-refine", action="store_true",
                        help="Refinamiento PnL con líneas (más lento, más preciso)")
    parser.add_argument("--radar-video", type=Path, default=None,
                        help="Si se indica, escribe además el vídeo cenital de validación")
    parser.add_argument("--teams", type=Path, default=None,
                        help="teams.json para colorear el radar por equipo")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from dataclasses import replace

    cfg = config.HomographyConfig()
    overrides = {}
    if args.stride is not None:
        overrides["stride"] = args.stride
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.no_half:
        overrides["half"] = False
    if args.pnl_refine:
        overrides["pnl_refine"] = True
    if overrides:
        cfg = replace(cfg, **overrides)

    tracks_path = args.tracks or args.source.with_suffix(".tracks.jsonl")
    if not tracks_path.exists():
        parser.error(f"No existe el tracking: {tracks_path} (corre antes core.tracking)")
    output_path = args.output or args.source.with_suffix(".tracks_2d.jsonl")

    homography_path = ensure_homography(args.source, cfg, device=args.device)
    project_tracks(tracks_path, homography_path, output_path,
                   max_gap_frames=cfg.max_gap_frames)

    if args.radar_video:
        team_by_tracker_id = None
        if args.teams and args.teams.exists():
            teams_data = json.loads(args.teams.read_text())
            team_by_tracker_id = {
                t["tracker_id"]: t["team_cluster"] for t in teams_data["teams"]
            }
        render_radar_video(output_path, args.radar_video, fps=video_fps(args.source),
                           team_by_tracker_id=team_by_tracker_id)


if __name__ == "__main__":
    main()
