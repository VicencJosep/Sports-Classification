"""
core/homography.py
===================

Calibración de campo por frame y proyección de tracking a coordenadas 2D
de campo (metros), usando los modelos de PnLCalib (keypoints + líneas)
que SoccerMaster reutiliza como módulo de calibración (SV_kp / SV_lines).

Referencia: https://github.com/mguti97/PnLCalib
Ese repo es la fuente de:
  - model/cls_hrnet.py, model/cls_hrnet_l.py   (arquitecturas HRNet)
  - utils/utils_calib.py                        (FramebyFrameCalib)
  - utils/utils_heatmap.py                      (decodificación heatmap -> keypoints)
  - config/hrnetv2_w48.yaml, hrnetv2_w48_l.yaml (configs de red)

⚠️  ASUNCIÓN A VERIFICAR EN TU ENTORNO
---------------------------------------
Este módulo asume que el CÓDIGO de PnLCalib (no solo los pesos .pt) está
disponible en disco, en `PitchCalibrationConfig.pnlcalib_repo_path`.
Los ficheros `SV_kp` / `SV_lines` que ya tienes descargados son solo los
*pesos*; hacen falta también `model/cls_hrnet*.py`, `utils/utils_calib.py`,
`utils/utils_heatmap.py` y los `.yaml` de config para reconstruir la red.

Comprueba primero si esos ficheros ya están vendorizados dentro de tu copia
de `sn-gamestate` (que SoccerMaster usa como dependencia de calibración).
Si no, clona PnLCalib aparte:
    git clone https://github.com/mguti97/PnLCalib.git
y apunta `pnlcalib_repo_path` a esa carpeta.

Diseño
------
- `PitchCalibrationConfig`: dataclass con todos los paths/umbrales. Muévela
  a tu `config.py` junto a `TrackingConfig`/`PipelineConfig` si quieres
  mantener un único punto de configuración (aquí vive standalone para que
  el módulo se pueda revisar sin depender de tu config.py real).
- `FieldCalibrator`: carga los dos HRNet (keypoints + líneas) una vez, y
  expone `.estimate(frame_bgr)` -> `CalibrationResult` con una homografía
  3x3 que mapea píxeles de imagen (resolución original del frame) a metros
  de campo (0<=X<=105, 0<=Y<=68, origen en la esquina).
- `project_point(s)`: aplica la homografía a puntos de imagen.
- `PitchRenderer`: dibuja una plantilla de campo en planta y los puntos
  proyectados, para poder "verlo" (imagen o vídeo).
- CLI: homografía frame a frame sobre un vídeo + tracks.jsonl (formato de
  core/tracking.py) -> tracks_2d.jsonl + (opcional) vídeo radar.

Nota de rendimiento: se pidió homografía POR FRAME (no reutilizar una sola
para todo el clip), lo cual es más robusto ante paneo/zoom de cámara pero
implica correr dos HRNets por frame. En CPU esto es notablemente más lento
que el propio tracking; considera --stride > 1 para sub-muestrear si el
clip es largo.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Reutilizamos el mismo mecanismo de aislamiento de imports que ya
# resolvió la colisión de paquetes `utils`/`model` en core/tracking.py.
# Si tu implementación real vive con otro nombre, ajusta este import.
try:
    from core.tracking import _repo_import_context
except ImportError:  # pragma: no cover - fallback standalone
    from contextlib import contextmanager
    import sys
    import os

    @contextmanager
    def _repo_import_context(repo_path: Path):
        """Fallback local: aísla sys.path/sys.modules y hace chdir
        mientras se importan los paquetes `model`/`utils` del repo
        vendorizado, para no chocar con nuestros propios `core/utils`.
        Si ya tienes la versión real en core/tracking.py, bórrame y usa
        el import de arriba.
        """
        repo_path = Path(repo_path).resolve()
        prev_cwd = os.getcwd()
        prev_sys_path = list(sys.path)
        saved_modules = {}
        for name in list(sys.modules):
            if name == "model" or name.startswith("model.") or \
               name == "utils" or name.startswith("utils."):
                saved_modules[name] = sys.modules.pop(name)
        try:
            os.chdir(repo_path)
            sys.path.insert(0, str(repo_path))
            yield
        finally:
            os.chdir(prev_cwd)
            sys.path = prev_sys_path
            for name in list(sys.modules):
                if name == "model" or name.startswith("model.") or \
                   name == "utils" or name.startswith("utils."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

@dataclass
class PitchCalibrationConfig:
    """Paths y umbrales para el calibrador. Candidata a fusionarse con
    tu config.py (TrackingConfig / PipelineConfig)."""

    pnlcalib_repo_path: Path
    weights_kp: Path
    weights_line: Path
    device: str = "cpu"  # tu contenedor no tiene GPU funcional (ver notas previas)

    # Umbrales por defecto: los mismos que usan los autores en inference.py
    kp_threshold: float = 0.3434
    line_threshold: float = 0.7867

    # Resolución de entrada del modelo (fija por entrenamiento)
    model_input_size: tuple[int, int] = (960, 540)  # (w, h)

    # Homografía: usar RANSAC al resolver el plano de campo
    use_ransac: float = 5.0
    pnl_refine: bool = False  # activa el módulo de refinamiento PnL (más lento, más preciso)

    # Dimensiones canónicas del campo (estándar FIFA / SoccerNet-Calibration)
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    # Config yaml de las redes (relativos a pnlcalib_repo_path)
    cfg_kp_yaml: str = "config/hrnetv2_w48.yaml"
    cfg_line_yaml: str = "config/hrnetv2_w48_l.yaml"


@dataclass
class CalibrationResult:
    homography: Optional[np.ndarray]  # 3x3: [x_img, y_img, 1] -> [X_m, Y_m, w] (proyectivo)
    num_keypoints: int
    reprojection_error: Optional[float]

    @property
    def ok(self) -> bool:
        return self.homography is not None


# --------------------------------------------------------------------------
# Calibrador
# --------------------------------------------------------------------------

class FieldCalibrator:
    """Envuelve los dos HRNet de PnLCalib (keypoints + líneas) y resuelve,
    frame a frame, la homografía imagen -> campo (metros)."""

    def __init__(self, config: PitchCalibrationConfig):
        self.config = config
        self._load_pnlcalib(config.pnlcalib_repo_path)
        self._load_models()

    # -- carga -------------------------------------------------------

    def _load_pnlcalib(self, repo_path: Path) -> None:
        """Importa las clases/funciones de PnLCalib de forma aislada."""
        import yaml

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

            cfg_kp = yaml.safe_load(open(repo_path / self.config.cfg_kp_yaml))
            cfg_line = yaml.safe_load(open(repo_path / self.config.cfg_line_yaml))

        # Guardamos referencias para usarlas fuera del context manager
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
        device = self.config.device
        state_kp = torch.load(self.config.weights_kp, map_location=device)
        self.model_kp = self._get_cls_net(self._cfg_kp)
        self.model_kp.load_state_dict(state_kp)
        self.model_kp.to(device).eval()

        state_line = torch.load(self.config.weights_line, map_location=device)
        self.model_line = self._get_cls_net_l(self._cfg_line)
        self.model_line.load_state_dict(state_line)
        self.model_line.to(device).eval()

        logger.info("PnLCalib cargado (kp+lines) en device=%s", device)

    # -- inferencia ----------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        w_target, h_target = self.config.model_input_size
        if tensor.shape[-1] != w_target or tensor.shape[-2] != h_target:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(h_target, w_target), mode="bilinear", align_corners=False
            )
        return tensor.to(self.config.device)

    def estimate(self, frame_bgr: np.ndarray) -> CalibrationResult:
        """Calibra un único frame (BGR, resolución original) y devuelve
        la homografía imagen(px, resolución original) -> campo(m)."""
        h_orig, w_orig = frame_bgr.shape[:2]
        cam = self._FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)

        tensor = self._preprocess(frame_bgr)
        _, _, h_in, w_in = tensor.shape

        with torch.no_grad():
            heatmaps = self.model_kp(tensor)
            heatmaps_l = self.model_line(tensor)

        kp_coords = self._get_kp_from_heatmap(heatmaps[:, :-1, :, :])
        line_coords = self._get_kp_from_heatmap_l(heatmaps_l[:, :-1, :, :])
        kp_dict = self._coords_to_dict(kp_coords, threshold=self.config.kp_threshold)
        lines_dict = self._coords_to_dict(line_coords, threshold=self.config.line_threshold)
        kp_dict, lines_dict = self._complete_keypoints(
            kp_dict[0], lines_dict[0], w=w_in, h=h_in, normalize=True
        )

        num_kp = len(kp_dict)
        if num_kp < 4:
            # Con menos de 4 puntos no hay homografía posible.
            return CalibrationResult(homography=None, num_keypoints=num_kp, reprojection_error=None)

        cam.update(kp_dict, lines_dict)
        try:
            H, err = cam.get_homography_from_ground_plane(
                use_ransac=self.config.use_ransac,
                inverse=True,
                refine_lines=self.config.pnl_refine,
            )
        except Exception:
            logger.exception("Fallo resolviendo homografía en este frame")
            H, err = None, None

        return CalibrationResult(
            homography=np.asarray(H) if H is not None else None,
            num_keypoints=num_kp,
            reprojection_error=err,
        )

    # -- proyección ------------------------------------------------------

    @staticmethod
    def project_point(xy: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
        """Proyecta un punto de imagen (px) a coordenadas de campo (metros,
        origen en la esquina, 0<=X<=105, 0<=Y<=68)."""
        p = np.array([xy[0], xy[1], 1.0])
        proj = H @ p
        proj = proj / proj[2]
        return float(proj[0]), float(proj[1])

    @staticmethod
    def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Versión vectorizada: points (N,2) -> (N,2) en metros."""
        if points.size == 0:
            return points.copy()
        ones = np.ones((points.shape[0], 1))
        homog = np.hstack([points, ones])  # (N,3)
        proj = (H @ homog.T).T  # (N,3)
        return proj[:, :2] / proj[:, 2:3]


# --------------------------------------------------------------------------
# Render (para "verlo")
# --------------------------------------------------------------------------

class PitchRenderer:
    """Dibuja una plantilla de campo en planta (105x68 m) a una resolución
    en píxeles fija, y permite pintar puntos proyectados encima. Pensado
    para generar una imagen o vídeo "radar view"."""

    LINE_COLOR = (60, 140, 60)
    BG_COLOR = (30, 110, 30)

    def __init__(self, pixels_per_meter: int = 10, pitch_length_m: float = 105.0,
                 pitch_width_m: float = 68.0, margin_px: int = 20):
        self.ppm = pixels_per_meter
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.margin = margin_px
        self.width_px = int(pitch_length_m * pixels_per_meter) + 2 * margin_px
        self.height_px = int(pitch_width_m * pixels_per_meter) + 2 * margin_px

    def _m2px(self, x_m: float, y_m: float) -> tuple[int, int]:
        return (
            int(x_m * self.ppm) + self.margin,
            int(y_m * self.ppm) + self.margin,
        )

    def blank_pitch(self) -> np.ndarray:
        img = np.full((self.height_px, self.width_px, 3), self.BG_COLOR, dtype=np.uint8)
        L, W = self.pitch_length_m, self.pitch_width_m

        def line(p1, p2):
            cv2.line(img, self._m2px(*p1), self._m2px(*p2), self.LINE_COLOR, 2)

        # Contorno
        line((0, 0), (L, 0)); line((L, 0), (L, W))
        line((L, W), (0, W)); line((0, W), (0, 0))
        # Línea de medio campo
        line((L / 2, 0), (L / 2, W))
        # Áreas grandes
        line((0, 13.84), (16.5, 13.84)); line((16.5, 13.84), (16.5, 54.16)); line((16.5, 54.16), (0, 54.16))
        line((L, 13.84), (L - 16.5, 13.84)); line((L - 16.5, 13.84), (L - 16.5, 54.16)); line((L - 16.5, 54.16), (L, 54.16))
        # Áreas pequeñas
        line((0, 24.84), (5.5, 24.84)); line((5.5, 24.84), (5.5, 43.16)); line((5.5, 43.16), (0, 43.16))
        line((L, 24.84), (L - 5.5, 24.84)); line((L - 5.5, 24.84), (L - 5.5, 43.16)); line((L - 5.5, 43.16), (L, 43.16))
        # Círculo central
        cv2.circle(img, self._m2px(L / 2, W / 2), int(9.15 * self.ppm), self.LINE_COLOR, 2)
        cv2.circle(img, self._m2px(L / 2, W / 2), 3, self.LINE_COLOR, -1)
        return img

    def draw_points(self, img: np.ndarray, points_m: np.ndarray, color=(0, 0, 255),
                     radius: int = 5, labels: Optional[list[str]] = None) -> np.ndarray:
        out = img.copy()
        for i, (x_m, y_m) in enumerate(points_m):
            px, py = self._m2px(x_m, y_m)
            cv2.circle(out, (px, py), radius, color, -1)
            if labels is not None and i < len(labels):
                cv2.putText(out, str(labels[i]), (px + 6, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return out


# --------------------------------------------------------------------------
# CLI: procesa vídeo + tracks.jsonl -> tracks_2d.jsonl (+ vídeo radar opcional)
# --------------------------------------------------------------------------

def _iter_tracks_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def process(source_video: Path, tracks_path: Path, output_path: Path,
            calibrator: FieldCalibrator, render_video_path: Optional[Path] = None,
            stride: int = 1) -> None:
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")

    renderer = PitchRenderer() if render_video_path else None
    writer = None
    if render_video_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(
            str(render_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps / max(stride, 1), (renderer.width_px, renderer.height_px)
        )

    tracks_by_frame = {}
    for rec in _iter_tracks_jsonl(tracks_path):
        tracks_by_frame[rec["frame_idx"]] = rec

    out_f = open(output_path, "w")
    frame_idx = 0
    last_H = None
    n_ok, n_fail = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            rec = tracks_by_frame.get(frame_idx)
            if rec is not None:
                result = calibrator.estimate(frame)
                if result.ok:
                    last_H = result.homography
                    n_ok += 1
                else:
                    n_fail += 1
                    logger.debug("Frame %d: homografía no resuelta (%d keypoints)",
                                 frame_idx, result.num_keypoints)

                anchors = np.array([t["anchor_xy"] for t in rec.get("tracks", [])], dtype=float)
                anchors = anchors.reshape(-1, 2) if anchors.size else anchors.reshape(0, 2)

                field_xy = []
                if last_H is not None and anchors.shape[0] > 0:
                    field_xy = calibrator.project_points(anchors, last_H).tolist()

                out_rec = dict(rec)
                out_rec["homography_ok"] = last_H is not None
                out_rec["num_calibration_keypoints"] = result.num_keypoints if rec is not None else None
                for t, fxy in zip(out_rec.get("tracks", []), field_xy):
                    t["field_xy"] = fxy

                ball = rec.get("ball")
                if ball is not None and last_H is not None:
                    bx, by = calibrator.project_point(tuple(ball["center_xy"]), last_H)
                    ball_out = dict(ball)
                    ball_out["field_xy"] = [bx, by]
                    out_rec["ball"] = ball_out

                out_f.write(json.dumps(out_rec) + "\n")

                if writer is not None:
                    pitch_img = renderer.blank_pitch()
                    if field_xy:
                        pitch_img = renderer.draw_points(pitch_img, np.array(field_xy), color=(0, 0, 255))
                    if "field_xy" in out_rec.get("ball", {}):
                        pitch_img = renderer.draw_points(
                            pitch_img, np.array([out_rec["ball"]["field_xy"]]), color=(0, 255, 255), radius=4
                        )
                    writer.write(pitch_img)

        frame_idx += 1

    cap.release()
    out_f.close()
    if writer is not None:
        writer.release()

    logger.info("Calibración: %d frames OK, %d frames sin homografía válida", n_ok, n_fail)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Homografía por frame + proyección 2D de tracking")
    parser.add_argument("--source", required=True, type=Path, help="Vídeo original (partido.mp4)")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl de core/tracking.py")
    parser.add_argument("--output", required=True, type=Path, help="Salida: tracks_2d.jsonl")
    parser.add_argument("--pnlcalib-repo", required=True, type=Path,
                         help="Path al código fuente de PnLCalib (model/, utils/, config/)")
    parser.add_argument("--weights-kp", required=True, type=Path)
    parser.add_argument("--weights-line", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=1, help="Procesar 1 de cada N frames")
    parser.add_argument("--render-video", type=Path, default=None,
                         help="Si se indica, genera un vídeo radar-view (bird's-eye)")
    parser.add_argument("--pnl-refine", action="store_true")
    args = parser.parse_args()

    config = PitchCalibrationConfig(
        pnlcalib_repo_path=args.pnlcalib_repo,
        weights_kp=args.weights_kp,
        weights_line=args.weights_line,
        device=args.device,
        pnl_refine=args.pnl_refine,
    )
    calibrator = FieldCalibrator(config)

    process(
        source_video=args.source,
        tracks_path=args.tracks,
        output_path=args.output,
        calibrator=calibrator,
        render_video_path=args.render_video,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()