"""
core/ball_actions.py
=====================
Clasificación de acciones de balón (pase, regate, entrada) sobre una ventana
de frames centrada en el jugador que posee el balón, usando un R3D-18
fine-tuned (Kinetics-400 preentrenado).

Complementa a ``core/events.py``: aquel clasifica clips ya recortados en
torno a un evento discreto (gol/falta/córner/...); este clasifica ventanas
DENSAS de ``T=32`` frames centradas en el poseedor del balón (ver
``core/possession.py``), pensado para correr en sliding-window sobre un
partido completo, no sobre clips pre-cortados por evento — de ahí que sea un
módulo separado en vez de ampliar ``EventClassifier`` (ver README.md).

Pipeline: frames + bbox del poseedor -> recorte estático
(``compute_crop_box``) -> ``R3D_18_Weights.KINETICS400_V1.transforms()`` ->
R3D-18 -> label.

El checkpoint (``models/r3d18_ball_actions.pt`` por defecto) se entrena con
``training/train_ball_actions.py`` sobre los clips que genera
``scripts/extract_ball_action_clips.py`` (mismo formato de recorte estático,
ver ``compute_crop_box`` — reutilizada por ambos para no divergir). Reconoce
6 clases en esta primera fase: ``pass``, ``drive``, ``high_pass``, ``header``,
``throw_in``, ``background`` (``shot``/``goal`` ya los cubre
``core/events.py``, no se duplican aquí; ``tackle``/``cross``/
``ball_player_block`` quedan fuera por ahora — clases minoritarias o con
riesgo de confusión visual, ver ``scripts/extract_ball_action_clips.py``).
Las clases reales las decide el checkpoint cargado (lista ``classes``
guardada en el ``.pt``), no están hardcodeadas aquí — si se reintroduce
alguna más adelante no hace falta tocar este fichero.

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.ball_actions --source clip.mp4
    python -m core.ball_actions --source clips_folder/ --output predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH: Path = _PROJECT_ROOT / "models" / "r3d18_ball_actions.pt"
DEFAULT_NUM_FRAMES: int = 32
DEFAULT_PADDING_FACTOR: float = 2.5

_BACKGROUND_LABEL: str = "background"


# --------------------------------------------------------------------------- #
#  Registro serializable (telemetría)
# --------------------------------------------------------------------------- #

@dataclass
class BallActionPrediction:
    """Predicción del clasificador de acciones de balón para una ventana."""

    source: str                        # ruta del clip, o "<window>" si viene de predict_window
    label: str                         # p.ej. "pass" | "drive" | "background" (según checkpoint)
    confidence: float                  # probabilidad de la clase predicha
    probabilities: dict[str, float]    # probabilidad de cada clase del checkpoint
    is_background: bool                # True si label == "background"

    def to_telemetry(self) -> dict:
        """Registro serializable para el JSON/JSONL de telemetría."""
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Recorte centrado en el poseedor (compartido con scripts/extract_ball_action_clips.py)
# --------------------------------------------------------------------------- #

def compute_crop_box(
    bbox: tuple[float, float, float, float],
    padding_factor: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Expande `bbox` (xyxy) por `padding_factor` centrado en su centro, recortado
    a los límites del frame. Único punto de esta lógica: tanto la extracción de
    dataset como la inferencia en vivo (`predict_window`) deben recortar igual,
    o el modelo verá un distribution shift entre train e inferencia."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * padding_factor / 2
    half_h = (y2 - y1) * padding_factor / 2

    left = max(0, int(cx - half_w))
    top = max(0, int(cy - half_h))
    right = min(frame_w, int(cx + half_w))
    bottom = min(frame_h, int(cy + half_h))
    if right <= left:
        right = min(frame_w, left + 1)
    if bottom <= top:
        bottom = min(frame_h, top + 1)
    return left, top, right, bottom


def expand_with_ball_trajectory(
    bbox: tuple[float, float, float, float],
    records_by_frame: dict[int, dict],
    start_frame: int,
    num_frames: int,
) -> tuple[float, float, float, float]:
    """Expande `bbox` (referencia del poseedor/balón en el frame del evento)
    para que también cubra la posición del balón en TODOS los frames de la
    ventana, no solo en el instante del evento.

    Necesario porque el recorte final es estático (mismo bbox para los 32
    frames, ver ``compute_crop_box``): si el balón se aleja del poseedor dentro
    de la ventana — el caso típico de un ``PASS``, donde el balón sale disparado
    hacia otro jugador justo después del evento — un recorte centrado solo en el
    poseedor en el frame del evento pierde el balón de vista a mitad de clip.
    Devuelve la unión (bounding box mínimo que contiene todo) del `bbox` de
    partida y el bbox del balón en cada frame donde esté detectado.

    Vive aquí, junto a ``compute_crop_box``, por el mismo motivo que aquella:
    es lógica de recorte que DEBEN compartir la extracción de dataset
    (``scripts/extract_ball_action_clips.py``) y la inferencia en vivo
    (``analytics/engine.py``). Medido sobre un partido de 20 min: sin esta
    expansión el recorte de inferencia era 12.7× más pequeño en área (mediana)
    que el de entrenamiento y el balón se salía de él en el 92% de las
    ventanas.

    ``records_by_frame`` es ``{frame_idx: record}`` con la forma de
    ``FrameTrackingResult.to_telemetry()`` (o su versión con posesión); los
    frames ausentes simplemente no aportan.
    """
    x1, y1, x2, y2 = bbox
    for frame_idx in range(start_frame, start_frame + num_frames):
        record = records_by_frame.get(frame_idx)
        ball = record.get("ball") if record else None
        if ball is None:
            continue
        bx1, by1, bx2, by2 = ball["xyxy"]
        x1, y1 = min(x1, bx1), min(y1, by1)
        x2, y2 = max(x2, bx2), max(y2, by2)
    return x1, y1, x2, y2


def ensure_num_frames(frames: Sequence[np.ndarray], num_frames: int) -> list[np.ndarray]:
    """Ajusta una lista de frames a exactamente `num_frames` (muestreo uniforme
    si sobran, repite el último si faltan) — robustez frente a clips que no
    midan exactamente T=32 (p.ej. cortados por el final del vídeo)."""
    n = len(frames)
    if n == num_frames:
        return list(frames)
    if n == 0:
        raise ValueError("No hay frames para clasificar")
    if n > num_frames:
        idx = np.linspace(0, n - 1, num=num_frames).round().astype(int)
        return [frames[i] for i in idx]
    return list(frames) + [frames[-1]] * (num_frames - n)


def read_clip_frames(video_path: Path | str) -> list[np.ndarray]:
    """Decodifica un .mp4 entero a una lista de frames RGB (uint8, HWC) vía PyAV.

    ``torchvision.io`` eliminó ``read_video``/``VideoReader`` en versiones
    recientes (0.26 instalada en este proyecto ya no los tiene, solo quedan
    funciones de imagen) — ``av`` es dependencia ya existente del proyecto
    (usada también en ``core/events.py::sample_frames``), así que se reutiliza
    aquí en vez de añadir una dependencia nueva. Único punto de lectura de
    clip compartido entre ``training/train_ball_actions.py`` y este módulo.
    """
    import av

    container = av.open(str(video_path))
    try:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()
    return frames


# --------------------------------------------------------------------------- #
#  Clasificador
# --------------------------------------------------------------------------- #

class BallActionClassifier:
    """Clasifica ventanas centradas en poseedor en pass/drive/background (u otras clases del checkpoint)
    vía R3D-18 fine-tuned.

    Carga el modelo una única vez en el constructor; ``predict_clip`` /
    ``predict_window`` se pueden llamar repetidamente (pensado para
    integrarse en ``main.py``/``analytics/engine.py`` o en sliding-window
    sobre un partido completo).

    Args:
        checkpoint_path: ``.pt`` con ``{"model_state_dict": ..., "classes": [...]}``,
            generado por ``training/train_ball_actions.py``.
        device: "cuda" | "cpu" (None -> cuda si está disponible).
        num_frames: frames por ventana (debe coincidir con el usado en
            extracción/entrenamiento, 32 por defecto).
        padding_factor: expansión del bbox del poseedor antes de recortar
            (debe coincidir con el usado en ``scripts/extract_ball_action_clips.py``).
    """

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
        device: Optional[str] = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        padding_factor: float = DEFAULT_PADDING_FACTOR,
    ) -> None:
        import torch
        from torchvision.models.video import R3D_18_Weights, r3d_18

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint de acciones de balón no encontrado: {checkpoint_path}")

        self._torch = torch
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.padding_factor = padding_factor

        logger.info("Cargando checkpoint de acciones de balón desde %s ...", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.classes: list[str] = checkpoint["classes"]

        model = r3d_18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device).eval()
        self.model = model
        self.transform = R3D_18_Weights.KINETICS400_V1.transforms()
        logger.info("Clases: %s", ", ".join(self.classes))

    # ------------------------------------------------------------------ #

    def predict_clip(self, video_path: Path | str) -> BallActionPrediction:
        """Clasifica un clip .mp4 ya recortado (p.ej. por
        ``scripts/extract_ball_action_clips.py``): se asume ya centrado en el
        poseedor, no se vuelve a recortar."""
        video_path = Path(video_path)
        frames = ensure_num_frames(read_clip_frames(video_path), self.num_frames)
        video = self._torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # (T, C, H, W)

        return self._predict_tensor(str(video_path), video)

    def predict_window(
        self,
        frames: Sequence[np.ndarray],
        possessor_bbox: tuple[float, float, float, float],
    ) -> BallActionPrediction:
        """Clasifica una ventana de frames BGR (p.ej. leídos en vivo de un
        partido) recortando en torno a `possessor_bbox` (xyxy en píxeles del
        frame original) con el mismo `compute_crop_box` que la extracción de
        dataset offline."""
        frames = ensure_num_frames(frames, self.num_frames)
        h, w = frames[0].shape[:2]
        left, top, right, bottom = compute_crop_box(possessor_bbox, self.padding_factor, w, h)

        cropped_rgb = [cv2.cvtColor(f[top:bottom, left:right], cv2.COLOR_BGR2RGB) for f in frames]
        video = self._torch.from_numpy(np.stack(cropped_rgb)).permute(0, 3, 1, 2)  # (T, C, H, W)

        return self._predict_tensor("<window>", video)

    def predict_many_clips(
        self, video_paths: Sequence[Path | str]
    ) -> Iterator[BallActionPrediction]:
        """Clasifica varios clips en secuencia (generador, apto para pipelines)."""
        for video_path in video_paths:
            yield self.predict_clip(video_path)

    # ------------------------------------------------------------------ #

    def _predict_tensor(self, source: str, video: "torch.Tensor") -> BallActionPrediction:
        torch = self._torch
        clip = self.transform(video).unsqueeze(0).to(self.device)  # -> (1, C, T, H, W)
        with torch.no_grad():
            logits = self.model(clip)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        label = self.classes[pred_idx]
        probabilities = {cls: float(p) for cls, p in zip(self.classes, probs)}

        return BallActionPrediction(
            source=source,
            label=label,
            confidence=float(probs[pred_idx]),
            probabilities=probabilities,
            is_background=(label == _BACKGROUND_LABEL),
        )


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def _resolve_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("**/*.mp4"))
    return [source]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificación de acciones de balón (pass/drive/background, R3D-18)"
    )
    parser.add_argument("--source", required=True, help="Clip .mp4 (ya recortado) o carpeta con .mp4")
    parser.add_argument("--output", default=None, help="JSONL de salida (si no, solo stdout)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = Path(args.source)
    if not source.exists():
        parser.error(f"No existe: {source}")
    video_paths = _resolve_sources(source)
    if not video_paths:
        parser.error(f"Ningún .mp4 encontrado en: {source}")

    classifier = BallActionClassifier(checkpoint_path=args.checkpoint, device=args.device)

    out_f = open(args.output, "w") if args.output else None
    try:
        for video_path in video_paths:
            try:
                prediction = classifier.predict_clip(video_path)
            except Exception:
                logger.exception("Error procesando %s", video_path)
                continue

            tag = "·bg·" if prediction.is_background else "ACTION"
            logger.info(
                "[%s] %-40s -> %-12s (%.1f%%)",
                tag, video_path.name, prediction.label, prediction.confidence * 100,
            )
            if out_f:
                out_f.write(json.dumps(prediction.to_telemetry()) + "\n")
    finally:
        if out_f:
            out_f.close()

    if args.output:
        logger.info("Predicciones escritas en %s", args.output)


if __name__ == "__main__":
    main()
