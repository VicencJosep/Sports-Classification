"""
core/tracking.py
================
Detección y tracking multi-objeto para vídeo de fútbol.

Pipeline: frames (BGR) -> Detector (inyectable) -> sv.Detections -> ByteTrack.

Hay tres detectores intercambiables (misma interfaz `Detector`):

* ``SoccerMasterDetector``: usa el head ``SoccerNetGSR_Detection`` del repo
  Soccer_Master. Es específico de fútbol y devuelve, además de la caja, el rol
  (player / goalkeeper / referee / other / ball) y el número de dorsal.
  Trabaja sobre clips de NUM_FRAMES frames (30, fijado por el checkpoint), por
  lo que declara ``chunk_size > 1`` y el tracker le entrega los frames en lotes.
  Limitación medida: casi nunca emite la clase "ball" (el dataset GSR se centra
  en personas).

* ``YoloDetector``: YOLO11x genérico COCO (person + sports ball). Sin roles ni
  dorsales, pero no requiere los checkpoints de Soccer_Master. ``chunk_size=1``.

* ``HybridDetector`` (por defecto): personas de SoccerMaster + balón de YOLO.

Limitación conocida del balón: en planos de broadcast lejanos el balón mide
~8 px y ni YOLO11x lo detecta de forma fiable. Para métricas de posesión serias
habrá que añadir inferencia por teselas (sv.InferenceSlicer) o un modelo
fine-tuned de balón; queda como iteración futura.

Ambos emiten `sv.Detections` en coordenadas de píxel del frame ORIGINAL, con
`class_id` según la taxonomía de Soccer_Master (`ROLE_TO_CLASS_ID`) y los campos
extra ``role`` / ``jersey`` en ``Detections.data``.

El balón NO se pasa por ByteTrack: es un objeto pequeño y rápido que el tracker
pierde constantemente, así que se resuelve por frame con la detección de mayor
confianza. El resto de personas sí se trackean con IDs persistentes.

Salida por frame: `FrameTrackingResult`, que transporta tanto los objetos
`sv.Detections` (para los anotadores de utils/visualizers.py) como los registros
serializables que consumirá analytics/engine.py para construir el JSON.

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.tracking --source partido.mp4 --output tracks.jsonl
    python -m core.tracking --source partido.mp4 --detector yolo
"""

from __future__ import annotations

import collections
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator, Iterator, Optional, Protocol, Sequence

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Taxonomía canónica de clases (tomada de Soccer_Master:
#  data/soccernet_gsr_detection.py::role_mapping)
# --------------------------------------------------------------------------- #

ROLE_TO_CLASS_ID: dict[str, int] = {
    "ball": 0,
    "goalkeeper": 1,
    "other": 2,
    "player": 3,
    "referee": 4,
}
CLASS_ID_TO_ROLE: dict[int, str] = {v: k for k, v in ROLE_TO_CLASS_ID.items()}
BALL_CLASS_ID: int = ROLE_TO_CLASS_ID["ball"]

# Resolución de entrada de YOLO para el balón. Ultralytics usa 640 por defecto,
# que en un 720p deja el balón en 3-5 px. Medido el 11-oct-2026 sobre 2 min de
# un partido juvenil de YouTube a 720p (YOLO11x COCO, "sports ball" >= 0.15):
# 640 -> balón en el 54% de los frames (20 ms/frame), 960 -> 68% (32 ms),
# 1280 -> 70% (40 ms). 1280 es la resolución nativa del 720p: no se inventa
# nada, solo deja de tirar píxeles.
DEFAULT_BALL_IMGSZ: int = 1280

# Pesos del detector de balón: el ajustado con SoccerNet Tracking (una sola
# clase, "ball") si existe; si no, el YOLO11x COCO genérico ("sports ball").
# Medido el 12-oct-2026 sobre 1.200 frames de los clips de test de SoccerNet
# Tracking (TV 1080p, partidos no vistos), conf >= 0.15: COCO a 1280 px recall
# 67% / precisión 48%; ajustado a 1280 px recall 80% / precisión 77%. En dos
# juveniles de YouTube el ajustado ve el balón en el 85-98% de los frames
# frente al 63-73% de COCO (output/ball_experiments/CONCLUSIONES.md). Se
# entrena con scripts/ball_detector/{prepare_soccernet_tracking_ball,train_ball_yolo}.py.
# NO es el detector por defecto: en el partido completo del Espanyol (13-oct-2026)
# vio balón en el 88% de los frames de juego, pero la mejor detección saltaba a
# botas, farolas y árboles del fondo (23% de frames consecutivos con saltos de
# >80 px, 412 cambios de poseedor por minuto frente a 80 con COCO a 1280). Hace
# falta coherencia temporal en _resolve_ball antes de usarlo; se activa con
# build_detector(ball_model_path=DEFAULT_BALL_MODEL_PATH) o --ball-model.
DEFAULT_BALL_MODEL_PATH: Path = Path(__file__).resolve().parents[1] / "models" / "ball" / "yolo11s_snt_ball.pt"

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_SM_DIR: Path = _PROJECT_ROOT / "Soccer_Master" / "SoccerMaster"


@contextmanager
def _repo_import_context(repo_dir: Path) -> Iterator[None]:
    """Aísla los imports top-level del repo SoccerMaster durante la carga del modelo.

    El repo define paquetes namespace (sin ``__init__.py``) llamados ``utils``,
    ``data``, ``models`` y ``configs``. El primero choca con el paquete regular
    ``utils/`` de este proyecto y, como los paquetes regulares tienen prioridad
    sobre los namespace *independientemente del orden de sys.path*, no basta con
    anteponer el repo: hay que retirar temporalmente la raíz del proyecto de
    ``sys.path`` y el ``utils`` del proyecto de ``sys.modules``. Además, los
    YAML del repo usan rutas relativas al CWD (``./configs/default.yaml``,
    ``./pretrained_models/...``), así que también se hace ``chdir`` al repo.

    Al salir se restaura todo; el repo queda al FINAL de sys.path para que sus
    eventuales imports perezosos no colisionantes (models/, data/) sigan
    resolviendo. Es seguro porque el repo no hace imports perezosos de `utils`
    (verificado) y los módulos ya cargados conservan sus referencias.
    """
    repo_path = str(repo_dir)
    saved_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".", 1)[0] == "utils"
    }
    saved_path = list(sys.path)
    saved_cwd = os.getcwd()

    def _is_project_root(entry: str) -> bool:
        try:
            return Path(entry or saved_cwd).resolve() == _PROJECT_ROOT
        except OSError:
            return False

    sys.path = [p for p in sys.path if not _is_project_root(p)]
    sys.path.insert(0, repo_path)
    os.chdir(repo_path)
    try:
        yield
    finally:
        os.chdir(saved_cwd)
        sys.path = saved_path
        if repo_path not in sys.path:
            sys.path.append(repo_path)
        for name in list(sys.modules):
            if name.split(".", 1)[0] == "utils":
                sys.modules.pop(name)
        sys.modules.update(saved_modules)


# --------------------------------------------------------------------------- #
#  Registros serializables (telemetría)
# --------------------------------------------------------------------------- #

@dataclass
class BallRecord:
    """Posición del balón en un frame (coordenadas de píxel)."""

    xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    confidence: float


@dataclass
class TrackRecord:
    """Una persona (jugador/portero/árbitro) trackeada en un frame."""

    tracker_id: int
    xyxy: tuple[float, float, float, float]
    anchor_xy: tuple[float, float]  # bottom-center: punto de apoyo que se
                                    # proyectará con la homografía (core/homography.py)
    confidence: float
    class_id: int
    role: str                       # "player" | "goalkeeper" | "referee" | "other"
    jersey: Optional[int] = None    # dorsal (solo SoccerMasterDetector)


@dataclass
class FrameTrackingResult:
    """Resultado completo del tracking para un frame."""

    frame_idx: int
    timestamp_s: float
    detections: sv.Detections       # personas con tracker_id (para visualizar)
    ball_detections: sv.Detections  # balón (0 o 1 detección)
    frame: np.ndarray               # frame BGR original (para utils/visualizers.py)
    tracks: list[TrackRecord] = field(default_factory=list)
    ball: Optional[BallRecord] = None

    def to_telemetry(self) -> dict:
        """Registro serializable para el JSON de telemetría (analytics/engine.py)."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 3),
            "tracks": [asdict(t) for t in self.tracks],
            "ball": asdict(self.ball) if self.ball else None,
        }


# --------------------------------------------------------------------------- #
#  Interfaz de detección (inyección de dependencias)
# --------------------------------------------------------------------------- #

class Detector(Protocol):
    """Contrato que debe cumplir cualquier detector inyectado en PlayerTracker.

    ``detect`` recibe una secuencia de frames BGR consecutivos (longitud entre 1
    y ``chunk_size``) y devuelve una lista paralela de `sv.Detections` en píxeles
    del frame original, con ``class_id`` según `ROLE_TO_CLASS_ID` y las claves
    ``role`` (np.ndarray de str) y ``jersey`` (np.ndarray de object) en ``data``.
    """

    @property
    def chunk_size(self) -> int:
        """Nº de frames que el detector procesa por lote (1 = frame a frame)."""
        ...

    def detect(self, frames: Sequence[np.ndarray]) -> list[sv.Detections]:
        ...


def _with_role_data(
    detections: sv.Detections,
    roles: list[str],
    jerseys: list[Optional[int]],
) -> sv.Detections:
    """Adjunta role/jersey a Detections.data (arrays paralelos a las cajas)."""
    detections.data["role"] = np.asarray(roles, dtype=object)
    detections.data["jersey"] = np.asarray(jerseys, dtype=object)
    return detections


def _empty_detections() -> sv.Detections:
    empty = sv.Detections.empty()
    return _with_role_data(empty, [], [])


class YoloDetector:
    """Detector genérico YOLO11x (COCO): person -> "player", sports ball -> "ball".

    No distingue porteros/árbitros ni lee dorsales; útil como fallback ligero
    cuando no están disponibles los checkpoints de Soccer_Master.

    Args:
        model_path: pesos de Ultralytics (se descargan solos si no existen).
        device: "cuda", "cuda:0", "cpu"... None deja elegir a Ultralytics.
        confidence: umbral mínimo global de inferencia (el filtrado fino por
            clase lo hace PlayerTracker).
        imgsz: lado mayor de la imagen que ve YOLO (ver ``DEFAULT_BALL_IMGSZ``:
            a 640 el balón de un 720p casi desaparece).
        ball_class_id / person_class_id: ids de clase del checkpoint. Por defecto
            los de COCO (32 / 0); para un detector de balón de una sola clase
            (``DEFAULT_BALL_MODEL_PATH``) son 0 / None (sin personas).
    """

    _COCO_PERSON: int = 0
    _COCO_BALL: int = 32

    def __init__(
        self,
        model_path: str = "yolo11x.pt",
        device: Optional[str] = None,
        confidence: float = 0.1,
        imgsz: int = DEFAULT_BALL_IMGSZ,
        ball_class_id: int = _COCO_BALL,
        person_class_id: Optional[int] = _COCO_PERSON,
    ) -> None:
        from ultralytics import YOLO  # import perezoso: solo si se usa este detector

        logger.info("Cargando YOLO %s (imgsz=%d, clase balón=%d, clase persona=%s) ...",
                    model_path, imgsz, ball_class_id, person_class_id)
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)
        self.confidence = confidence
        self.imgsz = imgsz
        self.ball_class_id = ball_class_id
        self.person_class_id = person_class_id
        self._classes = [ball_class_id] if person_class_id is None else [person_class_id, ball_class_id]

    @property
    def chunk_size(self) -> int:
        return 1

    def detect(self, frames: Sequence[np.ndarray]) -> list[sv.Detections]:
        results: list[sv.Detections] = []
        for frame in frames:
            raw = self.model(
                frame,
                conf=self.confidence,
                classes=self._classes,
                imgsz=self.imgsz,
                verbose=False,
            )[0]
            dets = sv.Detections.from_ultralytics(raw)
            roles = [
                "ball" if cid == self.ball_class_id else "player"
                for cid in dets.class_id
            ]
            dets.class_id = np.asarray(
                [ROLE_TO_CLASS_ID[r] for r in roles], dtype=int
            )
            # sv.Detections.from_ultralytics() añade su propia clave "class_name"
            # a .data; se descarta aquí para que el esquema de .data (solo
            # role/jersey) coincida con el de SoccerMasterDetector — si no,
            # HybridDetector.detect() no puede fusionar personas+balón
            # (sv.Detections.merge exige las mismas claves en todos los .data).
            dets.data = {}
            results.append(_with_role_data(dets, roles, [None] * len(dets)))
        return results


class SoccerMasterDetector:
    """Detector específico de fútbol basado en el repo Soccer_Master.

    Envuelve `MultiTaskingModel` con el head ``SoccerNetGSR_Detection``
    (Deformable-DETR sobre backbone SigLIP), que clasifica cada caja como
    player/goalkeeper/referee/other/ball y predice el dorsal.

    El modelo es temporal: procesa clips de ``NUM_FRAMES`` frames (30 según el
    checkpoint publicado) redimensionados a 512x512. Este wrapper acepta lotes
    parciales (rellena repitiendo el último frame, como hace infer_mp4.py del
    propio repo) y devuelve las cajas reescaladas al tamaño original del frame.

    Args:
        repo_dir: raíz del repo SoccerMaster (contiene models/, configs/, ...).
        config_path: yaml del modelo (por defecto configs/pretrain.yaml).
        checkpoint_dir: carpeta con backbone.pt y SoccerNetGSR_Detection.pt.
        device: "cuda" | "cpu" (None -> cuda si está disponible).
        score_threshold: umbral mínimo del PostProcess DETR. Se deja bajo a
            propósito: el filtrado fino por clase (personas vs balón) lo hace
            PlayerTracker, y el balón necesita un umbral laxo.
        fp16: autocast float16 en GPU (más rápido; activado si hay CUDA).
    """

    def __init__(
        self,
        repo_dir: Path | str = _DEFAULT_SM_DIR,
        config_path: Optional[Path | str] = None,
        checkpoint_dir: Optional[Path | str] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.1,
        fp16: Optional[bool] = None,
    ) -> None:
        import torch

        repo_dir = Path(repo_dir).resolve()
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"Repo Soccer_Master no encontrado: {repo_dir}")

        self._torch = torch
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.fp16 = fp16 if fp16 is not None else self.device.startswith("cuda")

        config_path = str(config_path or repo_dir / "configs" / "pretrain.yaml")
        checkpoint_dir = str(
            checkpoint_dir or repo_dir / "pretrained_models" / "SoccerMaster"
        )

        logger.info("Cargando SoccerMaster desde %s ...", checkpoint_dir)
        # Toda la carga (imports del repo + construcción + checkpoint) ocurre
        # dentro del contexto aislado: ver docstring de _repo_import_context.
        with _repo_import_context(repo_dir):
            from configs.util import load_super_config, yaml_to_dict
            from data.soccernet_gsr_detection import jn_mapping, role_mapping
            from models.deformable_detr.deformable_detr import PostProcess
            from models.multi_task import MultiTaskingModel

            cfg = yaml_to_dict(config_path)
            cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
            self.model = MultiTaskingModel(config=cfg)
            self.model.load_checkpoint(checkpoint_dir, logger=None, load_heads=True)
        self.model.eval().to(self.device)

        self._num_frames: int = int(cfg["NUM_FRAMES"])
        self._image_size: int = int(cfg.get("AUG_MAX_SIZE", 512))
        self._postprocess = PostProcess()
        self._role_id_to_name: dict[int, str] = {
            v: k for k, v in role_mapping.items() if k is not None
        }
        self._jn_id_to_number: dict[int, str] = {
            v: k for k, v in jn_mapping.items() if k is not None
        }
        logger.info(
            "SoccerMaster listo: clip=%d frames, input=%dx%d, device=%s, fp16=%s",
            self._num_frames, self._image_size, self._image_size, self.device, self.fp16,
        )

    @property
    def chunk_size(self) -> int:
        return self._num_frames

    # ------------------------------------------------------------------ #

    def detect(self, frames: Sequence[np.ndarray]) -> list[sv.Detections]:
        """Detecta sobre un clip de 1..chunk_size frames BGR consecutivos."""
        torch = self._torch
        n_real = len(frames)
        if n_real == 0:
            return []

        orig_h, orig_w = frames[0].shape[:2]
        tensors = [self._preprocess(f) for f in frames]
        while len(tensors) < self._num_frames:  # padding del último clip parcial
            tensors.append(tensors[-1])
        images = torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)  # [1,T,3,H,W]

        with torch.no_grad():
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(
                        images, dataset_name="SoccerNetGSR_Detection", metas=None
                    )
            else:
                outputs = self.model(
                    images, dataset_name="SoccerNetGSR_Detection", metas=None
                )

        det = outputs["SoccerNetGSR_Detection"]
        pred_logits = det["pred_logits"]  # [1, T, Q, C]
        pred_boxes = det["pred_boxes"]    # [1, T, Q, 4]
        bs, t = pred_logits.shape[:2]
        flat: dict = {
            "pred_logits": pred_logits.reshape(bs * t, *pred_logits.shape[2:]),
            "pred_boxes": pred_boxes.reshape(bs * t, *pred_boxes.shape[2:]),
        }
        for key in ("pred_roles", "pred_jn_holistic", "pred_digit_head", "pred_digit_tail"):
            if key in det:
                flat[key] = det[key].reshape(bs * t, *det[key].shape[2:])

        # PostProcess escala las cajas normalizadas a [h, w]: pasando el tamaño
        # ORIGINAL del frame las recuperamos directamente en píxeles reales.
        target_sizes = torch.tensor([[orig_h, orig_w]] * (bs * t), device=self.device)
        results = self._postprocess(flat, target_sizes)

        return [self._to_sv(results[i]) for i in range(n_real)]

    # ------------------------------------------------------------------ #

    def _preprocess(self, frame_bgr: np.ndarray) -> "torch.Tensor":  # noqa: F821
        """BGR -> tensor C,H,W normalizado como en el entrenamiento (SigLIP)."""
        torch = self._torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self._image_size, self._image_size), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = (tensor - 0.5) / 0.5  # AUG_MEAN/AUG_STD = 0.5
        return tensor.permute(2, 0, 1)

    def _to_sv(self, res: dict) -> sv.Detections:
        """Convierte la salida del PostProcess DETR de un frame a sv.Detections."""
        keep = res["scores"] >= self.score_threshold
        boxes = res["boxes"][keep].cpu().numpy().astype(float)
        scores = res["scores"][keep].cpu().numpy().astype(float)
        if len(boxes) == 0:
            return _empty_detections()

        if "roles" in res:
            roles = [
                self._role_id_to_name.get(int(r), "other")
                for r in res["roles"][keep].cpu().numpy()
            ]
        else:
            roles = ["player"] * len(boxes)

        jerseys: list[Optional[int]]
        if "jersey" in res:
            raw = [
                self._jn_id_to_number.get(int(j))
                for j in res["jersey"][keep].cpu().numpy()
            ]
            jerseys = [int(j) if j is not None else None for j in raw]
        else:
            jerseys = [None] * len(boxes)

        dets = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=np.asarray(
                [ROLE_TO_CLASS_ID.get(r, ROLE_TO_CLASS_ID["other"]) for r in roles],
                dtype=int,
            ),
        )
        return _with_role_data(dets, roles, jerseys)


class HybridDetector:
    """Combina dos detectores: personas de uno y balón del otro.

    Motivación (medido sobre clips reales): el head SoccerNetGSR de
    SoccerMaster prácticamente nunca emite la clase "ball" (el dataset GSR se
    centra en personas), mientras que YOLO sí puede detectar el balón (clase
    COCO "sports ball") pero no distingue roles ni dorsales. Este detector se
    queda con lo mejor de cada uno.

    Args:
        person_detector: fuente de las detecciones NO-balón (p.ej. SoccerMasterDetector).
        ball_detector: fuente de las detecciones de balón (p.ej. YoloDetector).
    """

    def __init__(self, person_detector: Detector, ball_detector: Detector) -> None:
        self.person_detector = person_detector
        self.ball_detector = ball_detector

    @property
    def chunk_size(self) -> int:
        """El lote lo dicta el detector de personas (el temporal, si lo hay)."""
        return self.person_detector.chunk_size

    def detect(self, frames: Sequence[np.ndarray]) -> list[sv.Detections]:
        person_dets = self.person_detector.detect(frames)
        ball_dets = self.ball_detector.detect(frames)
        merged: list[sv.Detections] = []
        for persons, balls in zip(person_dets, ball_dets):
            persons = persons[persons.class_id != BALL_CLASS_ID]
            balls = balls[balls.class_id == BALL_CLASS_ID]
            merged.append(sv.Detections.merge([persons, balls]))
        return merged


# --------------------------------------------------------------------------- #
#  Tracker
# --------------------------------------------------------------------------- #

class StaticBallFilter:
    """Descarta detecciones de balón clavadas en el mismo píxel durante mucho tiempo.

    Un balón real con una cámara que panea no se queda quieto en píxeles 20-30 s
    seguidos; el escudo del marcador, un logo o un gráfico, sí. Medido el
    12-oct-2026 con el detector ajustado (SoccerNet Tracking) en un juvenil de
    YouTube: el escudo del marcador superaba al balón real como mejor detección
    en el 45-80% de los frames (conf ~0,5), lo que habría llevado la posesión
    al marcador en vez de al campo.

    Guarda una ventana de ``window_frames`` frames con las celdas (``cell_px``
    px) donde hubo alguna detección, dilatadas a sus 8 vecinas al registrarlas
    (así una detección que baila entre dos celdas cuenta en las dos y no se
    escapa por partir su presencia); una detección es estática si su celda ha
    tenido detección en al menos ``min_presence`` de los frames de la ventana.
    No filtra nada hasta tener ``min_history_frames`` frames de historial.
    """

    def __init__(
        self,
        window_frames: int = 750,
        min_presence: float = 0.7,
        cell_px: int = 8,
        min_history_frames: int = 150,
    ) -> None:
        self.window_frames = window_frames
        self.min_presence = min_presence
        self.cell_px = cell_px
        self.min_history_frames = min_history_frames
        self._history: collections.deque = collections.deque()
        self._counts: collections.Counter = collections.Counter()

    def reset(self) -> None:
        self._history.clear()
        self._counts.clear()

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x // self.cell_px), int(y // self.cell_px))

    def update(self, centers: Sequence[tuple[float, float]]) -> None:
        """Registra las detecciones candidatas de ESTE frame (llamar en todos
        los frames, también sin candidatas, para que la ventana avance)."""
        cells: set[tuple[int, int]] = set()
        for x, y in centers:
            cx, cy = self._cell(x, y)
            cells.update((cx + dx, cy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        frozen = frozenset(cells)
        self._history.append(frozen)
        self._counts.update(frozen)
        while len(self._history) > self.window_frames:
            self._counts.subtract(self._history.popleft())

    def is_static(self, x: float, y: float) -> bool:
        n = len(self._history)
        if n < self.min_history_frames:
            return False
        return self._counts.get(self._cell(x, y), 0) / n >= self.min_presence


class PlayerTracker:
    """Orquesta Detector (inyectado) + ByteTrack.

    El detector se pasa ya construido (inyección de dependencias), lo que
    permite testear con un stub y alternar YOLO/SoccerMaster sin tocar esta
    clase. El tracker acumula frames hasta ``detector.chunk_size`` y procesa
    el lote, manteniendo el orden temporal que ByteTrack necesita.

    Args:
        detector: implementación de `Detector` (SoccerMasterDetector, YoloDetector...).
        confidence: umbral de confianza para personas.
        ball_confidence: umbral (más laxo) para el balón, que se detecta débil.
        min_box_area: descarta cajas diminutas (público en gradas, falsos positivos).
        track_activation_threshold / lost_track_buffer / minimum_matching_threshold:
            hiperparámetros de ByteTrack (ver docs de supervision).
        frame_rate: fps nominal del vídeo (para el buffer de tracks perdidos).
        static_ball_window_frames / static_ball_min_presence: ver ``StaticBallFilter``
            (descarta "balones" clavados en el mismo píxel: escudos, logos).
    """

    def __init__(
        self,
        detector: Detector,
        confidence: float = 0.3,
        ball_confidence: float = 0.15,
        min_box_area: float = 300.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: Optional[int] = None,
        static_ball_window_frames: int = 750,
        static_ball_min_presence: float = 0.7,
        static_ball_min_history_frames: int = 150,
    ) -> None:
        self.detector = detector
        self.confidence = confidence
        self.ball_confidence = ball_confidence
        self.min_box_area = min_box_area
        self._static_ball = StaticBallFilter(
            static_ball_window_frames, static_ball_min_presence, min_history_frames=static_ball_min_history_frames
        )
        self._track_activation_threshold = track_activation_threshold
        self._lost_track_buffer = lost_track_buffer
        self._minimum_matching_threshold = minimum_matching_threshold
        # frame_rate=None -> auto-detectar del vídeo real en process_video() (evita
        # el desfase silencioso de usar un frame_rate nominal distinto al real del
        # vídeo al dimensionar lost_track_buffer). Si se pasa explícito, se respeta
        # tal cual (p.ej. para process_batch() standalone sin pasar por process_video()).
        self._frame_rate = frame_rate
        self.tracker = self._build_tracker(frame_rate or 30)

    def _build_tracker(self, frame_rate: int) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=self._track_activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    # ------------------------------------------------------------------ #
    #  API por lote de frames
    # ------------------------------------------------------------------ #

    def process_batch(
        self,
        frames: Sequence[np.ndarray],
        frame_indices: Sequence[int],
        timestamps_s: Sequence[float],
    ) -> list[FrameTrackingResult]:
        """Detección + tracking sobre un lote de frames BGR consecutivos.

        El lote debe respetar el orden temporal; ByteTrack se actualiza frame
        a frame internamente para mantener IDs coherentes.
        """
        detections_per_frame = self.detector.detect(frames)
        results: list[FrameTrackingResult] = []
        for dets, frame, frame_idx, ts in zip(
            detections_per_frame, frames, frame_indices, timestamps_s
        ):
            results.append(self._track_frame(dets, frame, frame_idx, ts))
        return results

    def _track_frame(
        self,
        detections: sv.Detections,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
    ) -> FrameTrackingResult:
        persons = self._filter_persons(detections)
        persons = self.tracker.update_with_detections(persons)
        ball_dets, ball_record = self._resolve_ball(detections)

        roles = persons.data.get("role", np.asarray(["player"] * len(persons)))
        jerseys = persons.data.get("jersey", np.asarray([None] * len(persons)))
        tracks = [
            TrackRecord(
                tracker_id=int(persons.tracker_id[i]),
                xyxy=tuple(float(v) for v in persons.xyxy[i]),
                anchor_xy=self._bottom_center(persons.xyxy[i]),
                confidence=float(persons.confidence[i]),
                class_id=int(persons.class_id[i]),
                role=str(roles[i]),
                jersey=int(jerseys[i]) if jerseys[i] is not None else None,
            )
            for i in range(len(persons))
        ]

        return FrameTrackingResult(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            detections=persons,
            ball_detections=ball_dets,
            frame=frame,
            tracks=tracks,
            ball=ball_record,
        )

    # ------------------------------------------------------------------ #
    #  API por vídeo (generador — apto para pipelines asíncronos)
    # ------------------------------------------------------------------ #

    def process_video(
        self,
        source_path: str | Path,
        stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameTrackingResult, None, None]:
        """Recorre el vídeo y cede (yield) un FrameTrackingResult por frame.

        Internamente acumula ``detector.chunk_size`` frames antes de inferir
        (necesario para el modelo temporal de SoccerMaster; con YOLO el lote
        es de 1 y el comportamiento es frame a frame). Al ser un generador,
        main.py puede consumirlo de forma incremental y despachar cada
        resultado a analytics/engine.py o a una cola asíncrona sin cargar el
        vídeo entero en memoria.

        Args:
            stride: procesa 1 de cada N frames (los timestamps siguen siendo reales).
            max_frames: corte para pruebas rápidas.
        """
        source = str(source_path)
        video_info = sv.VideoInfo.from_video_path(source)
        fps = float(video_info.fps or 30)
        if self._frame_rate is None:
            # Reconstruye con el fps real del vídeo (p.ej. 25 en SoccerNet) en vez
            # del fallback de 30 usado en __init__ antes de conocerlo — evita que
            # lost_track_buffer represente una duración real distinta a la prevista.
            self.tracker = self._build_tracker(round(fps))
        else:
            self.tracker.reset()
        self._static_ball.reset()

        logger.info(
            "Procesando %s — %dx%d @ %.1f fps, %s frames (chunk=%d, stride=%d)",
            source, video_info.width, video_info.height, fps,
            video_info.total_frames, self.detector.chunk_size, stride,
        )

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {source}")

        buf_frames: list[np.ndarray] = []
        buf_indices: list[int] = []
        try:
            frame_idx = 0
            emitted = 0
            stop = False
            while not stop:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    buf_frames.append(frame)
                    buf_indices.append(frame_idx)
                    emitted += 1
                    if max_frames is not None and emitted >= max_frames:
                        stop = True
                    if len(buf_frames) == self.detector.chunk_size or stop:
                        yield from self.process_batch(
                            buf_frames, buf_indices, [i / fps for i in buf_indices]
                        )
                        buf_frames, buf_indices = [], []
                frame_idx += 1

            if buf_frames:  # último lote parcial
                yield from self.process_batch(
                    buf_frames, buf_indices, [i / fps for i in buf_indices]
                )
        finally:
            cap.release()

    # ------------------------------------------------------------------ #
    #  Helpers internos
    # ------------------------------------------------------------------ #

    def _filter_persons(self, detections: sv.Detections) -> sv.Detections:
        """No-balón con confianza suficiente y área mínima (fuera público lejano)."""
        if len(detections) == 0:
            return detections
        mask = (
            (detections.class_id != BALL_CLASS_ID)
            & (detections.confidence >= self.confidence)
            & (detections.box_area >= self.min_box_area)
        )
        return detections[mask]

    def _resolve_ball(
        self, detections: sv.Detections
    ) -> tuple[sv.Detections, Optional[BallRecord]]:
        """El balón se resuelve por frame: mejor detección sobre el umbral laxo,
        descartando antes las clavadas en el mismo píxel (``StaticBallFilter``)."""
        if len(detections) == 0:
            self._static_ball.update(())
            return detections, None
        balls = detections[
            (detections.class_id == BALL_CLASS_ID)
            & (detections.confidence >= self.ball_confidence)
        ]
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in balls.xyxy] if len(balls) else []
        self._static_ball.update(centers)
        if len(balls) == 0:
            return balls, None
        keep = np.array([not self._static_ball.is_static(x, y) for x, y in centers], dtype=bool)
        if not keep.any():
            return balls[keep], None
        balls = balls[keep]

        best = balls[int(np.argmax(balls.confidence))]
        x1, y1, x2, y2 = (float(v) for v in best.xyxy[0])
        record = BallRecord(
            xyxy=(x1, y1, x2, y2),
            center_xy=((x1 + x2) / 2, (y1 + y2) / 2),
            confidence=float(best.confidence[0]),
        )
        return best, record

    @staticmethod
    def _bottom_center(xyxy: np.ndarray) -> tuple[float, float]:
        """Punto de apoyo del jugador: centro del borde inferior de la caja."""
        x1, _, x2, y2 = (float(v) for v in xyxy)
        return ((x1 + x2) / 2, y2)


# --------------------------------------------------------------------------- #
#  Factory + ejecución standalone
# --------------------------------------------------------------------------- #

def build_detector(
    name: str,
    device: Optional[str] = None,
    yolo_model_path: str = "yolo11x.pt",
    sm_repo_dir: Path | str = _DEFAULT_SM_DIR,
    ball_imgsz: int = DEFAULT_BALL_IMGSZ,
    ball_model_path: Path | str | None = None,
) -> Detector:
    """Construye el detector por nombre ("hybrid" | "soccermaster" | "yolo").

    "hybrid" (recomendado): personas (rol + dorsal) de SoccerMaster y balón de
    YOLO a ``ball_imgsz`` (resolución nativa del 720p por defecto), porque el
    head GSR de SoccerMaster casi nunca detecta el balón. El balón sale del
    YOLO11x COCO genérico salvo que se pase ``ball_model_path`` (detector de una
    clase, p. ej. ``DEFAULT_BALL_MODEL_PATH``) y ese fichero exista.
    """
    if name == "hybrid":
        if ball_model_path and Path(ball_model_path).exists():
            logger.info("Detector de balón: %s (ajustado, clase única)", ball_model_path)
            ball_detector = YoloDetector(
                model_path=str(ball_model_path), device=device, imgsz=ball_imgsz,
                ball_class_id=0, person_class_id=None,
            )
        else:
            logger.info("Detector de balón: %s (COCO 'sports ball'; no hay pesos ajustados en %s)",
                        yolo_model_path, ball_model_path)
            ball_detector = YoloDetector(model_path=yolo_model_path, device=device, imgsz=ball_imgsz)
        return HybridDetector(
            person_detector=SoccerMasterDetector(repo_dir=sm_repo_dir, device=device),
            ball_detector=ball_detector,
        )
    if name == "soccermaster":
        return SoccerMasterDetector(repo_dir=sm_repo_dir, device=device)
    if name == "yolo":
        return YoloDetector(model_path=yolo_model_path, device=device, imgsz=ball_imgsz)
    raise ValueError(
        f"Detector desconocido: {name!r} (usa 'hybrid', 'soccermaster' o 'yolo')"
    )


def ensure_tracking(video_path: Path, detector_name: str = "hybrid", device: Optional[str] = None) -> Path:
    """Corre PlayerTracker sobre video_path si tracks.jsonl no existe ya (caché).

    Escribe primero a un ``.tmp`` y solo renombra al fichero final si termina
    sin errores — si un intento previo se cortó a mitad (crash/OOM/Ctrl-C), el
    ``.tmp`` queda huérfano pero NUNCA aparenta ser una caché completa, así que
    la siguiente ejecución vuelve a generarla en vez de reutilizar un JSONL
    vacío o a medias (ver incidente: un crash de HybridDetector dejó un
    tracks.jsonl de 0 bytes que esta función daba por bueno, produciendo 0
    clips en todas las extracciones posteriores sin ningún error visible).

    Usado tanto por scripts/extract_ball_action_clips.py (extracción offline de
    clips de entrenamiento) como por analytics/engine.py::analyze_full_match
    (Pase A del flujo de partido completo) — la misma caché sirve para ambos.
    """
    tracks_path = video_path.with_suffix(".tracks.jsonl")
    if tracks_path.exists():
        logger.info("tracks.jsonl ya existe, reutilizando: %s", tracks_path)
        return tracks_path

    logger.info("Corriendo core.tracking (%s) sobre %s ...", detector_name, video_path)
    detector = build_detector(detector_name, device=device)
    tracker = PlayerTracker(detector=detector)
    tmp_path = tracks_path.with_suffix(tracks_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for result in tracker.process_video(video_path):
            f.write(json.dumps(result.to_telemetry()) + "\n")
    tmp_path.rename(tracks_path)
    return tracks_path


def main() -> None:
    import argparse
    from contextlib import ExitStack

    parser = argparse.ArgumentParser(description="Tracking fútbol: detector + ByteTrack")
    parser.add_argument("--source", required=True, help="Ruta al vídeo .mp4")
    parser.add_argument("--output", default="tracks.jsonl", help="Salida JSONL")
    parser.add_argument(
        "--annotated-output", default=None,
        help="Si se indica, escribe además un .mp4 con cajas/IDs/estelas dibujados "
             "(ver utils/visualizers.py)",
    )
    parser.add_argument("--detector", choices=["hybrid", "soccermaster", "yolo"],
                        default="hybrid")
    parser.add_argument("--sm-dir", default=str(_DEFAULT_SM_DIR),
                        help="Raíz del repo SoccerMaster")
    parser.add_argument("--yolo-model", default="yolo11x.pt")
    parser.add_argument("--ball-model", default="",
                        help="pesos de un detector de balón de una clase (solo --detector hybrid), p. ej. "
                             f"{DEFAULT_BALL_MODEL_PATH}; vacío (por defecto) = YOLO11x COCO")
    parser.add_argument("--ball-imgsz", type=int, default=DEFAULT_BALL_IMGSZ,
                        help="resolución de entrada de YOLO para el balón (por defecto "
                             f"{DEFAULT_BALL_IMGSZ}, la nativa de un 720p; 640 es la de Ultralytics)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--frame-rate", type=int, default=None,
        help="fps nominal para ByteTrack (lost_track_buffer). Por defecto (sin "
             "indicar), se auto-detecta del vídeo real en process_video().",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    detector = build_detector(
        args.detector,
        device=args.device,
        yolo_model_path=args.yolo_model,
        sm_repo_dir=args.sm_dir,
        ball_imgsz=args.ball_imgsz,
        ball_model_path=args.ball_model or None,
    )
    tracker = PlayerTracker(detector=detector, frame_rate=args.frame_rate)

    with ExitStack() as stack:
        f = stack.enter_context(open(args.output, "w"))

        sink: Optional[sv.VideoSink] = None
        visualizer = None
        if args.annotated_output:
            from utils.visualizers import TrackingVisualizer  # import perezoso: solo si se pide --annotated-output

            video_info = sv.VideoInfo.from_video_path(args.source)
            video_info.fps = max(1, round(video_info.fps / args.stride))
            sink = stack.enter_context(sv.VideoSink(args.annotated_output, video_info))
            visualizer = TrackingVisualizer()

        for result in tracker.process_video(
            args.source, stride=args.stride, max_frames=args.max_frames
        ):
            f.write(json.dumps(result.to_telemetry()) + "\n")
            if sink is not None:
                sink.write_frame(visualizer.annotate(result.frame, result.detections, result.ball_detections))
            if result.frame_idx % 100 == 0:
                logger.info(
                    "frame %d — %d tracks, balón: %s",
                    result.frame_idx, len(result.tracks),
                    "sí" if result.ball else "no",
                )

    logger.info("Telemetría de tracking escrita en %s", args.output)
    if args.annotated_output:
        logger.info("Vídeo anotado escrito en %s", args.annotated_output)


if __name__ == "__main__":
    main()
