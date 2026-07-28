"""
config.py
=========
Configuración global del pipeline de Sports Analytics.

Centraliza paths e hiperparámetros para que main.py pueda construir los
módulos de core/ por inyección de dependencias sin hardcodear rutas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Paths base
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Repositorio Soccer_Master (detección player/goalkeeper/referee/ball + dorsal)
SOCCER_MASTER_DIR: Path = PROJECT_ROOT / "Soccer_Master" / "SoccerMaster"
SOCCER_MASTER_CONFIG: Path = SOCCER_MASTER_DIR / "configs" / "pretrain.yaml"
SOCCER_MASTER_CHECKPOINTS: Path = SOCCER_MASTER_DIR / "pretrained_models" / "SoccerMaster"

OUTPUTS_DIR: Path = PROJECT_ROOT / "output"


# --------------------------------------------------------------------------- #
#  Hiperparámetros por módulo
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrackingConfig:
    """Hiperparámetros de core/tracking.py (detector + ByteTrack)."""

    detector: str = "hybrid"                # "hybrid" | "soccermaster" | "yolo"
    yolo_model_path: str = "yolo11x.pt"
    device: str | None = None               # None -> auto (cuda si hay)
    confidence: float = 0.3                 # umbral personas
    ball_confidence: float = 0.15           # umbral (laxo) para el balón
    min_box_area: float = 300.0             # descarta cajas diminutas (gradas)
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int | None = None            # None -> auto-detectar del vídeo real (ver process_video)


@dataclass(frozen=True)
class EventsConfig:
    """Hiperparámetros de core/events.py (clasificador V-JEPA2 + SVC)."""

    classifier_path: Path = PROJECT_ROOT / "models" / "svc_classifier_more_penalties.joblib"
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    device: str | None = None               # None -> auto (cuda si hay)
    num_frames: int = 64                    # debe coincidir con el fpc del checkpoint
    window_s: float = 6.0                   # duración real de los clips de entrenamiento
                                             # (medido: 238/240 clips de dataset_vjepa_backup/
                                             # duran exactamente 6.000s a 25fps) — sample_frames
                                             # muestrea num_frames uniformemente por TIEMPO, no
                                             # por frame, así que la ventana de inferencia debe
                                             # cubrir esta misma duración real, no num_frames crudos


@dataclass(frozen=True)
class BallActionsConfig:
    """Hiperparámetros de core/ball_actions.py (clasificador R3D-18 fine-tuned).

    Complementa a EventsConfig: en vez de un clip ya recortado por evento
    (gol/falta/...), este clasificador opera sobre ventanas de ``num_frames``
    centradas en el jugador que posee el balón (ver core/possession.py).
    Clases: pass / drive / high_pass / header / throw_in / background (shot y
    goal ya los cubre core/events.py, no se duplican aquí; tackle/cross/
    ball_player_block quedan fuera de esta primera fase — clases minoritarias
    o con riesgo de confusión visual, ver scripts/extract_ball_action_clips.py).
    """

    checkpoint_path: Path = PROJECT_ROOT / "models" / "r3d18_ball_actions-v2.pt"
    device: str | None = None               # None -> auto (cuda si hay)
    num_frames: int = 32                    # ~1.06 s a 30 fps
    padding_factor: float = 2.5             # expansión del bbox del poseedor antes de recortar
    classes: tuple[str, ...] = ("pass", "drive", "high_pass", "header", "throw_in", "background")
    max_possessor_distance_px: float = 150.0  # distancia máx. balón-jugador para resolver poseedor


@dataclass(frozen=True)
class ClassificationConfig:
    """Hiperparámetros de core/classification.py (embeddings PRTReId + K-Means de equipos)."""

    embedder: str = "prtreid"               # "prtreid" | "stub"
    reid_weights_path: Path = PROJECT_ROOT / "models" / "reid" / "prtreid-soccernet-baseline.pth.tar"
    reid_backbone_weights_path: Path = PROJECT_ROOT / "models" / "reid" / "hrnetv2_w32_imagenet_pretrained.pth"
    device: str | None = None               # None -> auto (cuda si hay)
    batch_size: int = 64                    # crops por lote a PRTReId
    image_size: tuple[int, int] = (256, 128)  # (height, width), fijo por el checkpoint publicado
    n_clusters: int = 2                     # nº de equipos (igual que TrackletTeamClustering original)
    random_state: int = 0                   # semilla de K-Means (igual que el original)


@dataclass(frozen=True)
class AnalyticsConfig:
    """Hiperparámetros de analytics/engine.py (composición final del pipeline).

    Cubre tanto ``analyze_match`` (clip corto ya recortado) como
    ``analyze_full_match`` (partido completo, ver ``FullMatchConfig`` para los
    parámetros específicos de ese segundo flujo).
    """

    max_possessor_distance_px: float = 150.0  # igual que possession.DEFAULT_MAX_DISTANCE_PX
    hysteresis_frames: int = 10               # igual que possession.DEFAULT_HYSTERESIS_FRAMES
    trace_length: int = 30                    # estela de trayectoria en el vídeo final
    action_highlight_frames: int = 60         # ~2.4s a 25fps: duración de la tarjeta de notificación
                                               # + resalte de jugador/balón tras una acción de balón
                                               # reconocida (ver utils.visualizers/render_match_video)


@dataclass(frozen=True)
class FullMatchConfig:
    """Hiperparámetros del flujo de partido completo (analytics/engine.py::
    analyze_full_match, main.py --mode full-match). Complementa a
    ``EventsConfig``: ``window_s`` vive allí porque es intrínseco al
    clasificador (duración de sus clips de entrenamiento); el solape/stride es
    una decisión del flujo de partido completo, no del clasificador en sí.
    """

    event_stride_s: float = 3.0             # ~50% de solape con EventsConfig.window_s=6.0
    min_window_fraction: float = 0.5        # descarta la ventana final si mide menos de
                                             # esta fracción de EventsConfig.window_s
    render_video: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración raíz que main.py pasará a cada módulo."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    ball_actions: BallActionsConfig = field(default_factory=BallActionsConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    full_match: FullMatchConfig = field(default_factory=FullMatchConfig)
    outputs_dir: Path = OUTPUTS_DIR
