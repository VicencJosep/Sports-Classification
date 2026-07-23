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

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"


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
    frame_rate: int = 30


@dataclass(frozen=True)
class EventsConfig:
    """Hiperparámetros de core/events.py (clasificador V-JEPA2 + SVC)."""

    classifier_path: Path = PROJECT_ROOT / "svc_classifier_more_penalties.joblib"
    model_id: str = "facebook/vjepa2-vitl-fpc64-256"
    device: str | None = None               # None -> auto (cuda si hay)
    num_frames: int = 64                    # debe coincidir con el fpc del checkpoint


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración raíz que main.py pasará a cada módulo."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    outputs_dir: Path = OUTPUTS_DIR
