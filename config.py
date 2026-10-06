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

# Repositorio PnLCalib (calibración de campo: keypoints + líneas -> homografía).
# Clonado de https://github.com/mguti97/PnLCalib — solo se usa su CÓDIGO
# (model/, utils/, config/). Vive bajo models/homography/ — para no ensuciar la
# raíz — junto a sus pesos SV_kp/SV_lines, en models/homography/weights/.
PNLCALIB_DIR: Path = PROJECT_ROOT / "models" / "homography" / "PnLCalib"

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
    temporal_offsets_frames: tuple[int, ...] = (0,)
    # Ventanas (desplazadas respecto al frame del trigger) cuyas probabilidades
    # se promedian. (0,) = una sola ventana centrada en el trigger.
    # Medido contra ground_truth/ (20 min, 367 acciones): el trigger ya cae
    # encima de la acción real (desfase MEDIANO de 0 frames, p25 -7 / p75 +6),
    # así que promediar (-8, 0, 8) solo añade ventanas que ya no contienen la
    # acción: F1 global 29.0% -> 29.2% a 3x de coste de inferencia. No
    # reintentar sin arreglar antes el disparador (ver README/analytics.engine).


@dataclass(frozen=True)
class BallSpottingConfig:
    """Hiperparámetros de core/ball_spotting.py (T-DEED, detección puntual de
    acciones de balón). Sustituye a core/events.py para shot/goal y, opcional,
    al disparador por cambio de posesión de las acciones de balón.

    Umbrales medidos (ver el docstring de core/ball_spotting.py):
    ``shot_min_score=0.3`` -> P 65% / R 84% en 6 partidos SoccerNet-v2 de otras
    ligas (0.5 -> P 75% / R 73%). ``action_min_score=0.5`` -> F1 90% en
    ground_truth/ (0.3 -> 84%); elegido sobre ese mismo partido, así que es
    algo optimista.
    """

    repo_dir: Path = PROJECT_ROOT / "models" / "tdeed" / "T-DEED"
    checkpoints_dir: Path = PROJECT_ROOT / "models" / "tdeed"
    model_name: str = "SoccerNetBall_challenge2"
    shot_min_score: float = 0.3
    goal_min_score: float = 0.3
    event_nms_s: float = 4.0                # dos tiros a menos de 4 s son el mismo pico
    event_smooth_frames: int = 6            # ~0,5 s (frames muestreados a 12,5 fps)
    action_min_score: float = 0.5
    action_nms_s: float = 0.5               # ~12 frames: acciones seguidas del mismo tipo


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
class HomographyConfig:
    """Hiperparámetros de core/homography.py (calibración de campo PnLCalib).

    Los defaults de ``stride``/``batch_size``/``half`` salen de medir el coste
    real en la RTX 4090 del contenedor (2 HRNet-W48 a 960x540 por frame):

        fp32 batch=1   58 ms/frame   1.9 GiB      fp16 batch=1   43 ms/frame
        fp32 batch=4   41 ms/frame   6.1 GiB      fp16 batch=4   26 ms/frame  4.7 GiB
        fp32 batch=8   42 ms/frame  11.6 GiB      fp16 batch=8   24 ms/frame  8.7 GiB

    El solve de la homografía en CPU (RANSAC + correspondencias) es
    despreciable al lado: 0.5 ms/frame. Es decir, esto es GPU-bound y el
    único parámetro que mueve la aguja de verdad es ``stride``: a 25 fps un
    partido de 90 min son 135k frames = ~1h a stride=1, ~12 min a stride=5.
    stride=5 (una calibración cada 0.2 s) es de sobra para el paneo de una
    cámara de retransmisión, que es un movimiento suave y lento.
    """

    pnlcalib_repo_path: Path = PNLCALIB_DIR
    weights_kp: Path = PROJECT_ROOT / "models" / "homography" / "weights" / "SV_kp"
    weights_line: Path = PROJECT_ROOT / "models" / "homography" / "weights" / "SV_lines"
    device: str | None = None               # None -> auto (cuda si hay)
    half: bool = True                       # autocast fp16 (ver tabla de arriba)
    batch_size: int = 4                     # frames por forward de HRNet
    stride: int = 5                         # calibra 1 de cada N frames

    # Umbrales de decodificación de heatmaps: los mismos que usan los autores
    # en PnLCalib/inference.py — no son redondos porque salen de su barrido.
    kp_threshold: float = 0.3434
    line_threshold: float = 0.7867
    model_input_size: tuple[int, int] = (960, 540)  # (w, h), fijo por entrenamiento

    use_ransac: float = 5.0                 # umbral (px) de cv2.findHomography
    pnl_refine: bool = False                # refinamiento PnL con líneas (más lento/preciso)

    # Descarta homografías con error de reproyección absurdo: una H "resuelta"
    # pero mala proyecta jugadores a cientos de metros y contamina las métricas
    # más que no tener homografía.
    max_reprojection_error_px: float = 25.0

    # Hueco máximo (en frames) para reutilizar la H de un frame vecino ya
    # calibrado. Más allá (repetición, plano corto, cámara táctica) es
    # preferible no tener homografía a arrastrar una obsoleta.
    max_gap_frames: int = 25

    # Dimensiones canónicas del campo (estándar FIFA / SoccerNet-Calibration)
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0


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
class PublisherConfig:
    """Hiperparámetros de automation/molmo_publisher.py (MolmoWeb-4B + Playwright).

    A diferencia del resto de módulos, aquí casi todo son parámetros de
    *navegación*, no de modelo: el VLM solo aporta las coordenadas del clic.

    ``viewport`` es el que más se nota en la tasa de acierto: los prompts
    describen posiciones ("arriba a la derecha") y el layout de YouTube Studio
    es responsive, así que cambiarlo cambia lo que ve el modelo. Fijarlo hace
    reproducible el flujo entre ejecuciones y máquinas.
    """

    model_id: str = "allenai/MolmoWeb-4B"
    device: str | None = None               # None -> auto (cuda si hay)
    dtype: str = "bfloat16"                 # los autores recomiendan float32, pero el
                                             # modelo (Qwen3-8B de base) no cabe en 24 GB
                                             # así; si las coordenadas salen erráticas,
                                             # esta es la primera sospechosa
    max_new_tokens: int = 200               # el JSON {"thought":…, "action":{…}} cabe de sobra

    # El VLM corre en un intérprete APARTE. No es capricho: el checkpoint se
    # guardó con transformers 4.57.3 y su repo pinea >=4.51,<5, mientras que
    # este repo fija 5.14.1 por la API de Siglip de Soccer_Master. Con la 5.x el
    # código remoto del checkpoint falla en cadena (RoPE, ProcessorMixin,
    # cache_position). Ver automation/molmo_server.py.
    venv_python: Path = Path("/opt/venv-molmo/bin/python")

    # Perfil persistente de Chromium: guarda las cookies de Google para no
    # repetir login/2FA en cada ejecución. Ignorado por .gitignore — contiene
    # credenciales de sesión, NO se versiona.
    user_data_dir: Path = PROJECT_ROOT / "automation" / "chrome_session"
    headless: bool = False                  # Studio se comporta peor en headless
                                             # y el login inicial exige ventana
    viewport_width: int = 1440
    viewport_height: int = 900

    studio_url: str = "https://studio.youtube.com"
    default_visibility: str = "private"     # "private" | "unlisted"; "public" no
                                             # se ofrece a propósito (banco de pruebas)

    # Pausas (ms) antes de cada captura: una captura tomada a mitad de la
    # animación de un modal le da al VLM una UI a medio pintar y las
    # coordenadas salen desplazadas.
    settle_ms: int = 2500
    modal_settle_ms: int = 4000             # apertura del diálogo de subida
    upload_settle_ms: int = 8000            # tras soltar el fichero: parseo + formulario


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración raíz que main.py pasará a cada módulo."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    ball_actions: BallActionsConfig = field(default_factory=BallActionsConfig)
    ball_spotting: BallSpottingConfig = field(default_factory=BallSpottingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    full_match: FullMatchConfig = field(default_factory=FullMatchConfig)
    publisher: PublisherConfig = field(default_factory=PublisherConfig)
    outputs_dir: Path = OUTPUTS_DIR
