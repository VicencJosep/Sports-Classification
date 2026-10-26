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
    ball_model_path: str | None = None      # None = YOLO11x COCO. models/ball/yolo11s_snt_ball.pt (ajustado con SoccerNet
                                             # Tracking) acierta más en TV (top-1 78% frente a 62%) pero en YouTube su mejor
                                             # detección salta a botas y fondo: no usarlo sin coherencia temporal (13-oct-2026)
    ball_imgsz: int = 1280                  # resolución de YOLO para el balón (= core.tracking.DEFAULT_BALL_IMGSZ;
                                             # 640 dejaba el balón de un 720p en 3-5 px: 54% de frames con balón
                                             # frente a 70% a 1280, medido el 11-oct-2026)
    device: str | None = None               # None -> auto (cuda si hay)
    confidence: float = 0.3                 # umbral personas
    ball_confidence: float = 0.15           # umbral (laxo) para el balón
    min_box_area: float = 300.0             # descarta cajas diminutas (gradas)
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int | None = None            # None -> auto-detectar del vídeo real (ver process_video)
    static_ball_window_frames: int = 750     # core.tracking.StaticBallFilter: un "balón" clavado en el mismo píxel
    static_ball_min_history_frames: int = 150  # frames de historial antes de empezar a filtrar (~6 s a 25 fps)
    static_ball_min_presence: float = 0.7    # en >= 70% de los últimos 750 frames es un escudo/logo del marcador,
                                             # no el balón (medido el 12-oct-2026: sin esto el escudo del marcador
                                             # ganaba al balón real en el 45-80% de los frames de un juvenil de YouTube)


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
    """Hiperparámetros de core/ball_spotting.py (T-DEED, detección puntual).
    T-DEED dispara las acciones de balón y, con ``event_source="tdeed"`` (por
    defecto desde el 15-oct-2026), pone TODOS los eventos de la línea
    temporal: tiro, gol y saque de banda con su cabeza de acciones de balón;
    córner, falta, falta directa/indirecta, tarjeta, saque inicial, penalti,
    sustitución y fuera de juego con su cabeza de SoccerNet-v2. V-JEPA2
    (core/events.py) ya no corre: medía AP de tiros 34% y fuera de la EFL
    disparaba saques de banda 4-5 veces más de la cuenta. ``"mixed"`` es el
    comportamiento anterior (V-JEPA2 + tiro/gol/saque de T-DEED) y
    ``"vjepa2"`` el de antes de T-DEED.

    Umbrales por defecto (= techo de los automáticos):

    * ``action_min_score=0.5`` -> F1 90% en ground_truth/; en el partido
      completo de Middlesbrough F1 93% (0.3 -> 88%).
    * ``shot``/``goal`` 0.3 -> P 65% / R 84% en 6 partidos SoccerNet-v2 de
      otras ligas (0.5 -> P 75% / R 73%); ``throw-in`` 0.5.
    * Clases de la cabeza SoccerNet: máximo F1 a ±3 s en 10 partidos del split
      de ENTRENAMIENTO de SoccerNet-v2 (scripts/tune_tdeed_event_thresholds.py,
      output/sn_event_tuning/). T-DEED se preentrenó con SoccerNet-v2, así que
      ahí puntúa alto: son umbrales prudentes.

    Umbrales automáticos (``auto_thresholds``, ver
    analytics.engine.calibrate_spot_thresholds): fuera de la EFL T-DEED
    dispara en los mismos sitios con menos confianza (dos juveniles de YouTube,
    11-oct-2026: a 0.5 salen 4-5,6 acciones/min y 5-7 tiros por partido). Por
    partido se elige el umbral de acciones que da el ritmo de los 4 partidos
    EFL de entrenamiento (15,8 acciones/min de juego), acotado a
    [``auto_action_floor``, ``action_min_score``]: solo baja, nunca sube.
    Medido el 15-oct-2026: Middlesbrough completo -> 0.45 (F1 92%, el máximo
    está en 0.45-0.5); Mallorca 0.27 y Espanyol 0.24; en SoccerTrack (cámara
    panorámica, anotación más densa) el ritmo pediría 0.73-0.88 y bajaría el
    F1, de ahí el techo. Los EVENTOS no se escalan (``auto_event_min_factor=1``):
    en 10 partidos de SoccerNet-v2, donde la cabeza de acciones también puntúa
    bajo (umbral automático 0.21-0.22), bajar los de tiro y saque con ese
    factor empeoraba el F1 a ±3 s (tiro 0.66 -> 0.62, saque 0.80 -> 0.71): lo
    que se apaga fuera de la EFL son pases y conducciones, no los eventos; y
    la cabeza SoccerNet sale con factor ~1 en los juveniles. El mecanismo
    sigue ahí por si un dominio nuevo lo pide. ``main.py --thresholds fixed``
    usa los fijos para todo.
    """

    repo_dir: Path = PROJECT_ROOT / "models" / "tdeed" / "T-DEED"
    checkpoints_dir: Path = PROJECT_ROOT / "models" / "tdeed"
    model_name: str = "SoccerNetBall_challenge2"
    event_source: str = "tdeed"             # "tdeed" | "mixed" | "vjepa2"
    # (evento, umbral, nms_s): evento = clave de core.ball_spotting.EVENT_SOURCES. Los de la cabeza
    # SoccerNet, al máximo F1 a ±3 s en 10 partidos de entrenamiento de SoccerNet-v2 (F1 entre paréntesis;
    # a ±10 s sale casi igual, así que no es la tolerancia): output/sn_event_tuning/thresholds.json.
    event_params: tuple[tuple[str, float, float], ...] = (
        ("shot", 0.3, 4.0),
        ("goal", 0.3, 4.0),
        ("throw-in", 0.5, 4.0),
        ("corner", 0.25, 2.0),        # F1 0.89
        ("foul", 0.65, 8.0),          # F1 0.57
        ("free-kick", 0.4, 8.0),      # F1 0.31
        ("card", 0.3, 4.0),           # F1 0.17
        ("kick-off", 0.3, 8.0),       # F1 0.12 (máximo a 0.05: inservible)
        ("penalty", 0.5, 2.0),        # 3 penaltis en los 10 partidos: sin medida
        ("substitution", 0.2, 8.0),   # F1 0.23
        ("offside", 0.2, 2.0),        # F1 0.17
    )
    # Eventos que se emiten con event_source="tdeed". Solo los que T-DEED acierta: tiro/gol/saque (medidos
    # en la EFL) y córner/falta. Tarjeta, saque inicial, sustitución, penalti, fuera de juego y falta
    # directa/indirecta no pasan de F1 0.35 ni en partidos que T-DEED vio al entrenar: en partidos nuevos
    # serían sobre todo falsas alarmas. Para emitirlos igualmente, añadirlos aquí.
    tdeed_event_labels: tuple[str, ...] = ("shot", "goal", "throw-in", "corner", "foul")
    mixed_event_labels: tuple[str, ...] = ("shot", "goal", "throw-in")  # los de T-DEED con event_source="mixed"
    # Eventos que se emiten pero SIN verdad medida en nuestros partidos: van marcados ``estimated`` en la
    # telemetría y en la API para que la interfaz los enseñe como estimación. Falta: P 0/3 en los 20 min de
    # Middlesbrough (2 faltas reales), 17/32 en el Clásico (preliminar, partido que T-DEED vio al entrenar).
    # Córner (F1 0,89 en SoccerNet-v2, también in-sample) se mide primero en el juvenil anotado. Para dejar de
    # emitir una clase, quitarla de tdeed_event_labels.
    tdeed_estimated_event_labels: tuple[str, ...] = ("foul",)
    event_smooth_frames: int = 6            # ~0,5 s (frames muestreados a 12,5 fps)
    action_min_score: float = 0.5
    action_nms_s: float = 0.5               # ~12 frames: acciones seguidas del mismo tipo
    auto_thresholds: bool = True
    auto_action_rate_per_min: float = 15.8  # acciones/min de juego, 4 partidos EFL de entrenamiento
    auto_action_floor: float = 0.2
    auto_event_min_factor: float = 1.0      # 1 = los eventos no se escalan (ver docstring)
    # Cabeza SoccerNet: ritmo de anotaciones (todas las clases) por minuto de juego y umbral de "algún
    # evento" que da ese ritmo, en 10 partidos de entrenamiento de SoccerNet-v2 (tune_tdeed_event_thresholds).
    # 0 = sin calibrar esa cabeza (factor 1). Medido el 15-oct-2026 (umbral por mitad entre 0.59 y 0.78).
    auto_sn_rate_per_min: float = 2.5
    auto_sn_reference_score: float = 0.677


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
class HighlightsConfig:
    """Hiperparámetros de analytics/highlights (mejores jugadas de un partido
    ya analizado). Tres bloques: anclas/jugada (reglas, sin vídeo), VLM
    (selección y corte) y selección final.

    Los umbrales de ancla son MUY bajos a propósito: en los partidos
    juveniles T-DEED puntúa un gol real a SHOT 0,1-0,2 y GOAL ~0 (no sale en
    telemetry.json); aquí solo se piden candidatos, el VLM filtra después.
    """

    # Anclas (momento clave)
    shot_min_score: float = 0.08            # curva SHOT (o "Shots on/off target" de SoccerNet)
    goal_min_score: float = 0.02            # curva GOAL (o "Goal" de SoccerNet)
    audio_min_sustained_db: float = 6.0     # media 15 s del audio sobre la mediana móvil de 5 min
    audio_min_actions: int = 3              # acciones de balón en ±30 s: balón en juego (no descanso)
    anchor_merge_s: float = 15.0            # anclas más cercanas son la misma jugada

    # Jugada hacia atrás (cadena de acciones del equipo atacante)
    play_max_back_s: float = 35.0
    play_min_back_s: float = 6.0
    play_default_back_s: float = 10.0       # sin acciones conocidas
    play_pre_s: float = 1.5                 # margen antes de la primera acción
    play_gap_s: float = 6.0                 # hueco máximo entre acciones de la cadena
    dead_ball_out_score: float = 0.35       # curva OUT por encima = balón parado entre acciones

    # Fin provisional (reglas; el VLM lo afina)
    post_min_s: float = 8.0
    post_max_s: float = 45.0                # celebración + repeticiones de TV
    post_default_s: float = 12.0
    audio_tail_db: float = 3.0              # se alarga mientras el audio sostenido supere esto

    # Pesos de la puntuación por reglas (solo ordena candidatos)
    w_goal: float = 3.0
    w_shot: float = 2.0
    w_audio: float = 0.12                   # por dB sostenido (tope 15)
    w_restart: float = 0.8                  # saque de centro después = gol
    w_passes: float = 0.05                  # por pase de la jugada (tope 10)
    w_drive: float = 0.02                   # por metro de conducción de un jugador (tope 40)
    w_burst: float = 0.3                    # por tiro extra en ±60 s (tope 3)
    w_near_goal: float = 0.3                # balón a < 40 m de portería en el ancla

    # VLM (Qwen3.5 con visión). "ollama": API nativa; "openai": vLLM u otro compatible.
    vlm_enabled: bool = True
    vlm_api: str = "ollama"
    vlm_base_url: str = "http://localhost:11434"
    vlm_model: str = "qwen3.5:9b-q8_0"
    vlm_api_key: str = "EMPTY"
    vlm_timeout_s: float = 900.0
    vlm_max_candidates: int = 10            # mejores por reglas que ve el VLM (+ goles claros)
    vlm_frames_judge: int = 12
    vlm_frames_end: int = 16
    vlm_end_step_s: float = 4.0
    vlm_image_width: int = 640

    # Marcador de la retransmisión (cambio de marcador = gol); lo lee el VLM
    scoreboard_enabled: bool = True
    scoreboard_step_s: float = 30.0         # una lectura cada tanto; el cambio se localiza luego por bisección
    scoreboard_min_valid: float = 0.25      # fracción de lecturas legibles para fiarse de que hay rótulo
    scoreboard_goal_before_s: float = 45.0  # el gol (ancla) cae hasta tanto antes de que el rótulo cambie
    scoreboard_goal_near_s: float = 20.0    # ...pero normalmente dentro de este margen: si hay candidatos ahí, es uno de ellos
    w_scoreboard: float = 5.0               # bonus de reglas al candidato del gol confirmado

    # V-JEPA2 + SVC como segundo juez de gol (sin marcador): ventanas de 6 s
    # en rejilla fija; el golpeo se mira en todos los candidatos y lo que pasa
    # después (saque de centro / córner / banda) en los mejores o con indicio
    vjepa_enabled: bool = True
    vjepa_window_s: float = 6.0
    vjepa_goal_pre_s: float = 3.0           # ventana de gol: [ancla - pre, ancla + post]
    vjepa_goal_post_s: float = 9.0
    vjepa_after_from_s: float = 12.0        # saque de centro: [ancla + from, ancla + to]
    vjepa_after_to_s: float = 96.0
    vjepa_deadball_to_s: float = 45.0       # córner/banda poco después = no fue gol
    vjepa_max_candidates: int = 30          # ventanas posteriores para estos (+ vj_goal >= after_min_goal)
    vjepa_after_min_goal: float = 0.1
    vjepa_loud_min_goal: float = 0.2        # con griterío (audio informativo), esto basta para que el VLM lo vea
    vjepa_classifier_path: str | None = None  # por defecto core.events.DEFAULT_CLASSIFIER_PATH
    vjepa_model_id: str | None = None
    w_vj_goal: float = 2.0                  # bonus de reglas (ordena; que el VLM vea los goles probables)
    w_vj_kickoff: float = 0.5
    goal_fusion_min: float = 1.4            # vjepa.goal_fusion >= esto (y celebración vista) -> "gol probable"
    goal_audio_min_db: float = 6.0          # griterío sostenido que, con celebración vista, hace "gol probable" (cámara única)

    # Selección final: entre min y max clips, por encima de select_min_score
    min_clips: int = 3
    max_clips: int = 6
    select_min_score: float = 0.45
    w_final_rules: float = 0.4
    w_final_vlm: float = 0.6
    merge_gap_s: float = 20.0               # jugadas casi seguidas (parada -> córner -> gol) en un clip

    # Duración de los clips (hoja de ruta §3.4): 12-30 s; los goles son la excepción (hasta 45 s para que
    # quepa la celebración, contados desde el gol y no desde el ancla, que en cámara única puede ir 20-30 s
    # antes). Se aplica en pipeline.fit_duration tras fijar los límites con el VLM: primero se recorta lo
    # anterior al momento clave (clip_back_max_s) y luego la cola; lo más corto de clip_min_s se alarga.
    clip_min_s: float = 12.0
    clip_max_s: float = 30.0
    clip_max_goal_s: float = 45.0
    clip_back_max_s: float = 12.0           # como mucho tanto antes del momento clave (jugada sin gol)
    clip_back_max_goal_s: float = 15.0      # ...y en un gol
    end_replay_max_s: float = 8.0           # las repeticiones de TV (solo cuentan tras un gol) alargan como mucho esto
    clip_max_merged_s: float = 40.0         # dos jugadas seguidas solo se funden si el clip no pasa de esto

    # Corte
    ffmpeg_crf: int = 20
    ffmpeg_preset: str = "medium"


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
    highlights: HighlightsConfig = field(default_factory=HighlightsConfig)
    outputs_dir: Path = OUTPUTS_DIR
