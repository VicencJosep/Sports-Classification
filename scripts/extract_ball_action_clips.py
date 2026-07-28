"""Extrae clips de entrenamiento (T=32 frames, centrados en el poseedor del
balón) para el clasificador Pass/Drive/Tackle/Background de
``core/ball_actions.py``, a partir de los datos descargados con
``scripts/download_ball_action_data.py``.

Dos fuentes de clips, correspondientes a las dos fuentes de datos descritas en
ese script (ver también README.md):

* **Positivos** (`pass`/`drive`/`tackle`): exclusivamente de los partidos de
  Ball Action Spotting (``Labels-ball.json``), la única fuente con esas
  etiquetas. Un clip por anotación, centrado en el frame del evento.
* **Background**: de dos orígenes — (a) huecos sin anotación dentro de los
  propios partidos de Ball Action Spotting, y (b) opcionalmente partidos del
  corpus de 6 ligas ya descargado, usando ``Labels-v2.json`` (excluir cercanía
  a cualquier evento) + ``Labels-cameras.json`` (si existe, para diversificar
  tipo de plano). Ver ``extract_background_clips()``.

Cada clip se recorta con un ÚNICO bounding box estático (bbox del poseedor en
el momento del evento, expandido por ``padding_factor`` y recortado a los
límites del frame) aplicado a los 32 frames de la ventana — evita el jitter de
recortar frame a frame sobre detecciones ruidosas, y mantiene el mismo formato
ligero (.mp4) que ``clips_analisis/``, reutilizable con cualquier backbone.

Requiere que el vídeo de cada partido ya esté en disco (ver script de
descarga) y corre ``core.tracking``/``core.possession`` sobre él bajo demanda
(cacheando ``tracks.jsonl``/``possession.jsonl`` junto al vídeo).

Uso (desde la raíz del proyecto; con -m para que `import core.*` resuelva bien):
    python -m scripts.extract_ball_action_clips \\
        --match-dir datasets/SoccerNet/england_efl/2019-2020/<partido> \\
        --output dataset_ball_actions/ --background-per-half 40

Formato asumido de ``Labels-ball.json`` / ``Labels-v2.json`` / ``Labels-cameras.json``
(formato estándar de anotaciones SoccerNet; ``label`` va en MAYÚSCULAS en
``Labels-ball.json`` real, verificado contra los ficheros descargados):
    {"annotations": [{"gameTime": "1 - 03:24", "label": "PASS", "position": "204000", ...}, ...]}
``gameTime`` da la mitad ("1"/"2" antes de " - "); ``position`` son ms **dentro
de esa mitad** — EXCEPTO en ``Labels-ball.json`` (Ball Action Spotting), donde
el partido es un único vídeo sin dividir, ``gameTime`` marca siempre "1" y
``position`` corre sin reiniciar hasta el final del partido (~95-100 min; ver
``_HALF_VIDEO_CANDIDATES``). Si el fichero real difiere ligeramente (p.ej.
nombre de campo), ajustar solo ``_load_annotations()``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.ball_actions import compute_crop_box
from core.possession import DEFAULT_MAX_DISTANCE_PX, ensure_possession, iter_possession
from core.tracking import ensure_tracking

logger = logging.getLogger(__name__)

NUM_FRAMES = 32
DEFAULT_PADDING_FACTOR = 2.5
DEFAULT_OUTPUT_SIZE = 224  # lado del clip guardado en px (se re-preprocesa en training)

# Labels-ball.json (SN-BAS) -> nuestras clases positivas (Shot/Goal ya los
# cubre core/events.py, se excluyen aquí a propósito). Las labels van en
# MAYÚSCULAS en el JSON real (verificado contra los ficheros descargados),
# no en Title Case como en la documentación/README de SoccerNet.
#
# Volumen real medido sobre los 7 partidos descargados (total / min-max por
# partido): PASS 4985 (565-999), DRIVE 4300 (500-804), HIGH PASS 761 (82-131),
# HEADER 713 (54-128), THROW IN 362 (37-69) — las 5 con volumen suficiente y
# poco riesgo de confusión visual entre sí.
#
# Clases dejadas fuera deliberadamente en esta primera fase (descomentar para
# reintroducirlas; no requiere cambios en ningún otro sitio, solo aquí):
# - "PLAYER SUCCESSFUL TACKLE": muy minoritaria (74 total, 3-18/partido) y
#   clips confusos al revisarlos.
# - "CROSS": subtipo de pase definido por posición en el campo (banda -> área),
#   no por técnica corporal — mismo riesgo de confusión con HIGH PASS/PASS.
# - "BALL PLAYER BLOCK": probablemente confundible con tackle (acción
#   defensiva similar). "FREE KICK"/"GOAL": demasiado raras (21/13 total).
# - "OUT"/"SHOT"/"GOAL": no son técnicas del poseedor sino resultados/eventos
#   de otro tipo; Shot/Goal además ya los cubre core/events.py.
BALL_LABEL_TO_CLASS = {
    "PASS": "pass",
    "DRIVE": "drive",
    "HIGH PASS": "high_pass",
    "HEADER": "header",
    "THROW IN": "throw_in",
    # "PLAYER SUCCESSFUL TACKLE": "tackle",
    # "CROSS": "cross",
    # "BALL PLAYER BLOCK": "ball_player_block",
}
BACKGROUND_CLASS = "background"

# Dos esquemas de nombrado de vídeo distintos, según el corpus:
# - 6 ligas (Labels-v2.json/Labels-cameras.json, usado solo para Background):
#   un fichero POR MITAD, "{half}_720p.mkv" (half=1|2).
# - Ball Action Spotting (Labels-ball.json): un ÚNICO fichero para el partido
#   COMPLETO ("720p.mp4", sin prefijo de mitad ni extensión .mkv) — su
#   `gameTime` siempre marca "1 - MM:SS" con MM:SS corriendo sin reiniciar
#   hasta el final del partido (~95-100 min), es decir "half" aquí es un
#   campo vestigial que no corresponde a una mitad real de vídeo.
_HALF_VIDEO_CANDIDATES = ["{half}_720p.mkv", "{half}.mkv", "{half}_224p.mkv", "720p.mp4", "224p.mp4"]


# --------------------------------------------------------------------------- #
#  Carga de anotaciones SoccerNet
# --------------------------------------------------------------------------- #

@dataclass
class Annotation:
    half: int
    position_ms: int
    label: str


def _load_annotations(labels_path: Path) -> list[Annotation]:
    data = json.loads(labels_path.read_text())
    out = []
    for ann in data.get("annotations", []):
        game_time = ann["gameTime"]
        half = int(game_time.split(" - ")[0].strip())
        out.append(Annotation(half=half, position_ms=int(ann["position"]), label=ann["label"]))
    return out


def _load_camera_segments(labels_cameras_path: Path) -> dict[int, list[tuple[int, int, str]]]:
    """gameTime/position puntuales -> segmentos (start_ms, end_ms, label) por mitad."""
    anns = sorted(_load_annotations(labels_cameras_path), key=lambda a: (a.half, a.position_ms))
    segments: dict[int, list[tuple[int, int, str]]] = {}
    by_half: dict[int, list[Annotation]] = {}
    for a in anns:
        by_half.setdefault(a.half, []).append(a)
    for half, half_anns in by_half.items():
        segs = []
        for i, a in enumerate(half_anns):
            end_ms = half_anns[i + 1].position_ms if i + 1 < len(half_anns) else a.position_ms + 10 ** 9
            segs.append((a.position_ms, end_ms, a.label))
        segments[half] = segs
    return segments


def _is_main_camera(label: str) -> bool:
    return label.lower().startswith("main camera")


# --------------------------------------------------------------------------- #
#  Localización de vídeo por mitad
# --------------------------------------------------------------------------- #

def _find_half_video(match_dir: Path, half: int) -> Path:
    for pattern in _HALF_VIDEO_CANDIDATES:
        candidate = match_dir / pattern.format(half=half)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No encuentro el vídeo de la mitad {half} en {match_dir} "
        f"(probado: {[p.format(half=half) for p in _HALF_VIDEO_CANDIDATES]})"
    )


def _load_records_by_frame(jsonl_path: Path) -> dict[int, dict]:
    records = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["frame_idx"]] = rec
    return records


# --------------------------------------------------------------------------- #
#  Localización del bbox del poseedor + recorte
# --------------------------------------------------------------------------- #

def _bbox_of_tracker(record: dict, tracker_id: int) -> Optional[tuple[float, float, float, float]]:
    for track in record.get("tracks", []):
        if track["tracker_id"] == tracker_id:
            return tuple(track["xyxy"])
    return None


def find_reference_bbox(
    records_by_frame: dict[int, dict],
    center_frame_idx: int,
    search_radius: int = NUM_FRAMES,
) -> Optional[tuple[float, float, float, float]]:
    """Bbox del poseedor en `center_frame_idx`, o el válido más cercano dentro de
    `search_radius` frames. Si nunca hay poseedor resuelto, cae al bbox del
    balón más cercano (peor que nada: al menos centra el recorte en el balón).
    """
    for offset in range(search_radius + 1):
        for frame_idx in ({center_frame_idx - offset, center_frame_idx + offset}):
            record = records_by_frame.get(frame_idx)
            if record is None:
                continue
            possessor_id = record.get("possessor_tracker_id")
            if possessor_id is not None:
                bbox = _bbox_of_tracker(record, possessor_id)
                if bbox is not None:
                    return bbox

    for offset in range(search_radius + 1):
        for frame_idx in ({center_frame_idx - offset, center_frame_idx + offset}):
            record = records_by_frame.get(frame_idx)
            if record and record.get("ball"):
                return tuple(record["ball"]["xyxy"])
    return None


def expand_with_ball_trajectory(
    bbox: tuple[float, float, float, float],
    records_by_frame: dict[int, dict],
    start_frame: int,
    num_frames: int,
) -> tuple[float, float, float, float]:
    """Expande `bbox` (referencia del poseedor/balón en el frame del evento)
    para que también cubra la posición del balón en TODOS los frames de la
    ventana de extracción, no solo en el instante del evento.

    Necesario porque el recorte final es estático (mismo bbox para los 32
    frames, ver extract_window_clip): si el balón se aleja del poseedor
    dentro de la ventana — el caso típico de un ``PASS``, donde el balón sale
    disparado hacia otro jugador justo después del evento — un recorte
    centrado solo en el poseedor en el frame del evento pierde el balón de
    vista a mitad de clip. Devuelve la unión (bounding box mínimo que
    contiene todo) del `bbox` de partida y el bbox del balón en cada frame
    donde esté detectado.
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


def _ball_visible_in_window(records_by_frame: dict[int, dict], start_frame: int, num_frames: int) -> bool:
    """True si el balón se detecta en AL MENOS un frame de la ventana.

    Usado para descartar clips positivos (pass/drive) donde el balón nunca
    aparece — un hueco de detección demasiado largo para que
    `expand_with_ball_trajectory` compense; sin esto, el clip se guardaba
    igual con un recorte centrado solo en el poseedor y el balón ausente de
    principio a fin, mala señal de entrenamiento para una clase que por
    definición trata sobre qué le pasa al balón.
    """
    for frame_idx in range(start_frame, start_frame + num_frames):
        record = records_by_frame.get(frame_idx)
        if record and record.get("ball"):
            return True
    return False


def extract_window_clip(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    crop_box: tuple[int, int, int, int],
    out_path: Path,
    out_size: int = DEFAULT_OUTPUT_SIZE,
) -> bool:
    """Lee `num_frames` desde `start_frame`, recorta con `crop_box` (estático,
    igual para los 32 frames) y guarda como .mp4. Devuelve False si el vídeo
    no tiene suficientes frames a partir de start_frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    left, top, right, bottom = crop_box
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_size, out_size)
    )
    ok = True
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            ok = False
            break
        crop = frame[top:bottom, left:right]
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        writer.write(crop)
    writer.release()
    cap.release()

    if not ok:
        out_path.unlink(missing_ok=True)
    return ok


def _center_start_frame(center_frame_idx: int, num_frames: int, n_total_frames: int) -> int:
    """Ventana de `num_frames` centrada en center_frame_idx, desplazada (no
    truncada) para caber dentro de [0, n_total_frames)."""
    start = center_frame_idx - num_frames // 2
    start = max(0, min(start, max(0, n_total_frames - num_frames)))
    return start


# --------------------------------------------------------------------------- #
#  Extracción de positivos (Pass/Drive/Tackle) — solo partidos Ball Action Spotting
# --------------------------------------------------------------------------- #

def extract_positive_clips(
    match_dir: Path,
    labels_ball_path: Path,
    output_dir: Path,
    detector_name: str = "hybrid",
    device: Optional[str] = None,
    padding_factor: float = DEFAULT_PADDING_FACTOR,
    max_possessor_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
) -> dict[str, int]:
    annotations = [a for a in _load_annotations(labels_ball_path) if a.label in BALL_LABEL_TO_CLASS]
    counts = {cls: 0 for cls in BALL_LABEL_TO_CLASS.values()}

    by_half: dict[int, list[Annotation]] = {}
    for a in annotations:
        by_half.setdefault(a.half, []).append(a)

    for half, half_annotations in by_half.items():
        video_path = _find_half_video(match_dir, half)
        tracks_path = ensure_tracking(video_path, detector_name=detector_name, device=device)
        possession_path = ensure_possession(tracks_path, max_distance_px=max_possessor_distance_px)
        records_by_frame = _load_records_by_frame(possession_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        for ann in half_annotations:
            event_frame_idx = round(ann.position_ms / 1000.0 * fps)
            bbox = find_reference_bbox(records_by_frame, event_frame_idx)
            if bbox is None:
                logger.warning(
                    "%s h%d @%dms (%s): sin poseedor/balón resoluble cerca, se descarta",
                    match_dir.name, half, ann.position_ms, ann.label,
                )
                continue

            start_frame = _center_start_frame(event_frame_idx, NUM_FRAMES, n_total_frames)
            if not _ball_visible_in_window(records_by_frame, start_frame, NUM_FRAMES):
                logger.warning(
                    "%s h%d @%dms (%s): balón nunca detectado en la ventana, se descarta",
                    match_dir.name, half, ann.position_ms, ann.label,
                )
                continue

            reference_bbox = expand_with_ball_trajectory(bbox, records_by_frame, start_frame, NUM_FRAMES)
            crop_box = compute_crop_box(reference_bbox, padding_factor, frame_w, frame_h)
            cls = BALL_LABEL_TO_CLASS[ann.label]
            out_path = output_dir / cls / f"{match_dir.name}_h{half}_f{event_frame_idx}.mp4"

            if extract_window_clip(video_path, start_frame, NUM_FRAMES, crop_box, out_path):
                counts[cls] += 1
            else:
                logger.warning("%s: ventana incompleta (fin de vídeo), descartado", out_path)

    return counts


# --------------------------------------------------------------------------- #
#  Extracción de Background
# --------------------------------------------------------------------------- #

def extract_background_clips(
    match_dir: Path,
    output_dir: Path,
    exclusion_labels_path: Path,
    n_per_half: int,
    camera_segments_path: Optional[Path] = None,
    detector_name: str = "hybrid",
    device: Optional[str] = None,
    padding_factor: float = DEFAULT_PADDING_FACTOR,
    exclusion_margin_ms: int = 2000,
    max_possessor_distance_px: float = DEFAULT_MAX_DISTANCE_PX,
    seed: int = 0,
) -> int:
    """Ventanas T=32 sin ningún evento de `exclusion_labels_path` cerca (margen
    `exclusion_margin_ms`). Si `camera_segments_path` está disponible
    (Labels-cameras.json, solo existe para 200/500 partidos del corpus
    original), reparte las muestras entre planos `Main camera *` y el resto,
    tal y como se planteó para diversidad de cámara; si no está disponible
    (caso de los partidos de Ball Action Spotting), muestrea sin ese filtro.
    """
    rng = random.Random(seed)
    exclusions = _load_annotations(exclusion_labels_path)
    camera_segments = _load_camera_segments(camera_segments_path) if camera_segments_path else {}

    exclusions_by_half: dict[int, list[int]] = {}
    for a in exclusions:
        exclusions_by_half.setdefault(a.half, []).append(a.position_ms)

    n_extracted = 0
    halves = sorted(exclusions_by_half.keys()) or [1, 2]
    for half in halves:
        try:
            video_path = _find_half_video(match_dir, half)
        except FileNotFoundError:
            continue
        tracks_path = ensure_tracking(video_path, detector_name=detector_name, device=device)
        possession_path = ensure_possession(tracks_path, max_distance_px=max_possessor_distance_px)
        records_by_frame = _load_records_by_frame(possession_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        duration_ms = n_total_frames / fps * 1000.0

        excluded_ms = exclusions_by_half.get(half, [])
        segments = camera_segments.get(half, [])

        def far_from_events(candidate_ms: float) -> bool:
            return all(abs(candidate_ms - e) > exclusion_margin_ms for e in excluded_ms)

        extracted_this_half = 0
        attempts = 0
        max_attempts = n_per_half * 20
        while extracted_this_half < n_per_half and attempts < max_attempts:
            attempts += 1
            candidate_ms = rng.uniform(0, max(0.0, duration_ms - 1))
            if not far_from_events(candidate_ms):
                continue

            if segments:
                # Alterna main/no-main para diversidad de cámara (ver docstring).
                want_main = extracted_this_half % 2 == 0
                seg_label = next(
                    (label for (s, e, label) in segments if s <= candidate_ms < e), None
                )
                if seg_label is None or _is_main_camera(seg_label) != want_main:
                    continue

            candidate_frame_idx = int(candidate_ms / 1000.0 * fps)
            bbox = find_reference_bbox(records_by_frame, candidate_frame_idx)
            if bbox is None:
                continue

            start_frame = _center_start_frame(candidate_frame_idx, NUM_FRAMES, n_total_frames)
            reference_bbox = expand_with_ball_trajectory(bbox, records_by_frame, start_frame, NUM_FRAMES)
            crop_box = compute_crop_box(reference_bbox, padding_factor, frame_w, frame_h)
            out_path = output_dir / BACKGROUND_CLASS / f"{match_dir.name}_h{half}_f{candidate_frame_idx}.mp4"

            if extract_window_clip(video_path, start_frame, NUM_FRAMES, crop_box, out_path):
                extracted_this_half += 1
                n_extracted += 1

        if extracted_this_half < n_per_half:
            logger.warning(
                "%s h%d: solo %d/%d clips de background tras %d intentos",
                match_dir.name, half, extracted_this_half, n_per_half, attempts,
            )

    return n_extracted


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-dir", required=True, type=Path,
                        help="Carpeta del partido (contiene 1_720p.mkv/2_720p.mkv + Labels-*.json)")
    parser.add_argument("--output", required=True, type=Path, help="Carpeta raíz del dataset a generar")
    parser.add_argument("--labels-ball", default=None, type=Path,
                        help="Default: <match-dir>/Labels-ball.json (positivos pass/drive/tackle)")
    parser.add_argument("--background-labels", default=None, type=Path,
                        help="Fichero de exclusión para background (default: mismo --labels-ball; "
                             "usa Labels-v2.json si extraes background del corpus de 6 ligas)")
    parser.add_argument("--camera-labels", default=None, type=Path,
                        help="Labels-cameras.json opcional, para diversificar cámara en background")
    parser.add_argument("--background-per-half", type=int, default=0,
                        help="Nº de clips de background a muestrear por cada mitad (0 = desactivado)")
    parser.add_argument("--detector", default="hybrid", choices=["hybrid", "soccermaster", "yolo"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--padding-factor", type=float, default=DEFAULT_PADDING_FACTOR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-positives", action="store_true",
                        help="Solo extraer background (útil para partidos del corpus de 6 ligas)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    labels_ball_path = args.labels_ball or (args.match_dir / "Labels-ball.json")

    if not args.skip_positives:
        if not labels_ball_path.exists():
            parser.error(f"No existe {labels_ball_path} (pásalo con --labels-ball si tiene otro nombre)")
        counts = extract_positive_clips(
            args.match_dir, labels_ball_path, args.output,
            detector_name=args.detector, device=args.device, padding_factor=args.padding_factor,
        )
        logger.info("Positivos extraídos de %s: %s", args.match_dir.name, counts)

    if args.background_per_half > 0:
        exclusion_path = args.background_labels or labels_ball_path
        n_bg = extract_background_clips(
            args.match_dir, args.output, exclusion_path,
            n_per_half=args.background_per_half,
            camera_segments_path=args.camera_labels,
            detector_name=args.detector, device=args.device, padding_factor=args.padding_factor,
            seed=args.seed,
        )
        logger.info("Background extraído de %s: %d clips", args.match_dir.name, n_bg)


if __name__ == "__main__":
    main()
