"""
core/events.py
===============
Clasificación de eventos de fútbol (gol, penalti, falta, córner, tarjeta...)
sobre un clip de vídeo ya recortado, usando embeddings de V-JEPA2 + un SVC
entrenado offline.

Pipeline: clip.mp4 -> sample_frames() -> V-JEPA2 (embedding 3072-d) -> SVC -> label.

El clasificador (``svc_classifier_more_penalties.joblib`` por defecto) reconoce
16 clases: 9 de evento — card, corner, foul, goal, kick-off, penalty, shot,
substitution, throw-in — y 7 de "fondo" (tipo de plano sin evento propiamente
dicho: primeros planos, público, cámara principal...), usadas para que el
modelo no dispare eventos falsos durante callejones del partido. La taxonomía
es deliberadamente más gruesa que la de otros datasets del repo (p.ej. no
distingue shots_on_target/shots_off_target ni yellow/red card): se fusionaron
a propósito porque esa frontera era difusa para el modelo y perjudicaba la
precisión.

Importante: ``sample_frames``/``predict()`` muestrean NUM_FRAMES frames
uniformemente a lo largo de TODA la duración del vídeo de entrada, y asumen
que ``video_path`` ya es un CLIP recortado en torno a un único evento (unos
pocos segundos), como los de ``clips_analisis/`` — pasarles un partido
completo sin trocear diluye la señal del evento entre 90 minutos y no produce
una predicción útil. Para partidos completos, usar en su lugar
``sample_frames_window``/``predict_window()``, que clasifican solo una
ventana ``[start_s, end_s]`` sin decodificar el vídeo entero; el troceado en
ventana deslizante vive en ``core/windowing.py`` y se orquesta desde
``analytics/engine.py::analyze_full_match``. Los clips de entrenamiento del
SVC duran ~6.0s reales (ver ``config.EventsConfig.window_s``) — las ventanas
de inferencia deben cubrir esa misma duración real, no solo NUM_FRAMES crudos,
o hay riesgo de distribution-shift frente al entrenamiento.

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.events --source clip.mp4
    python -m core.events --source clips_folder/ --output events.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from transformers import VJEPA2Model, VJEPA2VideoProcessor

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_PATH: Path = _PROJECT_ROOT / "models" / "svc_classifier_more_penalties.joblib"
DEFAULT_MODEL_ID: str = "facebook/vjepa2-vitl-fpc64-256"
DEFAULT_NUM_FRAMES: int = 64

_BACKGROUND_PREFIX: str = "background_"


# --------------------------------------------------------------------------- #
#  Registro serializable (telemetría)
# --------------------------------------------------------------------------- #

@dataclass
class EventPrediction:
    """Predicción del clasificador de eventos para un clip."""

    video_path: str
    label: str                         # p.ej. "goal", "foul", "background_public"
    confidence: float                  # probabilidad de la clase predicha
    probabilities: dict[str, float]    # probabilidad de las 16 clases
    is_background: bool                # True si label empieza por "background_"
                                        # (plano sin evento real, útil para que
                                        # analytics/engine.py lo descarte)

    def to_telemetry(self) -> dict:
        """Registro serializable para el JSON/JSONL de telemetría."""
        return asdict(self)


def _uniform_sample_by_time(
    decoded: list[tuple[float, np.ndarray]], start_s: float, end_s: float, n: int
) -> list[np.ndarray]:
    """De una lista (timestamp, frame) ya decodificada, escoge n frames
    muestreados uniformemente por TIEMPO en [start_s, end_s] (el frame
    decodificado más cercano, por timestamp, a cada punto de muestreo)."""
    target_times = np.linspace(start_s, end_s, num=n)
    decoded_times = np.array([t for t, _ in decoded])
    return [decoded[int(np.argmin(np.abs(decoded_times - tt)))][1] for tt in target_times]


def sample_frames(video_path: Path, n: int = DEFAULT_NUM_FRAMES) -> list[np.ndarray]:
    """Extrae n frames RGB muestreados uniformemente por timestamp (0 -> duración).

    Decodifica el contenedor ENTERO — viable para clips cortos ya recortados
    (uso previsto de esta función), no para un partido completo. Para eso,
    usar ``sample_frames_window`` (busca/decodifica solo la ventana pedida).
    """
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        tb = stream.time_base

        if stream.duration is not None and tb is not None:
            duration = float(stream.duration * tb)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            duration = 0.0
            last_pts = None
            for frame in container.decode(video=0):
                if frame.pts is not None:
                    last_pts = frame.pts
            if last_pts is None:
                raise ValueError(f"No se pudo determinar la duración de {video_path}")
            duration = float(last_pts * tb)
            container.seek(0)

        decoded: list[tuple[float, np.ndarray]] = []
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            decoded.append((float(frame.pts * tb), frame.to_ndarray(format="rgb24")))
    finally:
        container.close()

    if not decoded:
        raise ValueError(f"No se pudo decodificar ningún frame de {video_path}")

    return _uniform_sample_by_time(decoded, 0.0, duration, n)


def sample_frames_window(
    video_path: Path, start_s: float, end_s: float, n: int = DEFAULT_NUM_FRAMES
) -> list[np.ndarray]:
    """Como ``sample_frames``, pero solo sobre la ventana ``[start_s, end_s]``
    de ``video_path``, SIN decodificar el vídeo entero: hace seek al keyframe
    anterior a ``start_s`` y decodifica hacia delante hasta superar ``end_s``.

    Pensada para partidos completos (``analytics/engine.py::
    analyze_full_match``, ventana deslizante vía ``core/windowing.py``): acota
    el coste a "retroceso de GOP + duración de la ventana" en vez de al vídeo
    completo. ``sample_frames`` sigue existiendo tal cual para el flujo de
    clip corto.
    """
    return _sample_range_at(video_path, start_s, end_s, np.linspace(start_s, end_s, num=n))


def sample_frames_at(video_path: Path, times: Sequence[float]) -> list[np.ndarray]:
    """Frames RGB en instantes arbitrarios (el decodificado más cercano a
    cada uno), decodificando una sola vez el tramo ``[min, max]`` de
    ``times``. Para muestreos no uniformes, p.ej. denso alrededor de un
    instante y disperso en el contexto (``core.events_vllm.ShotVerifier``)."""
    return _sample_range_at(video_path, float(min(times)), float(max(times)), times)


def _sample_range_at(
    video_path: Path, start_s: float, end_s: float, target_times: Sequence[float]
) -> list[np.ndarray]:
    """Seek al keyframe anterior a ``start_s``, decodifica hasta ``end_s`` y
    devuelve, para cada instante de ``target_times``, el frame decodificado
    más cercano por timestamp. Solo convierte a RGB los frames elegidos (no
    los ~150 de una ventana de 6 s), antes de cerrar el contenedor."""
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        tb = stream.time_base

        container.seek(int(start_s / tb), stream=stream, backward=True, any_frame=False)

        decoded: list[tuple[float, "av.VideoFrame"]] = []
        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            ts = float(frame.pts * tb)
            if ts < start_s:
                continue  # el seek aterriza en el keyframe anterior, puede ser < start_s
            if ts > end_s:
                break
            decoded.append((ts, frame))

        if not decoded:
            raise ValueError(
                f"No se pudo decodificar ningún frame de {video_path} en [{start_s}, {end_s}]"
            )

        decoded_times = np.array([t for t, _ in decoded])
        picks = [int(np.argmin(np.abs(decoded_times - tt))) for tt in target_times]
        rgb = {i: decoded[i][1].to_ndarray(format="rgb24") for i in set(picks)}
    finally:
        container.close()

    return [rgb[i] for i in picks]


# --------------------------------------------------------------------------- #
#  Clasificador
# --------------------------------------------------------------------------- #

class EventClassifier:
    """Clasifica clips de fútbol en eventos (gol/penalti/falta/...) vía V-JEPA2 + SVC.

    Carga el backbone V-JEPA2 y el clasificador SVC una única vez en el
    constructor; ``predict`` / ``predict_many`` se pueden llamar repetidamente
    sobre distintos clips sin recargar nada (pensado para integrarse en
    ``main.py`` o en un servicio de inferencia de larga duración).

    Args:
        classifier_path: ruta al ``.joblib`` con ``{"clf": Pipeline, "le": LabelEncoder}``.
        model_id: checkpoint de V-JEPA2 en Hugging Face.
        device: "cuda" | "cpu" (None -> cuda si está disponible).
        num_frames: frames muestreados por clip. Debe coincidir con el ``fpc``
            del checkpoint de V-JEPA2 (64 para ``vjepa2-vitl-fpc64-256``); no
            se recomienda cambiarlo salvo que también cambies ``model_id``.
    """

    def __init__(
        self,
        classifier_path: Path | str = DEFAULT_CLASSIFIER_PATH,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
    ) -> None:
        import joblib
        import torch
        from transformers import VJEPA2Model, VJEPA2VideoProcessor

        classifier_path = Path(classifier_path)
        if not classifier_path.exists():
            raise FileNotFoundError(f"Clasificador de eventos no encontrado: {classifier_path}")

        self._torch = torch
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames

        logger.info("Cargando V-JEPA2 (%s) en %s ...", model_id, self.device)
        self.processor: "VJEPA2VideoProcessor" = VJEPA2VideoProcessor.from_pretrained(model_id)
        self.model: "VJEPA2Model" = VJEPA2Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

        logger.info("Cargando clasificador de eventos desde %s ...", classifier_path)
        saved = joblib.load(classifier_path)
        self.clf: "Pipeline" = saved["clf"]
        self.label_encoder: "LabelEncoder" = saved["le"]
        logger.info("Clases: %s", ", ".join(self.label_encoder.classes_))

    # ------------------------------------------------------------------ #

    def predict(self, video_path: Path | str) -> EventPrediction:
        """Clasifica un único clip .mp4 entero. Lanza excepción si no se puede leer/procesar."""
        video_path = Path(video_path)
        frames = sample_frames(video_path, self.num_frames)
        return self._predict_from_frames(str(video_path), frames)

    def predict_window(
        self, video_path: Path | str, start_s: float, end_s: float
    ) -> EventPrediction:
        """Clasifica solo la ventana ``[start_s, end_s]`` de ``video_path``
        (partido completo, ventana deslizante — ver ``core/windowing.py``),
        sin decodificar el vídeo entero (``sample_frames_window``).

        ``EventPrediction.video_path`` sigue siendo la ruta del vídeo fuente;
        la ventana concreta la lleva el ``Window`` del llamador
        (``analytics/engine.py::analyze_full_match``), no esta predicción.
        """
        video_path = Path(video_path)
        frames = sample_frames_window(video_path, start_s, end_s, self.num_frames)
        return self._predict_from_frames(str(video_path), frames)

    def _predict_from_frames(
        self, video_path_label: str, frames: list[np.ndarray]
    ) -> EventPrediction:
        embedding = self._embed(frames)

        probs = self.clf.predict_proba(embedding.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))
        label = str(self.label_encoder.classes_[pred_idx])
        probabilities = {
            str(cls): float(p) for cls, p in zip(self.label_encoder.classes_, probs)
        }

        return EventPrediction(
            video_path=video_path_label,
            label=label,
            confidence=float(probs[pred_idx]),
            probabilities=probabilities,
            is_background=label.startswith(_BACKGROUND_PREFIX),
        )

    def predict_many(
        self, video_paths: Sequence[Path | str]
    ) -> Iterator[EventPrediction]:
        """Clasifica varios clips en secuencia (generador, apto para pipelines)."""
        for video_path in video_paths:
            yield self.predict(video_path)

    # ------------------------------------------------------------------ #

    def _embed(self, frames: list[np.ndarray]) -> np.ndarray:
        """64 frames RGB -> embedding 3072-d (concat de mean/max/std sobre tokens)."""
        torch = self._torch
        inputs = self.processor(videos=frames, return_tensors="pt").to(self.device)
        autocast_kwargs: dict = (
            {"device_type": "cuda", "dtype": torch.float16}
            if self.device == "cuda"
            else {"device_type": "cpu", "enabled": False}
        )
        with torch.no_grad(), torch.autocast(**autocast_kwargs):
            out = self.model(**inputs)

        hidden = out.last_hidden_state.squeeze(0)
        embedding = torch.cat([hidden.mean(dim=0), hidden.max(dim=0)[0], hidden.std(dim=0)])
        return embedding.cpu().numpy()


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def _resolve_sources(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(source.glob("**/*.mp4"))
    return [source]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clasificación de eventos de fútbol (V-JEPA2 + SVC)"
    )
    parser.add_argument("--source", required=True, help="Clip .mp4 o carpeta con .mp4")
    parser.add_argument("--output", default=None, help="JSONL de salida (si no, solo stdout)")
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER_PATH)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    source = Path(args.source)
    if not source.exists():
        parser.error(f"No existe: {source}")
    video_paths = _resolve_sources(source)
    if not video_paths:
        parser.error(f"Ningún .mp4 encontrado en: {source}")

    classifier = EventClassifier(
        classifier_path=args.classifier, model_id=args.model_id, device=args.device
    )

    out_f = open(args.output, "w") if args.output else None
    try:
        for video_path in video_paths:
            try:
                prediction = classifier.predict(video_path)
            except Exception:
                logger.exception("Error procesando %s", video_path)
                continue

            tag = "·bg·" if prediction.is_background else "EVENT"
            logger.info(
                "[%s] %-40s -> %-30s (%.1f%%)",
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
