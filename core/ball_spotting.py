"""
core/ball_spotting.py
======================
Detección PUNTUAL de acciones de balón en partido completo con T-DEED
(https://github.com/arturxe2/T-DEED, GPL-3.0, ganador de SoccerNet Ball Action
Spotting 2024): una probabilidad por frame para 12 clases (PASS, DRIVE,
HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER
SUCCESSFUL TACKLE, FREE KICK, GOAL) y, a partir de ella, eventos
``{label, t_s, score}`` por detección de picos (``core/spotting.py``).

Por qué existe, frente a ``core/events.py`` (V-JEPA2 + SVC por ventanas de
6 s) y al disparador por cambio de posesión de ``analytics/engine.py``.
Medido en partido completo:

* Tiros, 3 partidos EFL que el checkpoint ``SoccerNetBall_test`` no vio (69
  tiros): AP ±3 s 75,6% (V-JEPA2 por picos: 34,5%), AP ±1 s 70,0% (6,3%).
  Separa tiro de centro (AUC 0,89 sobre los candidatos de V-JEPA2), que es
  justo donde fallaban V-JEPA2 y los VLM (Qwen3.5-9B, VideoARM, Omni-30B:
  AUC ~0,5).
* Generalización, ``SoccerNetBall_challenge2`` en 6 partidos SoccerNet-v2 de
  6 ligas (130 tiros): AP ±3 s 78,0%; umbral 0,3 -> P 65% / R 84%.
* Pases/conducciones/cabezazos/saques de banda, GT de ``ground_truth/`` (20
  min, 367 acciones, ±1 s): F1 84% (umbral 0,3) frente al 38% del disparador
  por cambio de posesión + R3D-18.

No dice QUÉ jugador hace la acción: eso lo pone ``core/possession.py`` (ver
``analytics.engine._analyze_possession_and_ball_actions``).

El repo de T-DEED no es un paquete instalable: vive clonado en
``models/tdeed/T-DEED`` y se importa añadiéndolo a ``sys.path`` (sus paquetes
``util``/``model``/``dataset`` no chocan con los de este proyecto). Los pesos,
en ``models/tdeed/<modelo>/checkpoint_best.pt``.

La pasada por el vídeo es lo caro (~9-15 min por partido en una 4090) y se
cachea junto al vídeo (``720p.mp4`` -> ``720p.tdeed.<modelo>.npz``), igual que
el tracking; umbrales y NMS se aplican después, sobre la caché.

Uso rápido (desde la raíz del proyecto):
    python -m core.ball_spotting --source partido.mp4 --classes SHOT GOAL
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from core.spotting import SpottedEvent, find_peaks

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_REPO_DIR: Path = _PROJECT_ROOT / "models" / "tdeed" / "T-DEED"
DEFAULT_CHECKPOINTS_DIR: Path = _PROJECT_ROOT / "models" / "tdeed"
DEFAULT_MODEL_NAME: str = "SoccerNetBall_challenge2"

# El repo solo trae config para los dos modelos del challenge; el checkpoint
# "_test" (split train, para evaluar limpio en val/test) usa la del pequeño.
_CONFIG_OF: dict[str, str] = {
    "SoccerNetBall_test": "SoccerNetBall_challenge1",
    "SoccerNetBall_challenge1": "SoccerNetBall_challenge1",
    "SoccerNetBall_challenge2": "SoccerNetBall_challenge2",
}

# Geometría con la que se entrenó T-DEED en SoccerNet Ball: frames a 796x448,
# uno de cada dos (25 fps -> 12,5 fps), clips de 100 frames con solape del 75%.
FRAME_SIZE: tuple[int, int] = (796, 448)
STRIDE: int = 2

BACKGROUND: str = "background"

# Eventos de la línea temporal (vocabulario de core/events.py) -> clases de
# T-DEED cuya probabilidad se suma para formarlos. El checkpoint tiene dos
# cabezas: la de acciones de balón (SoccerNet Ball, MAYÚSCULAS) y la de
# SoccerNet-v2 (17 clases de eventos). Tiro, gol y saque de banda salen de la
# primera, la única medida limpia (AP de tiros 75,6% en 3 partidos EFL no
# vistos); el resto, de la segunda, que cubre todo lo que hacía V-JEPA2
# (tarjeta, córner, falta, saque inicial, penalti, sustitución) y añade falta
# directa/indirecta y fuera de juego. THROW IN está también en
# BALL_ACTION_LABELS a propósito: es a la vez evento y acción atribuida a un
# jugador. Qué eventos se emiten lo decide el ``params`` de spot_events.
EVENT_SOURCES: dict[str, tuple[str, ...]] = {
    "shot": ("SHOT",),
    "goal": ("GOAL",),
    "throw-in": ("THROW IN",),
    "corner": ("Corner",),
    "foul": ("Foul",),
    "free-kick": ("Direct free-kick", "Indirect free-kick"),
    "card": ("Yellow card", "Red card", "Yellow->red card"),
    "kick-off": ("Kick-off",),
    "penalty": ("Penalty",),
    "substitution": ("Substitution",),
    "offside": ("Offside",),
}
BALL_ACTION_LABELS: dict[str, str] = {
    "PASS": "pass", "DRIVE": "drive", "HIGH PASS": "high_pass",
    "HEADER": "header", "THROW IN": "throw_in",
}


@dataclass(frozen=True)
class SpotScores:
    """Probabilidad por frame muestreado de cada clase (caché de un vídeo)."""

    probs: np.ndarray            # [n, 1 + n_clases], columna 0 = background
    classes: tuple[str, ...]     # nombres de columna, classes[0] == "background"
    stride: int                  # la fila i es el frame i * stride del vídeo
    fps: float
    # Cabeza de eventos de SoccerNet-v2 (mismas filas). Las cachés anteriores
    # al 15-oct-2026 no la tienen: ``TDEEDSpotter.ensure_scores`` las rehace.
    sn_probs: Optional[np.ndarray] = None
    sn_classes: tuple[str, ...] = ()

    def frame_indices(self) -> np.ndarray:
        return np.arange(len(self.probs)) * self.stride

    def duration_s(self) -> float:
        return len(self.probs) * self.stride / self.fps

    def curve(self, label: str | Sequence[str]) -> np.ndarray:
        """Probabilidad por fila de una clase de cualquiera de las dos cabezas,
        o la suma de varias (clases excluyentes de una misma softmax: la suma
        es la probabilidad del grupo, p.ej. las tres tarjetas)."""
        if not isinstance(label, str):
            return np.sum([self.curve(one) for one in label], axis=0)
        if label in self.classes:
            return self.probs[:, self.classes.index(label)].astype(np.float32)
        if self.sn_probs is not None and label in self.sn_classes:
            return self.sn_probs[:, self.sn_classes.index(label)].astype(np.float32)
        raise KeyError(f"Clase de T-DEED desconocida o sin cabeza SoccerNet en la caché: {label}")


def scores_cache_path(source_video: Path, model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """``720p.mp4`` -> ``720p.tdeed.SoccerNetBall_challenge2.npz``."""
    return source_video.with_name(f"{source_video.stem}.tdeed.{model_name}.npz")


def read_scores(cache_path: Path | str, fps: Optional[float] = None) -> SpotScores:
    z = np.load(cache_path)
    has_sn = "probs_sn" in z.files
    return SpotScores(
        probs=z["probs"],
        classes=tuple(str(c) for c in z["classes"]),
        stride=int(z["stride"]),
        fps=float(z["fps"]) if "fps" in z.files else float(fps or 25.0),
        sn_probs=z["probs_sn"] if has_sn else None,
        sn_classes=tuple(str(c) for c in z["classes_sn"]) if has_sn else (),
    )


def cache_has_event_head(cache_path: Path | str) -> bool:
    """¿La caché guarda también la cabeza de eventos de SoccerNet-v2?"""
    with np.load(cache_path) as z:
        return "probs_sn" in z.files


def spot(
    scores: SpotScores,
    label: str | Sequence[str],
    min_score: float,
    nms_s: float,
    smooth_frames: int = 1,
) -> list[tuple[int, float]]:
    """Picos de la curva de ``label`` (etiqueta de T-DEED, p.ej. ``"SHOT"``, o
    varias que se suman): ``[(frame_idx, score)]`` ordenados por tiempo. ``smooth_frames`` > 1
    promedia la curva (en filas, es decir, frames muestreados) antes de buscar
    picos: T-DEED reparte la probabilidad de un evento entre 2-3 frames
    vecinos y sin suavizar un mismo tiro puede dar dos máximos."""
    curve = scores.curve(label)
    if smooth_frames > 1:
        curve = np.convolve(curve, np.ones(smooth_frames) / smooth_frames, mode="same")
    frames = scores.frame_indices()
    times = (frames / scores.fps).tolist()
    by_time = {round(t, 6): int(f) for t, f in zip(times, frames)}
    return [(by_time[round(t, 6)], float(s)) for t, s in find_peaks(times, curve.tolist(), min_score, nms_s)]


def spot_events(
    scores: SpotScores,
    params: Mapping[str, tuple[float, float]],
    smooth_frames: int = 1,
) -> list[SpottedEvent]:
    """Eventos puntuales de la línea temporal. ``params``: ``{evento del
    proyecto (clave de EVENT_SOURCES): (min_score, nms_s)}``. Los eventos de
    la cabeza SoccerNet se saltan (con aviso) si la caché no la tiene."""
    events: list[SpottedEvent] = []
    for label, (min_score, nms_s) in params.items():
        sources = EVENT_SOURCES[label]
        if scores.sn_probs is None and any(src not in scores.classes for src in sources):
            logger.warning("La caché de T-DEED no tiene la cabeza SoccerNet: sin eventos '%s'", label)
            continue
        events.extend(
            SpottedEvent(label=label, t_s=frame / scores.fps, score=score)
            for frame, score in spot(scores, sources, min_score, nms_s, smooth_frames)
        )
    events.sort(key=lambda e: e.t_s)
    return events


class TDEEDSpotter:
    """T-DEED cargado en memoria. Solo hace falta para calcular la caché de
    un vídeo (``ensure_scores``); leerla y sacar picos no necesita GPU.

    Args:
        model_name: carpeta del checkpoint bajo ``checkpoints_dir``.
            ``SoccerNetBall_challenge2`` (RegNetY-800MF, entrenado con 6 de los
            7 partidos EFL) es el de producción; ``SoccerNetBall_test``
            (RegNetY-200MF, split train) es el que permite evaluar limpio en
            Middlesbrough, Stoke y Reading.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        repo_dir: Path | str = DEFAULT_REPO_DIR,
        checkpoints_dir: Path | str = DEFAULT_CHECKPOINTS_DIR,
    ) -> None:
        import torch

        repo_dir = Path(repo_dir)
        checkpoint = Path(checkpoints_dir) / model_name / "checkpoint_best.pt"
        if not repo_dir.exists():
            raise FileNotFoundError(f"Repo de T-DEED no encontrado: {repo_dir}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint de T-DEED no encontrado: {checkpoint}")
        if model_name not in _CONFIG_OF:
            raise ValueError(f"Modelo T-DEED desconocido: {model_name} (conocidos: {sorted(_CONFIG_OF)})")

        self.model_name = model_name
        self._torch = torch
        self._repo_dir = repo_dir
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        # El repo lee config y ficheros de clases con rutas relativas a su raíz.
        with contextlib.chdir(repo_dir):
            from model.model import TDEEDModel
            from util.dataset import load_classes
            from util.io import load_json

            cfg = load_json(os.path.join("config", "SoccerNetBall", f"{_CONFIG_OF[model_name]}.json"))
            args = argparse.Namespace(**cfg)
            args.crop_dim = None if cfg["crop_dim"] <= 0 else cfg["crop_dim"]
            classes = load_classes(os.path.join("data", "soccernetball", "class.txt"))
            pretrain_classes = load_classes(os.path.join("data", cfg["pretrain"]["dataset"], "class.txt"))

            logger.info("Cargando T-DEED (%s) ...", model_name)
            self._model = TDEEDModel(args=args)
            # Entrenado a la vez con SoccerNet (17 clases): dos cabezas. La de
            # acciones de balón es la primera.
            n_classes = [len(classes) + 1, len(pretrain_classes) + 1]
            self._model._model.update_pred_head(n_classes)
            self._model._num_classes = int(np.sum(n_classes))
            self._model.load(torch.load(checkpoint))

        self.classes: tuple[str, ...] = (BACKGROUND,) + tuple(
            name for name, _ in sorted(classes.items(), key=lambda kv: kv[1])
        )
        self.sn_classes: tuple[str, ...] = (BACKGROUND,) + tuple(
            name for name, _ in sorted(pretrain_classes.items(), key=lambda kv: kv[1])
        )
        self._clip_len: int = cfg["clip_len"]
        self._batch_size: int = cfg["batch_size"]

    def ensure_scores(self, source_video: Path | str, cache_path: Optional[Path] = None) -> Path:
        """Calcula (si no existe) la caché de probabilidades de ``source_video``
        y devuelve su ruta. Escribe a un temporal y renombra al acabar, así que
        una pasada interrumpida no deja una caché a medias."""
        import cv2
        from torch.utils.data import DataLoader

        source_video = Path(source_video)
        cache_path = Path(cache_path) if cache_path else scores_cache_path(source_video, self.model_name)
        if cache_path.exists():
            if cache_has_event_head(cache_path):
                logger.info("Caché de T-DEED encontrada: %s", cache_path.name)
                return cache_path
            logger.info("Caché de T-DEED sin la cabeza de eventos (anterior al 15-oct-2026): se rehace %s",
                        cache_path.name)

        from dataset.frame import ActionSpotInferenceDataset

        cap = cv2.VideoCapture(str(source_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        dataset = ActionSpotInferenceDataset(
            str(source_video), clip_len=self._clip_len, overlap_len=self._clip_len // 4 * 3,
            stride=STRIDE, dataset="soccernetball", size=FRAME_SIZE,
        )
        loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False, num_workers=1, pin_memory=True)
        n = dataset._video_len // STRIDE
        heads = (len(self.classes), len(self.sn_classes))
        acc = [np.zeros((n, k), np.float32) for k in heads]
        support = np.zeros(n, np.int32)

        logger.info("T-DEED: escaneando %s (%d frames, dos cabezas) ...", source_video.name, dataset._video_len)
        started = time.perf_counter()
        for frames, starts in loader:
            batch_heads = self._predict_heads(frames)
            for i, start in enumerate(starts.tolist()):
                lo = -start if start < 0 else 0  # relleno inicial del dataset
                s0 = max(start, 0)
                end = min(s0 + batch_heads[0].shape[1] - lo, n)
                for h in range(2):
                    acc[h][s0:end] += batch_heads[h][i][lo: lo + end - s0]
                support[s0:end] += 1
        div = np.maximum(support, 1)[:, None]
        logger.info("T-DEED: %s escaneado en %.0fs", source_video.name, time.perf_counter() - started)

        tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
        np.savez(tmp_path, probs=(acc[0] / div).astype(np.float16), classes=np.array(self.classes),
                 probs_sn=(acc[1] / div).astype(np.float16), classes_sn=np.array(self.sn_classes),
                 stride=STRIDE, fps=fps)
        tmp_path.rename(cache_path)
        return cache_path

    def _predict_heads(self, frames) -> list[np.ndarray]:
        """Softmax de cada cabeza y recolocación temporal con el desplazamiento
        predicho (máximo), como ``model.modules.process_double_head`` del repo
        pero para las DOS cabezas: ``model.predict`` descarta la de SoccerNet.
        Para la primera da lo mismo que ``model.predict`` (comprobado en
        scripts/shot_benchmark/tdeed_scan_heads.py con ``--max-clips``)."""
        torch = self._torch
        net = self._model._model
        seq = frames.to(self._model.device).float()
        net.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred, _ = net(seq, inference=True)
        logits, displ = pred["im_feat"], pred["displ_feat"]
        if isinstance(logits, list):
            logits = logits[0]
        if isinstance(displ, list):
            displ = displ[0]
        logits, displ = logits.float(), displ.float()
        batch, length = logits.shape[:2]
        d = displ.reshape(batch, length, -1)[:, :, 0].round().long()
        target = (torch.arange(length, device=d.device)[None] - d).clamp(0, length - 1)
        out = []
        k1 = len(self.classes)
        for sl in (slice(0, k1), slice(k1, k1 + len(self.sn_classes))):
            p = torch.softmax(logits[:, :, sl], dim=2)
            agg = torch.zeros_like(p).scatter_reduce(
                1, target[:, :, None].expand_as(p), p, reduce="amax", include_self=True
            )
            out.append(agg.cpu().numpy())
        return out

    def release(self) -> None:
        """Libera la VRAM del modelo (el resto del pipeline la necesita)."""
        self._model = None
        self._torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Detección de acciones de balón con T-DEED")
    parser.add_argument("--source", required=True, type=Path, help="Vídeo de partido")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--classes", nargs="+", default=["SHOT", "GOAL"], help="Etiquetas de T-DEED a listar")
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--nms-s", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=None, help="JSON de salida (si no, solo stdout)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cache_path = scores_cache_path(args.source, args.model)
    if not cache_path.exists():
        TDEEDSpotter(model_name=args.model).ensure_scores(args.source)
    scores = read_scores(cache_path)

    events = []
    for label in args.classes:
        for frame, score in spot(scores, label, args.min_score, args.nms_s, smooth_frames=6):
            events.append({"label": label, "frame_idx": frame, "t_s": round(frame / scores.fps, 2), "score": round(score, 3)})
    events.sort(key=lambda e: e["frame_idx"])
    for e in events:
        print(f"{int(e['t_s'] // 60):3d}:{e['t_s'] % 60:04.1f}  {e['label']:<10} {e['score']:.2f}")
    if args.output:
        args.output.write_text(json.dumps(events, indent=1))


if __name__ == "__main__":
    main()
