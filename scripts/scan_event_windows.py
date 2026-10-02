"""Escanea partidos completos con ventanas deslizantes y cachea la predicción
cruda de cada ventana (``analytics.engine.ensure_event_windows``).

Es el paso caro (~26 min de GPU por partido de 96 min con V-JEPA2); una vez
escrita la caché, ``core/spotting.py`` y ``scripts/evaluate_spotting.py``
trabajan solo sobre ella y corren en segundos.

La caché va junto a cada vídeo (``720p.mp4`` ->
``720p.event_windows.vjepa2.w6s3.jsonl``). Un vídeo que ya la tiene se salta;
uno con un ``.tmp`` a medias se reanuda (ver ``ensure_event_windows``).

Uso (desde la raíz del proyecto, dentro del contenedor):
    # Los 7 partidos EFL con Labels-ball.json
    python -m scripts.scan_event_windows \\
        --video datasets/SoccerNet/england_efl/2019-2020/*/720p.mp4

    # El VLM como escáner de tiros (experimento: ¿qué ve que V-JEPA2 no propone?)
    python -m scripts.scan_event_windows --backend vlm-shot --video partido.mp4

    # Prueba rápida: solo las primeras N ventanas, a un fichero aparte
    python -m scripts.scan_event_windows --video partido.mp4 --max-windows 20 \\
        --output /tmp/prueba.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import config
from analytics.engine import ensure_event_windows, iter_event_window_predictions

logger = logging.getLogger(__name__)

BACKENDS: tuple[str, ...] = ("vjepa2", "vlm-shot")


def _make_classifier_factory(backend: str, device: str | None, base_url: str | None):
    """Devuelve una función que construye el clasificador la primera vez y
    reutiliza la misma instancia después (varios vídeos, un solo modelo)."""
    instance = None

    def build():
        nonlocal instance
        if instance is None:
            if backend == "vjepa2":
                from core.events import EventClassifier

                ecfg = config.EventsConfig()
                instance = EventClassifier(
                    classifier_path=ecfg.classifier_path,
                    model_id=ecfg.model_id,
                    device=device or ecfg.device,
                    num_frames=ecfg.num_frames,
                )
            elif backend == "vlm-shot":
                from core.events_vllm import ShotVerifier, VLMShotWindowClassifier

                vcfg = config.VideoARMEventsConfig()
                instance = VLMShotWindowClassifier(ShotVerifier(
                    base_url=base_url or vcfg.base_url, model_id=vcfg.model_id, api_key=vcfg.api_key,
                ))
            else:
                raise ValueError(f"Backend desconocido: {backend}")
        return instance

    return build


def _meta_extra(backend: str) -> dict:
    if backend == "vjepa2":
        ecfg = config.EventsConfig()
        return {"classifier_path": str(ecfg.classifier_path), "model_id": ecfg.model_id}
    if backend == "vlm-shot":
        from core.events_vllm import SHOT_VERIFIER_PROMPT_VERSION

        return {"model_id": config.VideoARMEventsConfig().model_id,
                "prompt_version": SHOT_VERIFIER_PROMPT_VERSION}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, nargs="+", type=Path, help="Vídeo(s) de partido completo")
    parser.add_argument("--backend", choices=BACKENDS, default="vjepa2")
    parser.add_argument("--window-s", type=float, default=config.EventsConfig().window_s)
    parser.add_argument("--stride-s", type=float, default=config.FullMatchConfig().event_stride_s)
    parser.add_argument("--min-window-fraction", type=float,
                        default=config.FullMatchConfig().min_window_fraction)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base-url", default=None,
                        help="Solo vlm-shot: servidor vLLM (por defecto el de config.VideoARMEventsConfig)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Solo para pruebas: escanea las primeras N ventanas y escribe a "
                             "--output (no toca la caché real)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Con --max-windows: fichero de salida de la prueba")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    missing = [v for v in args.video if not v.exists()]
    if missing:
        parser.error(f"No existen: {', '.join(map(str, missing))}")

    build = _make_classifier_factory(args.backend, args.device, args.base_url)

    if args.max_windows is not None:
        if args.output is None or len(args.video) != 1:
            parser.error("--max-windows necesita --output y un único --video")
        import time

        classifier = build()
        started = time.perf_counter()
        with open(args.output, "w") as f:
            for window, pred in itertools.islice(
                iter_event_window_predictions(
                    args.video[0], classifier, window_s=args.window_s, stride_s=args.stride_s,
                    min_window_fraction=args.min_window_fraction,
                ),
                args.max_windows,
            ):
                f.write(json.dumps({
                    "index": window.index, "start_s": window.start_s, "end_s": window.end_s,
                    "label": pred.label, "confidence": pred.confidence,
                }) + "\n")
        elapsed = time.perf_counter() - started
        logger.info("%d ventanas en %.1fs (%.2fs/ventana) -> %s",
                    args.max_windows, elapsed, elapsed / args.max_windows, args.output)
        return

    for video in args.video:
        ensure_event_windows(
            video, build, backend=args.backend,
            window_s=args.window_s, stride_s=args.stride_s,
            min_window_fraction=args.min_window_fraction,
            meta_extra=_meta_extra(args.backend),
        )


if __name__ == "__main__":
    main()
