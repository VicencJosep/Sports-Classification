"""Segunda etapa de la cascada de tiros: pasa ``core.events_vllm.ShotVerifier``
por los candidatos que propone V-JEPA2 (picos de la curva de tiro de
``core/spotting.py`` sobre la caché de ``scripts/scan_event_windows.py``).

Los veredictos se guardan junto al vídeo, uno por candidato e indexados por
instante (``720p.event_windows.vjepa2.w6s3.jsonl`` ->
``720p.shot_verdicts.vjepa2.w6s3.v1.jsonl``, con la versión del prompt al
final). Relanzar con un umbral más bajo solo verifica los candidatos nuevos;
cambiar el prompt exige subir ``SHOT_VERIFIER_PROMPT_VERSION`` para no mezclar
respuestas. ``scripts/evaluate_spotting.py --verdicts`` los puntúa.

Requiere el servidor vLLM (``scripts/serve_events_vllm.sh``), que corre con
``--network host``. Desde el contenedor ``Sports-Classification`` (red bridge)
NO se alcanza: el cortafuegos del host corta bridge -> host:8010. Hay que
correr esto en un contenedor con ``--network host`` (ver
``scripts/run_efl_shot_verifier.sh``) y ``--base-url http://localhost:8010/v1``.

Uso (desde la raíz del proyecto, en un contenedor con --network host):
    python -m scripts.verify_shot_candidates --min-score 0.02 \\
        --base-url http://localhost:8010/v1 \\
        --cache datasets/SoccerNet/england_efl/2019-2020/*/720p.event_windows.vjepa2.w6s3.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import config
from core.spotting import class_curve, find_peaks

logger = logging.getLogger(__name__)


def verdicts_path_for(cache_path: Path, prompt_version: str) -> Path:
    """``X.event_windows.<tag>.jsonl`` -> ``X.shot_verdicts.<tag>.<versión>.jsonl``."""
    name = cache_path.name
    if ".event_windows." not in name:
        raise ValueError(f"No parece una caché de scan_event_windows: {cache_path}")
    stem = name.removesuffix(".jsonl").replace(".event_windows.", ".shot_verdicts.")
    return cache_path.with_name(f"{stem}.{prompt_version}.jsonl")


def read_verdicts(path: Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        meta = json.loads(f.readline())["meta"]
        rows = [json.loads(line) for line in f if line.strip()]
    return meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True, nargs="+", type=Path,
                        help="Caché(s) *.event_windows.*.jsonl de V-JEPA2")
    parser.add_argument("--min-score", type=float, required=True,
                        help="Umbral de candidatos sobre la curva de tiro (ver el barrido de "
                             "evaluate_spotting: bajo = más recall y más llamadas al VLM)")
    parser.add_argument("--nms-s", type=float, default=4.0,
                        help="Igual que --nms-shot de evaluate_spotting (ver allí por qué 4 s)")
    parser.add_argument("--offset-s", type=float, default=0.0,
                        help="Corrección del instante del pico (la que reporta evaluate_spotting)")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from analytics.engine import read_event_windows
    from core.events_vllm import SHOT_VERIFIER_PROMPT_VERSION, ShotVerifier

    vcfg = config.VideoARMEventsConfig()
    verifier = ShotVerifier(
        base_url=args.base_url or vcfg.base_url, model_id=vcfg.model_id, api_key=vcfg.api_key
    )

    for cache in args.cache:
        meta, rows = read_event_windows(cache)
        video = Path(meta["source_video"])
        times, scores = class_curve(rows, "shot")
        candidates = [
            (round(t + args.offset_s, 3), s)
            for t, s in find_peaks(times, scores, args.min_score, args.nms_s)
        ]

        out_path = verdicts_path_for(cache, SHOT_VERIFIER_PROMPT_VERSION)
        done: set[float] = set()
        if out_path.exists():
            _, previous = read_verdicts(out_path)
            done = {round(v["t_s"], 3) for v in previous}
        else:
            with open(out_path, "w") as f:
                f.write(json.dumps({"meta": {
                    "source_video": str(video),
                    "event_windows_cache": str(cache),
                    "prompt_version": SHOT_VERIFIER_PROMPT_VERSION,
                    "model_id": vcfg.model_id,
                    "nms_s": args.nms_s,
                    "before_s": verifier.before_s, "after_s": verifier.after_s,
                    "dense_half_s": verifier.dense_half_s,
                    "n_dense": verifier.n_dense, "n_context": verifier.n_context,
                }}) + "\n")

        pending = [(t, s) for t, s in candidates if t not in done]
        logger.info("%s: %d candidatos (θ=%.3f), %d ya verificados, %d pendientes",
                    video.parent.name, len(candidates), args.min_score, len(candidates) - len(pending),
                    len(pending))

        started = time.perf_counter()
        with open(out_path, "a") as f:
            for i, (t, s) in enumerate(pending, start=1):
                try:
                    verdict = verifier.verify(video, t)
                except Exception:
                    logger.exception("Error verificando %s @ %.1fs", video.name, t)
                    continue
                f.write(json.dumps({"candidate_score": s, **verdict.to_telemetry()}) + "\n")
                f.flush()
                if i % 20 == 0:
                    elapsed = time.perf_counter() - started
                    logger.info("  %d/%d (%.1fs/candidato)", i, len(pending), elapsed / i)
        logger.info("Veredictos en %s", out_path)


if __name__ == "__main__":
    main()
