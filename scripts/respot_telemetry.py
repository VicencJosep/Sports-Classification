"""Regenera ``telemetry.json`` de un partido completo desde las cachés: sin GPU, en segundos.

Sirve para comparar umbrales de T-DEED (automáticos o fijos, por acción y por evento) sin volver
a inferir nada. Lee la caché de probabilidades de T-DEED
(``<vídeo>.tdeed.<modelo>.npz``) y la posesión (``<vídeo>.possession.jsonl``), las dos escritas
por ``main.py --mode full-match`` junto al vídeo, y escribe una telemetría nueva con:

* acciones de balón disparadas por T-DEED con ``--action-min-score`` y etiquetadas por T-DEED
  (equivale a ``--ball-actions-label spotter``: aquí no hay R3D-18 porque exigiría decodificar
  el vídeo entero); el jugador es el poseedor del frame, o el último conocido, igual que en el
  pipeline;
* eventos de T-DEED (``analytics.engine.spotted_event_spans``): todos con ``--event-source tdeed``
  (por defecto; la caché debe tener la cabeza de SoccerNet, ver ``core.ball_spotting``) o solo
  shot/goal/throw-in con ``mixed``, que copia el resto de tramos de V-JEPA2 de ``--telemetry-in``
  (opcionalmente filtrados por ``--events-min-confidence``).

Uso (dentro del contenedor, cwd /app):
    python -m scripts.respot_telemetry --source backend/storage/videos/<id>/source.mp4 \\
        --telemetry-in backend/storage/outputs/<match>/full/output/telemetry.json \\
        --output output/respot/telemetry_0.3.json --thresholds fixed --action-min-score 0.3 \\
        --event-min-score shot=0.2 foul=0.4
    python -m scripts.evaluate_pipeline --gt ground_truth/x.gt.json --telemetry output/respot/telemetry_0.3.json
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path

import config
from analytics.engine import (
    EventSpan,
    FullMatchTelemetry,
    _analyze_possession_and_ball_actions,
    calibrate_spot_thresholds,
    fixed_spot_thresholds,
    spotted_ball_actions,
    spotted_event_spans,
)
from core.ball_spotting import read_scores, scores_cache_path

logger = logging.getLogger(__name__)


def _counts(items, key) -> dict[str, int]:
    return dict(sorted(collections.Counter(key(i) for i in items).items()))


def main() -> None:
    from dataclasses import asdict, replace

    cfg = config.PipelineConfig()
    spot_cfg = cfg.ball_spotting

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="vídeo del partido (sus cachés están al lado)")
    parser.add_argument("--telemetry-in", type=Path, default=None,
                        help="telemetry.json anterior del mismo vídeo: con --event-source mixed, de ahí se "
                             "copian los eventos de V-JEPA2 que T-DEED no pone")
    parser.add_argument("--output", required=True, type=Path, help="telemetry.json de salida")
    parser.add_argument("--model-name", default=spot_cfg.model_name, help="modelo de la caché .tdeed.<modelo>.npz")
    parser.add_argument("--event-source", choices=["tdeed", "mixed"], default=spot_cfg.event_source)
    parser.add_argument("--thresholds", choices=["auto", "fixed"],
                        default="auto" if spot_cfg.auto_thresholds else "fixed")
    parser.add_argument("--action-min-score", type=float, default=None, help="fija el umbral de acciones")
    parser.add_argument("--event-min-score", nargs="*", default=[], metavar="EVENTO=UMBRAL",
                        help="fija el umbral de esos eventos (p.ej. shot=0.2 card=0.6)")
    parser.add_argument("--events-min-confidence", type=float, default=None,
                        help="descarta los tramos de V-JEPA2 copiados cuyo peak_confidence sea menor")
    parser.add_argument("--fps", type=float, default=None, help="solo si la caché de T-DEED no guarda fps")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache_path = scores_cache_path(args.source, args.model_name)
    tracks_path = args.source.with_suffix(".tracks.jsonl")
    for path in (cache_path, tracks_path):
        if not path.exists():
            raise SystemExit(f"Falta la caché {path}: este script no infiere nada, corre antes main.py --mode full-match")
    scores = read_scores(cache_path, fps=args.fps)

    labels = spot_cfg.tdeed_event_labels if args.event_source == "tdeed" else spot_cfg.mixed_event_labels
    thresholds = (calibrate_spot_thresholds(scores, spot_cfg, labels) if args.thresholds == "auto"
                  else fixed_spot_thresholds(spot_cfg, labels))
    overrides = dict(item.split("=", 1) for item in args.event_min_score)
    thresholds = replace(
        thresholds,
        action_min_score=thresholds.action_min_score if args.action_min_score is None else args.action_min_score,
        event_min_scores={**thresholds.event_min_scores, **{k: float(v) for k, v in overrides.items()}},
    )
    logger.info("Umbrales (%s): acciones %.3f, factor de dominio %.2f, %d min de juego; eventos %s",
                thresholds.mode, thresholds.action_min_score, thresholds.domain_factor,
                thresholds.active_minutes, thresholds.event_min_scores)

    # Acciones de balón: disparo y etiqueta de T-DEED, jugador de la posesión.
    spotted = spotted_ball_actions(scores, thresholds.action_min_score, spot_cfg.action_nms_s)
    ball_actions, _possessor_by_frame, n_frames = _analyze_possession_and_ball_actions(
        source_video=args.source,
        tracks_jsonl_path=tracks_path,
        ball_action_classifier=None,
        max_possessor_distance_px=cfg.analytics.max_possessor_distance_px,
        hysteresis_frames=cfg.analytics.hysteresis_frames,
        temporal_offsets_frames=cfg.ball_actions.temporal_offsets_frames,
        spotted_actions=spotted,
        label_source="spotter",
    )

    nms_of = {label: nms for label, _, nms in spot_cfg.event_params}
    event_params = {label: (thresholds.event_min_scores[label], nms_of[label]) for label in labels}
    spans = spotted_event_spans(scores, event_params, spot_cfg.event_smooth_frames)
    replaced = set(event_params)

    kept: list[EventSpan] = []
    if args.telemetry_in is not None and args.event_source == "mixed":
        previous = json.loads(args.telemetry_in.read_text()).get("events", [])
        dropped_label = dropped_conf = 0
        for e in previous:
            if e["label"] in replaced:
                dropped_label += 1
                continue
            if args.events_min_confidence is not None and e["peak_confidence"] < args.events_min_confidence:
                dropped_conf += 1
                continue
            kept.append(EventSpan(**e))
        logger.info(
            "Eventos de V-JEPA2 copiados de %s: %d (descartados: %d por etiqueta sustituida por T-DEED, %d por confianza)",
            args.telemetry_in, len(kept), dropped_label, dropped_conf,
        )

    events = sorted(kept + spans, key=lambda e: e.start_s)
    telemetry = FullMatchTelemetry(
        source_video=str(args.source),
        events=events,
        ball_actions=ball_actions,
        n_frames=n_frames,
        n_possession_changes=len(ball_actions),
        spot_thresholds=asdict(thresholds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(telemetry.to_telemetry(), indent=2))

    minutes = n_frames / scores.fps / 60 if n_frames else 0.0
    logger.info("Acciones de balón: %d, %.1f/min, sin poseedor: %d -> %s",
                len(ball_actions), len(ball_actions) / minutes if minutes else 0.0,
                sum(a.tracker_id < 0 for a in ball_actions), _counts(ball_actions, lambda a: a.label))
    logger.info("Eventos: %d -> %s (de T-DEED: %s)", len(events), _counts(events, lambda e: e.label),
                _counts(spans, lambda e: e.label))
    logger.info("Telemetría escrita en %s", args.output)


if __name__ == "__main__":
    main()
