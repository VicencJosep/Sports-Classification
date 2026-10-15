"""Regenera ``telemetry.json`` de un partido completo desde las cachés: sin GPU, en segundos.

Sirve para comparar umbrales de T-DEED (``action_min_score``, ``throw_in_min_score``,
``shot_min_score``...) sin volver a inferir nada. Lee la caché de probabilidades de T-DEED
(``<vídeo>.tdeed.<modelo>.npz``) y la posesión (``<vídeo>.possession.jsonl``), las dos escritas
por ``main.py --mode full-match`` junto al vídeo, y escribe una telemetría nueva con:

* acciones de balón disparadas por T-DEED con ``--action-min-score`` y etiquetadas por T-DEED
  (equivale a ``--ball-actions-label spotter``: aquí no hay R3D-18 porque exigiría decodificar
  el vídeo entero); el jugador es el poseedor del frame, o el último conocido, igual que en el
  pipeline;
* eventos shot/goal/throw-in de T-DEED (``analytics.engine.spotted_event_spans``) y, para las
  etiquetas que T-DEED no cubre (foul, corner, card, kick-off, penalty, substitution), los
  tramos de V-JEPA2 copiados de ``--telemetry-in``, opcionalmente filtrados por
  ``--events-min-confidence``.

Uso (dentro del contenedor, cwd /app):
    python -m scripts.respot_telemetry --source backend/storage/videos/<id>/source.mp4 \\
        --telemetry-in backend/storage/outputs/<match>/full/output/telemetry.json \\
        --output output/respot/telemetry_0.3.json --action-min-score 0.3
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
    spotted_ball_actions,
    spotted_event_spans,
)
from core.ball_spotting import EVENT_LABELS, read_scores, scores_cache_path

logger = logging.getLogger(__name__)


def _counts(items, key) -> dict[str, int]:
    return dict(sorted(collections.Counter(key(i) for i in items).items()))


def main() -> None:
    cfg = config.PipelineConfig()
    spot_cfg = cfg.ball_spotting
    default_throw_in = spot_cfg.throw_in_min_score if spot_cfg.throw_in_min_score is not None else -1.0

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="vídeo del partido (sus cachés están al lado)")
    parser.add_argument("--telemetry-in", type=Path, default=None,
                        help="telemetry.json anterior del mismo vídeo: de ahí se copian los eventos de V-JEPA2 "
                             "(sin él la telemetría solo lleva los eventos de T-DEED)")
    parser.add_argument("--output", required=True, type=Path, help="telemetry.json de salida")
    parser.add_argument("--model-name", default=spot_cfg.model_name, help="modelo de la caché .tdeed.<modelo>.npz")
    parser.add_argument("--action-min-score", type=float, default=spot_cfg.action_min_score)
    parser.add_argument("--action-nms-s", type=float, default=spot_cfg.action_nms_s)
    parser.add_argument("--shot-min-score", type=float, default=spot_cfg.shot_min_score)
    parser.add_argument("--goal-min-score", type=float, default=spot_cfg.goal_min_score)
    parser.add_argument("--throw-in-min-score", type=float, default=default_throw_in,
                        help="umbral del saque de banda como evento; negativo = dejar el throw-in de V-JEPA2")
    parser.add_argument("--event-nms-s", type=float, default=spot_cfg.event_nms_s)
    parser.add_argument("--event-smooth-frames", type=int, default=spot_cfg.event_smooth_frames)
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

    # Acciones de balón: disparo y etiqueta de T-DEED, jugador de la posesión.
    spotted = spotted_ball_actions(scores, args.action_min_score, args.action_nms_s)
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

    # Eventos: shot/goal(/throw-in) de T-DEED + el resto de V-JEPA2.
    event_params: dict[str, tuple[float, float]] = {
        "SHOT": (args.shot_min_score, args.event_nms_s),
        "GOAL": (args.goal_min_score, args.event_nms_s),
    }
    if args.throw_in_min_score >= 0:
        event_params["THROW IN"] = (args.throw_in_min_score, args.event_nms_s)
    spans = spotted_event_spans(scores, event_params, args.event_smooth_frames)
    replaced = {EVENT_LABELS[raw] for raw in event_params}

    kept: list[EventSpan] = []
    if args.telemetry_in is not None:
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(telemetry.to_telemetry(), indent=2))

    minutes = n_frames / scores.fps / 60 if n_frames else 0.0
    logger.info("Acciones de balón (action_min_score=%.2f): %d, %.1f/min, sin poseedor: %d -> %s",
                args.action_min_score, len(ball_actions), len(ball_actions) / minutes if minutes else 0.0,
                sum(a.tracker_id < 0 for a in ball_actions), _counts(ball_actions, lambda a: a.label))
    logger.info("Eventos: %d -> %s (de T-DEED: %s)", len(events), _counts(events, lambda e: e.label),
                _counts(spans, lambda e: e.label))
    logger.info("Telemetría escrita en %s", args.output)


if __name__ == "__main__":
    main()
