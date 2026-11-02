"""Goles confirmados en el ``telemetry.json`` del partido completo.

T-DEED apenas emite el gol como evento (1/5 en el Clásico y 0/3 en el
Mallorca a umbral 0,3), pero hay dos pruebas de gol que no dependen de su
curva GOAL y que ``analytics.highlights`` ya calcula:

* el marcador de la retransmisión (``highlights.scoreboard``): cada cambio es
  un gol; si el marcador se lee antes y después de un gol de T-DEED y no
  cambia, ese gol era un tiro;
* sin marcador, el saque de centro tras un tiro (balón en el círculo central
  con los equipos en su mitad según ``tracks_2d.jsonl``, y griterío si el
  audio es informativo). Solo se ve en cámara única (2/3 en el Mallorca, 0/5
  en el Clásico), pero no dio ningún falso en los 4 partidos medidos.

La decisión es la misma de highlights (``pipeline.goal_decision``) sin el
juez VLM, para que un gol del análisis y uno de highlights salgan de la misma
regla. Si no hay marcador legible ni saque de centro, los eventos de T-DEED
quedan como estaban.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from config import HighlightsConfig

    from .engine import EventSpan
    from .highlights.scoreboard import ScoreTimeline

logger = logging.getLogger(__name__)

GOAL_MERGE_S = 10.0     # un tiro/gol ya emitido a esta distancia del gol confirmado es ese mismo gol


def _span_mid(e: "EventSpan") -> float:
    return (e.start_s + e.end_s) / 2.0


def _score_timeline(video: Path, duration_s: float, cache_path: Path, cfg: "HighlightsConfig") -> Optional["ScoreTimeline"]:
    """Marcador leído con el VLM de highlights; ``None`` si el VLM no responde."""
    from .highlights.scoreboard import build_timeline
    from .highlights.vlm import CacheOnlyClient, VLMClient

    client = (CacheOnlyClient(cfg.vlm_model) if cfg.vlm_api == "cache"
              else VLMClient(cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_api, cfg.vlm_api_key, cfg.vlm_timeout_s))
    if not client.ping():
        logger.warning("VLM no disponible en %s: goles sin marcador", cfg.vlm_base_url)
        return None
    try:
        return build_timeline(client, video, duration_s, cfg.scoreboard_step_s, cfg.scoreboard_min_valid,
                              cache_path, cfg.vlm_model)
    except RuntimeError as exc:
        logger.warning("Lectura del marcador interrumpida (%s): goles sin marcador", exc)
        return None


def confirm_goals(
    video: Path, output_dir: Path, events: list["EventSpan"], cfg: "HighlightsConfig", use_scoreboard: bool = True,
) -> tuple[list["EventSpan"], dict]:
    """Devuelve ``(eventos, resumen)``. Lee ``telemetry.json``, ``teams.json``
    y ``tracks_2d.jsonl`` de ``output_dir`` (ya escritos) y deja la caché del
    marcador en ``output_dir/scoreboard.json``, que highlights reutiliza.

    Cada gol confirmado convierte en gol el tiro/gol emitido más cercano
    (``GOAL_MERGE_S``) o, si no hay, añade un tramo nuevo. Los eventos de gol
    llevan ``goal_source``: ``scoreboard``, ``kickoff`` o ``tdeed`` (solo la
    curva GOAL, sin confirmar)."""
    from .engine import SPOT_EVENT_HALF_SPAN_S, EventSpan
    from .highlights.candidates import build_candidates
    from .highlights.pipeline import HighlightInputs, apply_scoreboard, audio_informative, goal_decision
    from .highlights.signals import build_signals

    inputs = HighlightInputs.from_pipeline_layout(video, output_dir)
    sig = build_signals(inputs.video, inputs.telemetry, inputs.teams, inputs.tdeed_scores,
                        inputs.possession, inputs.tracks_2d)
    cands = build_candidates(sig, cfg)
    audio_ok = audio_informative(sig)

    timeline = None
    if use_scoreboard and cfg.scoreboard_enabled:
        timeline = _score_timeline(Path(video), sig.duration_s, Path(output_dir) / "scoreboard.json", cfg)
    score_info = apply_scoreboard(sig, cands, timeline, cfg) if timeline is not None else {}

    goals: list[dict] = []
    for c in cands:
        is_goal, why = goal_decision({}, c.features, audio_ok)
        if is_goal:
            after = score_info.get(c.id, {}).get("after")
            goals.append({"t_s": round(c.anchor_s, 2), "source": "scoreboard" if why == "scoreboard" else "kickoff",
                          "score": after,
                          "confidence": 1.0 if why == "scoreboard" else max(c.features["shot"], c.features["goal"])})
    goals.sort(key=lambda g: g["t_s"])

    events = list(events)
    claimed: set[int] = set()
    for g in goals:
        near = [e for e in events if e.label in ("goal", "shot") and id(e) not in claimed
                and abs(_span_mid(e) - g["t_s"]) <= GOAL_MERGE_S]
        if near:
            e = min(near, key=lambda e: (e.label != "goal", abs(_span_mid(e) - g["t_s"])))
            e.label = "goal"
            e.peak_confidence = max(e.peak_confidence, g["confidence"])
            e.mean_confidence = max(e.mean_confidence, g["confidence"])
        else:
            e = EventSpan("goal", max(0.0, g["t_s"] - SPOT_EVENT_HALF_SPAN_S), g["t_s"] + SPOT_EVENT_HALF_SPAN_S,
                          peak_confidence=g["confidence"], mean_confidence=g["confidence"], n_windows=1)
            events.append(e)
        e.goal_source, e.score = g["source"], g["score"]
        claimed.add(id(e))

    n_demoted = 0
    for e in events:
        if e.label != "goal" or id(e) in claimed:
            continue
        if timeline is not None and timeline.available and timeline.brackets(_span_mid(e)):
            e.label = "shot"            # el marcador se leyó a ambos lados y no cambió
            n_demoted += 1
        else:
            e.goal_source = "tdeed"
    events.sort(key=lambda e: e.start_s)

    logger.info("Goles: %d confirmados (%s), %d goles de T-DEED pasan a tiro por el marcador",
                len(goals), ", ".join(f"{g['t_s']:.0f}s {g['source']}" for g in goals) or "ninguno", n_demoted)
    summary = {
        "scoreboard": timeline.to_dict() if timeline is not None else None,
        "audio_informative": audio_ok,
        "confirmed": goals,
        "n_tdeed_goals_demoted": n_demoted,
    }
    return events, summary


def main() -> None:
    """Aplica la confirmación de goles a un partido ya analizado sin repetir el pipeline::

        python -m analytics.goals --video backend/storage/videos/<id>/source.mp4 \\
            --output-dir backend/storage/outputs/<match>/<part>/output

    La primera vez guarda el original en ``telemetry.pre_goals.json`` y parte
    siempre de él, así que se puede repetir."""
    import argparse
    import json
    import shutil

    import config

    from .engine import EventSpan

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-scoreboard", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    telemetry_path = args.output_dir / "telemetry.json"
    original = args.output_dir / "telemetry.pre_goals.json"
    if not original.exists():
        shutil.copyfile(telemetry_path, original)
    with open(original) as f:
        telemetry = json.load(f)
    events = [EventSpan(**{k: v for k, v in e.items() if k in EventSpan.__dataclass_fields__}) for e in telemetry["events"]]
    events, telemetry["goals"] = confirm_goals(args.video, args.output_dir, events, config.PipelineConfig().highlights,
                                               use_scoreboard=not args.skip_scoreboard)
    telemetry["events"] = [e.to_telemetry() for e in events]
    with open(telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)


if __name__ == "__main__":
    main()
