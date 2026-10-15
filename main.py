"""
main.py
=======
Orquestador de nivel superior del pipeline de fútbol: construye
``config.PipelineConfig`` y encadena los módulos de ``core/`` +
``analytics/engine.py`` en un único comando, para los dos flujos que hoy
existen como funciones independientes:

  --mode clip         Clip corto ya recortado en torno a un evento (p.ej.
                       ``datasets/clips_analisis/``). Sustituye la cadena
                       manual de 3 comandos (tracking -> classification ->
                       analytics.engine) por uno solo, invocando exactamente
                       las mismas funciones (``analyze_match`` +
                       ``render_match_video``, sin cambios).
  --mode full-match    Partido completo (~90 min). Usa
                       ``analytics.engine.analyze_full_match`` (tracking
                       cacheado + posesión/acciones de balón en streaming +
                       eventos por ventana deslizante) y añade clasificación
                       de equipos + render, orquestados aquí.

Convención de rutas: los cachés (``*.tracks.jsonl``, ``*.possession.jsonl``)
se guardan junto al vídeo fuente (ver ``core.tracking.ensure_tracking`` /
``core.possession.ensure_possession``); los entregables (``teams.json``,
``telemetry.json``, ``final.mp4``) van a ``--output-dir``.

Uso rápido (desde la raíz del proyecto):
    python -m main --source clip.mp4 --output-dir output/miclip --mode clip
    python -m main --source partido.mp4 --output-dir output/mipartido --mode full-match
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import config

if TYPE_CHECKING:
    from core.ball_actions import BallActionClassifier
    from core.events import EventClassifier

logger = logging.getLogger(__name__)


def _build_ball_action_classifier(
    cfg: config.PipelineConfig, device: Optional[str], skip: bool
) -> Optional["BallActionClassifier"]:
    if skip:
        return None
    from core.ball_actions import BallActionClassifier

    return BallActionClassifier(checkpoint_path=cfg.ball_actions.checkpoint_path, device=device)


def _build_event_classifier(
    cfg: config.PipelineConfig, device: Optional[str], skip: bool
) -> Optional["EventClassifier"]:
    if skip:
        return None
    from core.events import EventClassifier

    return EventClassifier(
        classifier_path=cfg.events.classifier_path, model_id=cfg.events.model_id, device=device
    )


def _build_team_by_tracker_id(
    source_video: Path,
    tracks_path: Path,
    teams_path: Path,
    cfg: config.PipelineConfig,
    device: Optional[str],
    skip: bool,
) -> Optional[dict[int, int]]:
    """Corre core.classification.assign_teams si teams_path no existe ya
    (mismo patrón de caché que ensure_tracking/ensure_possession, aunque aquí
    no es atómico porque assign_teams no es tan costoso como tracking/eventos
    y el fichero es pequeño). Devuelve None si se pide --skip-classification."""
    if skip:
        return None

    if not teams_path.exists():
        from core.classification import PRTReIdEmbedder, StubEmbedder, assign_teams

        embedder = (
            PRTReIdEmbedder(
                weights_path=cfg.classification.reid_weights_path,
                backbone_weights_path=cfg.classification.reid_backbone_weights_path,
                device=device,
                image_size=cfg.classification.image_size,
            )
            if cfg.classification.embedder == "prtreid"
            else StubEmbedder()
        )
        result = assign_teams(
            source_video=source_video,
            tracks_jsonl_path=tracks_path,
            embedder=embedder,
            embedder_name=cfg.classification.embedder,
            n_clusters=cfg.classification.n_clusters,
            random_state=cfg.classification.random_state,
            batch_size=cfg.classification.batch_size,
        )
        with open(teams_path, "w") as f:
            json.dump(result.to_telemetry(), f, indent=2)
        logger.info("Equipos asignados y escritos en %s", teams_path)
    else:
        logger.info("teams.json ya existe, reutilizando: %s", teams_path)

    teams_data = json.loads(teams_path.read_text())
    return {t["tracker_id"]: t["team_cluster"] for t in teams_data["teams"]}


def _run_homography(
    source_video: Path,
    tracks_path: Path,
    output_dir: Path,
    cfg: config.PipelineConfig,
    args: argparse.Namespace,
) -> Optional[Path]:
    """Calibra el campo y proyecta el tracking a metros.

    Etapa opcional e independiente del resto del análisis: produce
    ``tracks_2d.jsonl`` (tracking enriquecido con ``field_xy``) y, si se pide,
    ``radar.mp4``. No modifica ``telemetry.json`` — la posesión y las acciones
    de balón siguen calculándose en píxeles hasta que se migren de forma
    deliberada (cambiar el umbral de posesión a metros altera los resultados
    medidos contra ground_truth/ y merece su propia evaluación).

    La caché cara (``<video>.homography.jsonl``) se guarda junto al vídeo, como
    tracks/possession; ``tracks_2d.jsonl`` es entregable y va a --output-dir.
    """
    if args.skip_homography:
        return None

    from dataclasses import replace

    from core.homography import (
        ensure_homography, project_tracks, render_radar_video, video_fps,
    )

    hcfg = cfg.homography
    if args.homography_stride is not None:
        hcfg = replace(hcfg, stride=args.homography_stride)

    homography_path = ensure_homography(source_video, hcfg, device=args.device)
    tracks_2d_path = output_dir / "tracks_2d.jsonl"
    project_tracks(tracks_path, homography_path, tracks_2d_path,
                   max_gap_frames=hcfg.max_gap_frames)

    if args.radar:
        teams_path = output_dir / "teams.json"
        team_by_tracker_id = None
        if teams_path.exists():
            teams_data = json.loads(teams_path.read_text())
            team_by_tracker_id = {
                t["tracker_id"]: t["team_cluster"] for t in teams_data["teams"]
            }
        render_radar_video(tracks_2d_path, output_dir / "radar.mp4",
                           fps=video_fps(source_video),
                           team_by_tracker_id=team_by_tracker_id)
        logger.info("Radar escrito en %s", output_dir / "radar.mp4")

    # Los dos HRNet-W48 ocupan varios GB de VRAM y ya no hacen falta; el
    # FieldCalibrator vive dentro de ensure_homography, así que aquí ya está
    # libre y solo queda devolver la memoria reservada por el allocator antes
    # de cargar R3D-18 / V-JEPA2.
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tracks_2d_path


def _run_clip(args: argparse.Namespace, cfg: config.PipelineConfig) -> None:
    from analytics.engine import analyze_match, render_match_video
    from core.tracking import ensure_tracking

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = ensure_tracking(args.source, detector_name=args.detector, device=args.device)

    team_by_tracker_id = _build_team_by_tracker_id(
        args.source, tracks_path, output_dir / "teams.json", cfg, args.device, args.skip_classification
    )
    # Antes de cargar R3D-18/V-JEPA2: la calibración es la etapa más pesada en
    # VRAM y conviene que tenga la GPU para ella sola.
    _run_homography(args.source, tracks_path, output_dir, cfg, args)

    ball_action_classifier = _build_ball_action_classifier(cfg, args.device, args.skip_ball_actions)
    event_classifier = _build_event_classifier(cfg, args.device, args.skip_events)

    telemetry, possession_records = analyze_match(
        source_video=args.source,
        tracks_jsonl_path=tracks_path,
        ball_action_classifier=ball_action_classifier,
        event_classifier=event_classifier,
        max_possessor_distance_px=cfg.analytics.max_possessor_distance_px,
        hysteresis_frames=cfg.analytics.hysteresis_frames,
    )
    logger.info(
        "Análisis: %d cambios de posesión, %d acciones de balón, evento=%s",
        telemetry.n_possession_changes, len(telemetry.ball_actions),
        telemetry.event["label"] if telemetry.event else "(omitido)",
    )

    telemetry_path = output_dir / "telemetry.json"
    with open(telemetry_path, "w") as f:
        json.dump(telemetry.to_telemetry(), f, indent=2)
    logger.info("Telemetría escrita en %s", telemetry_path)

    if not args.skip_render:
        possessor_by_frame = {
            r["frame_idx"]: r.get("possessor_tracker_id") for r in possession_records
        }
        final_path = output_dir / "final.mp4"
        render_match_video(
            source_video=args.source,
            tracks_jsonl_path=tracks_path,
            output_path=final_path,
            team_by_tracker_id=team_by_tracker_id,
            possessor_by_frame=possessor_by_frame,
            telemetry=telemetry,
            trace_length=cfg.analytics.trace_length if args.trace_length is None else args.trace_length,
            action_highlight_frames=cfg.analytics.action_highlight_frames,
        )
        logger.info("Vídeo final escrito en %s", final_path)

        if args.publish:
            _run_publish(final_path, args, cfg)


def _run_publish(
    video_path: Path, args: argparse.Namespace, cfg: config.PipelineConfig
) -> None:
    """Último eslabón del pipeline: sube ``final.mp4`` a YouTube Studio.

    Import perezoso (como el resto de módulos pesados de main.py): quien no use
    ``--publish`` no paga ni Playwright ni la carga de MolmoWeb-4B. El resultado
    se escribe junto a los demás entregables, en ``--output-dir``.
    """
    from automation.molmo_publisher import publish_clip

    pub = cfg.publisher
    result = publish_clip(
        video_path,
        args.publish_title,
        visibility=args.publish_visibility,
        model_id=pub.model_id,
        device=args.device or pub.device,
        dtype=pub.dtype,
        max_new_tokens=pub.max_new_tokens,
        venv_python=pub.venv_python,
        user_data_dir=pub.user_data_dir,
        headless=pub.headless,
        viewport={"width": pub.viewport_width, "height": pub.viewport_height},
        settle_ms=pub.settle_ms,
        modal_settle_ms=pub.modal_settle_ms,
        upload_settle_ms=pub.upload_settle_ms,
        dry_run=args.publish_dry_run,
        debug_dir=args.output_dir / "publish_debug",
        studio_url=pub.studio_url,
    )

    publish_path = args.output_dir / "publish.json"
    with open(publish_path, "w") as f:
        json.dump(result.to_telemetry(), f, indent=2, ensure_ascii=False)

    if result.error:
        # No se relanza: el análisis (telemetry.json + final.mp4) ya está en
        # disco y es el entregable de verdad; la publicación es un extra.
        logger.error("Publicación fallida: %s (detalle en %s)", result.error, publish_path)
    else:
        logger.info(
            "%s -> %s", "Publicado" if result.published else "Dry-run completado", publish_path
        )


def _build_spot_kwargs(args: argparse.Namespace, cfg: config.PipelineConfig) -> dict:
    """Argumentos de T-DEED para ``analyze_full_match`` (vacío si no se usa).
    Calcula la caché de probabilidades del vídeo si no existe y libera el
    modelo antes de volver: el resto del pipeline necesita esa VRAM."""
    use_for_events = args.shot_backend == "tdeed" and not args.skip_events
    use_for_actions = args.ball_actions_trigger == "tdeed" and not args.skip_ball_actions
    if not (use_for_events or use_for_actions):
        return {}

    from core.ball_spotting import TDEEDSpotter, read_scores, scores_cache_path

    spot_cfg = cfg.ball_spotting
    cache_path = scores_cache_path(args.source, spot_cfg.model_name)
    if not cache_path.exists():
        spotter = TDEEDSpotter(spot_cfg.model_name, spot_cfg.repo_dir, spot_cfg.checkpoints_dir)
        spotter.ensure_scores(args.source, cache_path)
        spotter.release()

    kwargs: dict = {"spot_scores": read_scores(cache_path)}
    if use_for_events:
        kwargs["event_spot_params"] = {
            "SHOT": (spot_cfg.shot_min_score, spot_cfg.event_nms_s),
            "GOAL": (spot_cfg.goal_min_score, spot_cfg.event_nms_s),
        }
        if spot_cfg.throw_in_min_score is not None:
            # El saque de banda también sale de T-DEED; analyze_full_match
            # descarta entonces los tramos "throw-in" de V-JEPA2.
            kwargs["event_spot_params"]["THROW IN"] = (spot_cfg.throw_in_min_score, spot_cfg.event_nms_s)
        kwargs["event_spot_smooth_frames"] = spot_cfg.event_smooth_frames
    if use_for_actions:
        kwargs["ball_action_spot_params"] = (spot_cfg.action_min_score, spot_cfg.action_nms_s)
        kwargs["ball_action_label_source"] = args.ball_actions_label
    return kwargs


def _apply_spot_overrides(args: argparse.Namespace, cfg: config.PipelineConfig) -> config.PipelineConfig:
    """``--action-min-score`` / ``--throw-in-min-score`` -> copia de ``cfg`` con
    esos umbrales de T-DEED (los dataclasses de config son frozen). Un
    ``--throw-in-min-score`` negativo deja el saque de banda de V-JEPA2."""
    from dataclasses import replace

    changes: dict = {}
    if args.action_min_score is not None:
        changes["action_min_score"] = args.action_min_score
    if args.throw_in_min_score is not None:
        changes["throw_in_min_score"] = None if args.throw_in_min_score < 0 else args.throw_in_min_score
    if not changes:
        return cfg
    return replace(cfg, ball_spotting=replace(cfg.ball_spotting, **changes))


def _run_full_match(args: argparse.Namespace, cfg: config.PipelineConfig) -> None:
    from analytics.engine import analyze_full_match, render_match_video

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ball_action_classifier = _build_ball_action_classifier(cfg, args.device, args.skip_ball_actions)
    event_classifier = _build_event_classifier(cfg, args.device, args.skip_events)

    spot_kwargs = _build_spot_kwargs(args, cfg)

    telemetry, tracks_path, possessor_by_frame = analyze_full_match(
        source_video=args.source,
        detector_name=args.detector,
        ball_action_classifier=ball_action_classifier,
        event_classifier=event_classifier,
        max_possessor_distance_px=cfg.analytics.max_possessor_distance_px,
        hysteresis_frames=cfg.analytics.hysteresis_frames,
        event_window_s=cfg.events.window_s,
        event_stride_s=args.event_stride_s,
        min_window_fraction=cfg.full_match.min_window_fraction,
        ball_action_temporal_offsets_frames=cfg.ball_actions.temporal_offsets_frames,
        device=args.device,
        **spot_kwargs,
    )
    logger.info(
        "Partido analizado: %d frames, %d cambios de posesión, %d tramos de evento",
        telemetry.n_frames, telemetry.n_possession_changes, len(telemetry.events),
    )

    telemetry_path = output_dir / "telemetry.json"
    with open(telemetry_path, "w") as f:
        json.dump(telemetry.to_telemetry(), f, indent=2)
    logger.info("Telemetría escrita en %s", telemetry_path)

    team_by_tracker_id = _build_team_by_tracker_id(
        args.source, tracks_path, output_dir / "teams.json", cfg, args.device, args.skip_classification
    )

    # Aquí (y no antes) porque el tracking del partido completo lo produce
    # analyze_full_match, y así el radar puede colorearse ya por equipo.
    _run_homography(args.source, tracks_path, output_dir, cfg, args)

    if not args.skip_render and cfg.full_match.render_video:
        final_path = output_dir / "final.mp4"
        render_match_video(
            source_video=args.source,
            tracks_jsonl_path=tracks_path,
            output_path=final_path,
            team_by_tracker_id=team_by_tracker_id,
            possessor_by_frame=possessor_by_frame,
            telemetry=telemetry,
            trace_length=cfg.analytics.trace_length if args.trace_length is None else args.trace_length,
            event_spans=telemetry.events,
            action_highlight_frames=cfg.analytics.action_highlight_frames,
        )
        logger.info("Vídeo final escrito en %s", final_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestador del pipeline de fútbol (clip corto o partido completo)"
    )
    parser.add_argument("--source", required=True, type=Path, help="Vídeo .mp4 de entrada")
    parser.add_argument("--output-dir", required=True, type=Path, help="Carpeta de salida (teams.json/telemetry.json/final.mp4)")
    parser.add_argument("--mode", choices=["clip", "full-match"], required=True)
    parser.add_argument("--detector", choices=["hybrid", "soccermaster", "yolo"], default="hybrid")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-classification", action="store_true", help="No asignar equipos (team_by_tracker_id=None)")
    parser.add_argument("--skip-ball-actions", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--shot-backend", choices=["tdeed", "events"], default="tdeed",
                        help="Solo --mode full-match. Quién detecta shot/goal: 'tdeed' "
                             "(core/ball_spotting.py, por defecto: AP de tiros 75-78%% frente al "
                             "~34%% de V-JEPA2; el resto de eventos sigue saliendo de "
                             "core/events.py) o 'events' (todo de core/events.py, como antes)")
    parser.add_argument("--ball-actions-trigger", choices=["tdeed", "possession"], default="tdeed",
                        help="Solo --mode full-match. Cuándo hay una acción de balón: en cada "
                             "acción detectada por T-DEED (por defecto; F1 88%% frente al 38%% del "
                             "cambio de posesión en ground_truth/) o en cada cambio de posesión, "
                             "como antes; el jugador sale siempre de la posesión")
    parser.add_argument("--ball-actions-label", choices=["classifier", "spotter", "combined"],
                        default="combined",
                        help="Con --ball-actions-trigger tdeed, quién pone la etiqueta: el R3D-18 "
                             "('classifier'), T-DEED ('spotter') o los dos ('combined')")
    parser.add_argument("--action-min-score", type=float, default=None,
                        help="Solo --mode full-match: score mínimo de T-DEED para disparar una acción "
                             "de balón (por defecto config.BallSpottingConfig.action_min_score=0.5, "
                             "elegido en la EFL; fuera de ese dominio 0.2-0.3 recupera la densidad "
                             "real de pases y conducciones, ver config.py)")
    parser.add_argument("--throw-in-min-score", type=float, default=None,
                        help="Solo --mode full-match: score mínimo de T-DEED para el saque de banda "
                             "como evento (por defecto config.BallSpottingConfig.throw_in_min_score="
                             "0.5); un valor negativo deja el throw-in de V-JEPA2, como antes")
    parser.add_argument("--skip-homography", action="store_true",
                        help="No calibrar el campo (sin tracks_2d.jsonl ni radar.mp4)")
    parser.add_argument("--homography-stride", type=int, default=None,
                        help="Calibra 1 de cada N frames (por defecto, "
                             "config.HomographyConfig.stride)")
    parser.add_argument("--radar", action="store_true",
                        help="Escribe radar.mp4 (vista cenital) además de tracks_2d.jsonl")
    parser.add_argument("--skip-render", action="store_true", help="No escribir final.mp4, solo telemetry.json")
    parser.add_argument("--trace-length", type=int, default=None,
                        help="Frames de estela por jugador en el vídeo final (0 = solo el círculo, sin estela); "
                             "por defecto config.AnalyticsConfig.trace_length")
    parser.add_argument("--publish", action="store_true",
                        help="Solo --mode clip: sube final.mp4 a YouTube Studio con "
                             "automation/molmo_publisher.py (requiere sesión de Google ya "
                             "guardada: python -m automation.molmo_publisher --login-only)")
    parser.add_argument("--publish-title", default="Test Clip Short",
                        help="Título del vídeo publicado")
    parser.add_argument("--publish-visibility", choices=["private", "unlisted", "public"],
                        default=None,
                        help="Visibilidad al publicar (por defecto, "
                             "config.PublisherConfig.default_visibility)")
    parser.add_argument("--publish-dry-run", action="store_true",
                        help="Ensayo de publicación: navega hasta el selector de ficheros "
                             "y para ahí, sin entregar el vídeo (capturas en publish_debug/)")
    parser.add_argument(
        "--event-stride-s", type=float, default=None,
        help="Solo --mode full-match: paso (s) de la ventana deslizante de eventos "
             "(por defecto, config.FullMatchConfig.event_stride_s)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.source.exists():
        parser.error(f"No existe: {args.source}")

    cfg = _apply_spot_overrides(args, config.PipelineConfig())
    if args.event_stride_s is None:
        args.event_stride_s = cfg.full_match.event_stride_s
    if args.publish_visibility is None:
        args.publish_visibility = cfg.publisher.default_visibility

    if args.publish and args.mode != "clip":
        parser.error("--publish solo está soportado con --mode clip (se publica final.mp4)")
    if args.publish and args.skip_render:
        parser.error("--publish necesita el render: quita --skip-render")

    if args.mode == "clip":
        _run_clip(args, cfg)
    else:
        _run_full_match(args, cfg)


if __name__ == "__main__":
    main()
