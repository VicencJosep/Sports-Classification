"""Orquestación: señales -> candidatos -> VLM (juicio por jugada + ranking
relativo + fin del clip) -> selección -> clips.

Cada etapa deja su resultado en ``out_dir`` (``signals.npz``,
``candidates.json``, ``vlm_cache.json``) y se reutiliza si ya existe, así
que repetir con otros pesos o sin VLM no vuelve a leer los JSONL ni a
preguntar al modelo. ``status.json`` lleva la etapa en curso (el backend lo
enseña mientras dura).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from config import HighlightsConfig

from .candidates import Candidate, build_candidates, make_goal_candidate
from .cut import cut_clip
from .scoreboard import ScoreTimeline, build_timeline
from .signals import MatchSignals, build_signals
from .vjepa import VJ_FEATURES, VJEPAWindows, apply_vjepa, goal_fusion
from .vlm import (
    PROMPT_VERSION,
    CacheOnlyClient,
    VLMClient,
    clip_end_from_kinds,
    first_celebration,
    hms,
    judge_candidate,
    rank_candidates,
    refine_end,
)

logger = logging.getLogger(__name__)

HIGHLIGHTS_JSON = "highlights.json"


@dataclass(frozen=True)
class HighlightInputs:
    video: Path
    telemetry: Path
    teams: Optional[Path] = None
    tdeed_scores: Optional[Path] = None
    possession: Optional[Path] = None
    tracks_2d: Optional[Path] = None

    @classmethod
    def from_pipeline_layout(cls, video: Path, output_dir: Path) -> "HighlightInputs":
        """Rutas tal y como las deja ``python -m main --mode full-match``:
        cachés junto al vídeo, resultados en ``output_dir``."""
        from core.ball_spotting import scores_cache_path

        video = Path(video)
        output_dir = Path(output_dir)
        possession = video.with_name(f"{video.stem}.possession.jsonl")
        return cls(
            video=video, telemetry=output_dir / "telemetry.json", teams=output_dir / "teams.json",
            tdeed_scores=scores_cache_path(video), possession=possession,
            tracks_2d=output_dir / "tracks_2d.jsonl",
        )


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _set_status(out_dir: Path, stage: str, **extra) -> None:
    _write_json(out_dir / "status.json", {"stage": stage, "updated_at": _now(), **extra})


def _cand_key(c: Candidate) -> str:
    return f"{c.id}@{round(c.anchor_s, 1)}"


# --------------------------------------------------------------------------- #
#  Gol: VLM + reglas
# --------------------------------------------------------------------------- #

def _vlm_says_goal(verdict: dict) -> bool:
    try:
        return bool(verdict.get("is_goal")) and float(verdict.get("goal_confidence", 1.0) or 0.0) >= 0.5
    except (TypeError, ValueError):
        return bool(verdict.get("is_goal"))


# Fuentes de gol que NO lo confirman: título "Gol probable" y compiten por
# el tope de clips como cualquier jugada. Confirman el marcador y el saque
# de centro visto en el campo.
PROBABLE_GOAL_SOURCES: frozenset[str] = frozenset({"fusion", "celebration+audio"})


def goal_decision(
    verdict: dict, f: dict[str, float], has_audio: bool, fusion_min: Optional[float] = None, audio_min_db: float = 6.0,
) -> tuple[bool, str]:
    """``(es gol, por qué)``. El ``is_goal`` del VLM sobre el golpeo no vale
    (1,0 en paradas repetidas por TV, 0,0 en goles claros); lo que no engaña
    es lo que pasa DESPUÉS: tras un gol hay celebración y saque de centro
    (y griterío en cámara única), tras una parada o un fallo hay córner,
    saque de puerta o sigue el juego. Medido en docs/highlights_evaluacion.md."""
    vlm_goal = _vlm_says_goal(verdict)
    # El marcador de la retransmisión manda: cambia justo en el gol y no
    # cambia en una parada, una repetición o un tiro fuera.
    on_scoreboard = f.get("score_goal")
    if on_scoreboard is not None:
        return (True, "scoreboard") if on_scoreboard >= 0.5 else (False, "scoreboard_unchanged")
    restart = _restart_seen(f) and f.get("n_actions_before_90s", 99.0) >= 4.0
    deadball_soon = f.get("next_deadball_s", 999.0) <= 45.0
    loud = has_audio and f.get("audio_sustained", 0.0) >= audio_min_db
    shot_like = vlm_goal or bool(verdict.get("is_shot")) or f.get("shot", 0.0) >= 0.08
    if restart and (loud or not has_audio) and shot_like:
        return True, "vlm+restart" if vlm_goal else "restart"
    # Sin saque de centro visto solo hay goles PROBABLES, y siempre con la
    # celebración vista por el VLM en los frames posteriores (sin ella, el
    # resto de votos marca como gol paradas repetidas por TV):
    # - cámara única: celebración + griterío (V-JEPA2 está fuera de dominio);
    # - TV (audio comprimido, inútil): celebración + votos sumados de
    #   V-JEPA2 (gol en el golpeo, saque de centro después) y del VLM.
    celebration = bool(verdict.get("_celebration_seen"))
    if celebration and not deadball_soon and shot_like:
        if loud:
            return True, "celebration+audio"
        fused = goal_fusion(verdict, f)
        if fusion_min is not None and fused is not None and fused >= fusion_min:
            return True, "fusion"
    return False, "contradicted" if vlm_goal else "none"


def audio_informative(sig: MatchSignals) -> bool:
    """¿Sirve el audio para contrastar un gol? En grabaciones de cámara única
    el griterío sube +10-17 dB sobre el fondo; en una retransmisión de TV el
    audio va comprimido y con narración continua y el pico de un gol apenas
    asoma (+2 dB en el Clásico): ahí se trata como si no hubiera audio."""
    if not sig.has_audio:
        return False
    import numpy as np

    return float(np.percentile(sig.audio_sustained, 99.5)) >= 5.0


def _goal_hint(f: dict[str, float], cfg: HighlightsConfig, audio_ok: bool) -> bool:
    """Indicio de gol que las reglas no ponderan bastante para que el
    candidato entre en los mejores: curva GOAL o saque de centro de T-DEED,
    marcador, V-JEPA2, o griterío con algo de V-JEPA2 (cámara única: ahí
    V-JEPA2 está fuera de dominio y solo da 0,2-0,5 en goles reales)."""
    return (f.get("goal", 0.0) >= 0.3 or f.get("center_restart", 0.0) >= 1.0 or f.get("score_goal", 0.0) >= 0.5
            or f.get("vj_goal", 0.0) >= 0.3
            or (audio_ok and f.get("audio_sustained", 0.0) >= cfg.goal_audio_min_db
                and f.get("vj_goal", 0.0) >= cfg.vjepa_loud_min_goal))


def vlm_candidates(cands: list[Candidate], cfg: HighlightsConfig, audio_ok: bool) -> list[Candidate]:
    """Los que ve el VLM: los mejores por reglas más la red de seguridad de
    ``_goal_hint`` (un gol puede ir muy abajo por reglas)."""
    return cands[: cfg.vlm_max_candidates] + [c for c in cands[cfg.vlm_max_candidates:] if _goal_hint(c.features, cfg, audio_ok)]


def _restart_seen(f: dict[str, float]) -> bool:
    # La geometría (balón en el círculo central, equipos en su mitad) es lo
    # fiable; la clase Kick-off de la cabeza SoccerNet solo cuenta si es nítida.
    return f.get("center_restart", 0.0) >= 1.0 or f.get("kickoff_curve", 0.0) >= 0.2


def after_play_text(f: dict[str, float]) -> str:
    """Lo que el detector vio después de la jugada, para el ranking."""
    on_scoreboard = f.get("score_goal")
    if on_scoreboard is not None:
        return ("then the scoreboard changed (a goal was scored)" if on_scoreboard >= 0.5
                else "then the scoreboard did not change (no goal)")
    if _restart_seen(f):
        return "then a kick-off from the centre spot (a goal was scored)"
    nd = f.get("next_deadball_s", 999.0)
    if nd <= 45.0:
        return f"then a dead ball (corner/throw-in/foul) {nd:.0f} s later (no goal)"
    return "then play goes on"


_GOAL_WORDS = ("gol", "goal")
_GOAL_CLAIM_RE = re.compile(r"\b(gol|golazo|goal|marca|anota|mete el bal[oó]n|scores?|scored)\b", re.IGNORECASE)
_NO_GOAL_RE = re.compile(r"\b(evit\w+|salv\w+|impid\w+|sin|no)\s+(el\s+|un\s+)?(gol|goal)\b", re.IGNORECASE)


def claims_goal(text: str) -> bool:
    """La descripción afirma que entra un gol (sin contar "salva el gol", "evita el gol")."""
    return bool(_GOAL_CLAIM_RE.search(_NO_GOAL_RE.sub("", text or "")))


def _description(is_goal: bool, judge: dict, score: Optional[dict], source: Optional[str] = None) -> str:
    """Texto del VLM coherente con la decisión final: el VLM describe "un
    gol" donde el marcador no cambia y "una parada" donde sí."""
    text = judge.get("description_es") or judge.get("what_happens") or ""
    if is_goal and score:
        return f"Gol: el marcador pasa de {score['before'][0]}-{score['before'][1]} a {score['after'][0]}-{score['after'][1]}."
    if is_goal and source in PROBABLE_GOAL_SOURCES:
        return text if claims_goal(text) else (f"Probable gol. {text}" if text else "Probable gol.")
    if is_goal and not claims_goal(text):
        return "Gol."
    if not is_goal and claims_goal(text):
        return "Ocasión de peligro sin gol."
    return text


def _score_title(score: dict) -> str:
    """"Gol del local (1-0)": quién marca y el marcador, que es lo que sabemos
    seguro (el VLM se inventa jugadores y equipos)."""
    (b0, b1), (a0, a1) = score["before"], score["after"]
    who = "local" if a0 > b0 and a1 == b1 else "visitante" if a1 > b1 and a0 == b0 else None
    return f"Gol del {who} ({a0}-{a1})" if who else f"Gol ({a0}-{a1})"


def _title(label: str, judge: dict, anchor_s: float, score: Optional[dict] = None, source: Optional[str] = None) -> str:
    """Título coherente con la decisión final: el VLM a veces titula "gol"
    una parada (y al revés), y la decisión de gol la toman VLM + reglas."""
    vlm_title = str(judge.get("title_es") or "").strip()
    says_goal = any(w in vlm_title.lower() for w in _GOAL_WORDS)
    if label == "goal":
        if score:
            return _score_title(score)
        if source in PROBABLE_GOAL_SOURCES:
            return f"Gol probable ({hms(anchor_s)})"
        return vlm_title if says_goal else f"Gol ({hms(anchor_s)})"
    if vlm_title and not says_goal:
        return vlm_title
    return {"shot": "Ocasión de gol", "play": "Jugada destacada"}.get(label, label) + f" ({hms(anchor_s)})"


# --------------------------------------------------------------------------- #
#  Marcador de la retransmisión
# --------------------------------------------------------------------------- #

SCOREBOARD_LAG_S = 4.0              # el rótulo cambia unos segundos después del gol
GOAL_REANCHOR_MIN_SHIFT_S = 10.0    # el gol se re-ancla solo si el ancla va claramente por delante


def goal_time(anchor_s: float, score: Optional[dict], kinds: list, cfg: HighlightsConfig) -> Optional[float]:
    """Instante del gol cuando el ancla (tiro de T-DEED, griterío) va claramente por delante de él, para contar
    desde ahí el tope de duración: por el cambio de marcador (fiable; ``t_hi`` − retraso del rótulo) o, sin
    rótulo, por el primer frame de celebración. En TV las repeticiones pueden ir ANTES de la celebración (el
    1-1 del Clásico: 7 frames de repetición y la celebración a +30 s), por eso con rótulo manda el rótulo y sin
    él solo cuenta un desplazamiento claro. ``None`` = el ancla ya es el gol."""
    if score is not None and score.get("t_hi") is not None:
        t = float(score["t_hi"]) - SCOREBOARD_LAG_S
        return t if t - anchor_s >= GOAL_REANCHOR_MIN_SHIFT_S else None
    t_c = first_celebration(kinds)
    if t_c is not None:
        t = t_c - cfg.vlm_end_step_s
        if GOAL_REANCHOR_MIN_SHIFT_S <= t - anchor_s <= cfg.post_max_s:
            return t
    return None


def apply_scoreboard(sig: MatchSignals, cands: list[Candidate], timeline: ScoreTimeline, cfg: HighlightsConfig) -> dict[str, dict]:
    """Cada cambio de marcador es un gol: se asigna al candidato de la
    ventana con más tiro/gol (o se crea uno si ningún detector lo vio) y se le
    pone ``score_goal=1``. Los demás candidatos con lecturas del marcador a
    ambos lados y sin cambio no son gol (``score_goal=0``); el resto queda
    sin ``score_goal`` y decide la cascada de siempre. Devuelve
    ``{candidate_id: {"before": [h, a], "after": [h, a]}}`` y reordena ``cands``."""
    info: dict[str, dict] = {}
    if not timeline.available:
        return info
    goal_ids: set[str] = set()
    for k, ch in enumerate(timeline.changes):
        pool = [c for c in cands if ch.t_lo - cfg.scoreboard_goal_before_s <= c.anchor_s <= ch.t_hi + 3.0
                and c.id not in goal_ids]
        # El rótulo cambia pocos segundos después del gol: si hay candidatos en ese margen corto, el gol es
        # uno de ellos aunque T-DEED puntúe más alto un tiro anterior (Mallorca 1-0: el tiro 27 s antes del
        # gol ganaba por SHOT 0,35 frente a 0,14 y el clip se quedaba sin celebración).
        near = [c for c in pool if c.anchor_s >= ch.t_hi - cfg.scoreboard_goal_near_s]
        if near:
            pool = near
        if pool:
            c = max(pool, key=lambda c: (c.features.get("shot", 0.0) + c.features.get("goal", 0.0),
                                         -abs(c.anchor_s - (ch.t_hi - SCOREBOARD_LAG_S))))
        else:
            c = make_goal_candidate(sig, max(ch.t_hi - SCOREBOARD_LAG_S, 0.0), f"cand-sb{k:02d}", cfg)
            cands.append(c)
        goal_ids.add(c.id)
        info[c.id] = {"before": list(ch.before), "after": list(ch.after), "t_lo": ch.t_lo, "t_hi": ch.t_hi}
    for c in cands:
        if c.id in goal_ids:
            c.features["score_goal"] = 1.0
            c.rule_score += cfg.w_scoreboard
            c.sources = sorted(set(c.sources) | {"scoreboard"})
        elif timeline.brackets(c.anchor_s):
            c.features["score_goal"] = 0.0
    cands.sort(key=lambda c: -c.rule_score)
    return info


# --------------------------------------------------------------------------- #
#  Puntuación y selección
# --------------------------------------------------------------------------- #

def vlm_score(verdict: dict, is_goal: bool = False) -> Optional[float]:
    """0-1 a partir del juicio individual del VLM; None si no contestó nada útil."""
    if not verdict or ("excitement" not in verdict and "is_goal" not in verdict):
        return None
    try:
        excitement = float(verdict.get("excitement", 5))
        skill = float(verdict.get("skill", 5))
    except (TypeError, ValueError):
        excitement, skill = 5.0, 5.0
    score = (0.6 * excitement + 0.4 * skill) / 10.0
    if is_goal:
        score = max(score, 0.9)
    if verdict.get("worth_highlight") is False:
        score *= 0.5
    return min(max(score, 0.0), 1.0)


def select_highlights(
    cands: list[Candidate], judges: dict[str, dict], ranking: Optional[dict], cfg: HighlightsConfig, has_audio: bool,
) -> list[dict]:
    """Orden de prioridad: goles, luego el ranking del VLM, luego puntuación
    combinada. Entre ``min_clips`` y ``max_clips``; por encima de
    ``min_clips`` solo entran los que superan ``select_min_score``."""
    if not cands:
        return []
    max_rule = max(c.rule_score for c in cands) or 1.0
    rank = (ranking or {}).get("rank", {})
    rejected = set((ranking or {}).get("rejected", []))
    rows = []
    for c in cands:
        v = judges.get(c.id, {})
        is_goal, goal_source = goal_decision(v, c.features, has_audio, cfg.goal_fusion_min, cfg.goal_audio_min_db)
        rule_norm = c.rule_score / max_rule
        vs = vlm_score(v, is_goal)
        if vs is not None:
            final = cfg.w_final_rules * rule_norm + cfg.w_final_vlm * vs
        elif judges:
            final = 0.3 * rule_norm      # con VLM en marcha, lo que no ha visto solo rellena
        else:
            final = rule_norm
        if c.id in rank:
            final = max(final, 1.0 - 0.08 * (rank[c.id] - 1))
        elif c.id in rejected and not is_goal:
            # El ranking (solo texto) rechaza jugadas por su propio resumen,
            # que en los goles es lo menos fiable: no veta a un gol.
            final = min(final, 0.3 * rule_norm)
        rows.append({"candidate": c, "final_score": final, "is_goal": is_goal, "goal_source": goal_source,
                     "rank": rank.get(c.id), "reason": (ranking or {}).get("reasons", {}).get(c.id)})
    rows.sort(key=lambda r: (not r["is_goal"], r["rank"] if r["rank"] is not None else 10 ** 6, -r["final_score"]))

    chosen: list[dict] = []
    for r in rows:
        # Los goles confirmados (marcador, saque de centro visto) entran todos;
        # el tope limita el resto, también a los "probables" (fusión).
        confirmed = r["is_goal"] and r["goal_source"] not in PROBABLE_GOAL_SOURCES
        if len(chosen) >= cfg.max_clips and not confirmed:
            continue
        if not confirmed and len(chosen) >= cfg.min_clips and r["final_score"] < cfg.select_min_score:
            continue
        if not r["is_goal"] and r["candidate"].id in rejected and len(chosen) >= cfg.min_clips:
            continue
        chosen.append(r)
    return chosen


def _dedupe(items: list[dict]) -> list[dict]:
    """``items`` en orden de prioridad, ya con ``start_s``/``end_s`` finales:
    fuera los que solapan con uno más prioritario o cuyo momento clave cae
    dentro de él (p.ej. un "tiro" en plena celebración de un gol)."""
    kept: list[dict] = []
    for it in items:
        c = it["candidate"]
        clash = False
        for k in kept:
            if k["start_s"] <= c.anchor_s <= k["end_s"] or it["start_s"] <= k["candidate"].anchor_s <= it["end_s"]:
                clash = True
            elif min(it["end_s"], k["end_s"]) - max(it["start_s"], k["start_s"]) > 3.0:
                clash = True
            if clash:
                break
        if not clash:
            kept.append(it)
    return kept


def fit_duration(
    start_s: float, end_s: float, anchor_s: float, is_goal: bool, cfg: HighlightsConfig, duration_s: float,
    goal_s: Optional[float] = None,
) -> tuple[float, float]:
    """Política de duración (hoja de ruta §3.4) sobre unos límites ya decididos: 12-30 s para una jugada,
    hasta 45 s para un gol. El momento clave es el gol real (``goal_s``, primer frame de celebración − un paso)
    si se conoce, que en cámara única puede ir 20-30 s después del ancla; si no, el ancla.

    1. Lo anterior al momento clave se recorta a ``clip_back_max_s`` (15 s en gol): es lo que menos se echa de
       menos (en TV la jugada casi nunca se forma; en un gol importa la celebración).
    2. El clip no pasa de ``clip_max_s`` / ``clip_max_goal_s``: se recorta la cola.
    3. Si queda por debajo de ``clip_min_s`` se alarga hacia delante y, si no hay vídeo, hacia atrás."""
    max_s = cfg.clip_max_goal_s if is_goal else cfg.clip_max_s
    back_max = cfg.clip_back_max_goal_s if is_goal else cfg.clip_back_max_s
    key = goal_s if (is_goal and goal_s is not None) else anchor_s
    start = max(0.0, start_s, key - back_max)
    end = min(duration_s, max(end_s, key + cfg.post_min_s))
    if end - start > max_s:
        end = start + max_s
    if end - start < cfg.clip_min_s:
        end = min(duration_s, start + cfg.clip_min_s)
        if end - start < cfg.clip_min_s:
            start = max(0.0, end - cfg.clip_min_s)
    return start, end


def _merge_close(items: list[dict], gap_s: float, max_merged_s: Optional[float] = None) -> list[dict]:
    """``items`` ya ordenados por inicio: dos jugadas casi seguidas (parada y
    córner con gol, dos remates seguidos) van en un mismo clip, salvo que el
    resultado pase de ``max_merged_s``."""
    merged: list[dict] = []
    for it in items:
        if merged and it["start_s"] - merged[-1]["end_s"] <= gap_s and (
                max_merged_s is None or max(merged[-1]["end_s"], it["end_s"]) - merged[-1]["start_s"] <= max_merged_s):
            prev = merged[-1]
            lead = it if (it["is_goal"] and not prev["is_goal"]) or (
                it["is_goal"] == prev["is_goal"] and it["final_score"] > prev["final_score"]) else prev
            lead = dict(lead)
            lead["start_s"], lead["end_s"] = prev["start_s"], max(prev["end_s"], it["end_s"])
            lead["is_goal"] = prev["is_goal"] or it["is_goal"]
            lead["final_score"] = max(prev["final_score"], it["final_score"])
            lead["merged_from"] = prev.get("merged_from", [prev["candidate"].id]) + [it["candidate"].id]
            merged[-1] = lead
        else:
            merged.append(it)
    return merged


# --------------------------------------------------------------------------- #
#  Generación
# --------------------------------------------------------------------------- #

def generate_highlights(
    inputs: HighlightInputs,
    out_dir: Path,
    cfg: HighlightsConfig = HighlightsConfig(),
    use_vlm: Optional[bool] = None,
    use_vjepa: Optional[bool] = None,
    cut: bool = True,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_vlm = cfg.vlm_enabled if use_vlm is None else use_vlm
    use_vjepa = cfg.vjepa_enabled if use_vjepa is None else use_vjepa

    def stage(name: str, **extra) -> None:
        _set_status(out_dir, name, **extra)
        if progress:
            progress(name)
        logger.info("[highlights] %s %s", name, extra or "")

    # 1. Señales
    signals_path = out_dir / "signals.npz"
    if signals_path.exists() and not force:
        sig = MatchSignals.load(signals_path)
    else:
        stage("signals")
        sig = build_signals(inputs.video, inputs.telemetry, inputs.teams, inputs.tdeed_scores,
                            inputs.possession, inputs.tracks_2d)
        sig.save(signals_path)

    # 2. Candidatos (siempre se recalculan: es barato y depende de los pesos)
    stage("candidates")
    cands = build_candidates(sig, cfg)
    _write_json(out_dir / "candidates.json", [c.to_dict() for c in cands])
    logger.info("%d candidatos; top-5 por reglas: %s", len(cands),
                [(c.id, hms(c.anchor_s), round(c.rule_score, 2)) for c in cands[:5]])

    # 3. VLM: juicio individual de los mejores por reglas (+ goles claros)
    cache_path = out_dir / "vlm_cache.json"
    cache: dict[str, dict] = {}
    if cache_path.exists() and not force:
        with open(cache_path) as f:
            cache = json.load(f)
    client: Optional[VLMClient] = None
    vlm_state = {"failed": 0}       # llamadas al VLM que fallaron (modelo caído a mitad, o modo solo caché)
    if use_vlm and cands:
        client = (CacheOnlyClient(cfg.vlm_model) if cfg.vlm_api == "cache"
                  else VLMClient(cfg.vlm_base_url, cfg.vlm_model, cfg.vlm_api, cfg.vlm_api_key, cfg.vlm_timeout_s))
        if not client.ping():
            logger.warning("VLM no disponible en %s: selección solo por reglas", cfg.vlm_base_url)
            use_vlm, client = False, None

    def ask(what: str, fn, *fn_args):
        """Una llamada al VLM; si falla se sigue sin ese veredicto (no se cachea el fallo)."""
        try:
            return fn(*fn_args)
        except RuntimeError as exc:
            vlm_state["failed"] += 1
            logger.warning("VLM sin respuesta (%s): %s", what, exc)
            return None

    # 2b. Marcador de la retransmisión: cada cambio es un gol confirmado
    timeline: Optional[ScoreTimeline] = None
    score_info: dict[str, dict] = {}
    if client is not None and cfg.scoreboard_enabled:
        timeline = build_timeline(
            client, inputs.video, sig.duration_s, cfg.scoreboard_step_s, cfg.scoreboard_min_valid,
            out_dir / "scoreboard.json", cfg.vlm_model,
            progress=lambda done, total: stage("scoreboard", done=done, total=total) if done % 10 == 0 else None)
        score_info = apply_scoreboard(sig, cands, timeline, cfg)
        _write_json(out_dir / "candidates.json", [c.to_dict() for c in cands])
        logger.info("Marcador: %s; goles confirmados: %s", "legible" if timeline.available else "no legible",
                    [(hms((ch.t_lo + ch.t_hi) / 2), ch.before, ch.after) for ch in timeline.changes])

    # 2c. V-JEPA2: segundo juez de gol (golpeo, saque de centro después) sin
    # marcador; también reordena para que el VLM vea los goles probables.
    vjepa_used = False
    if use_vjepa and cands:
        try:
            vw = VJEPAWindows(inputs.video, sig.duration_s, out_dir / "vjepa_cache.json", cfg.vjepa_window_s,
                              cfg.vjepa_classifier_path, cfg.vjepa_model_id)
            n_new = apply_vjepa(vw, cands, cfg, progress=lambda done, total: stage("vjepa", done=done, total=total))
            vjepa_used = True
            _write_json(out_dir / "candidates.json", [c.to_dict() for c in cands])
            logger.info("V-JEPA2: %d ventanas nuevas; vj_goal >= 0.3 en %s", n_new,
                        [(c.id, hms(c.anchor_s), round(c.features["vj_goal"], 2)) for c in cands if c.features.get("vj_goal", 0.0) >= 0.3])
        except Exception:                      # modelo o GPU no disponibles: se sigue sin él
            logger.exception("V-JEPA2 no disponible: selección sin su voto")

    def fresh(entry: dict) -> bool:
        return entry.get("model") == cfg.vlm_model and entry.get("version") == PROMPT_VERSION

    judged: list[Candidate] = []
    judges: dict[str, dict] = {}
    ranking: Optional[dict] = None
    if client is not None:
        to_judge = vlm_candidates(cands, cfg, audio_informative(sig))
        for k, c in enumerate(to_judge):
            key = _cand_key(c)
            entry = cache.get(key, {})
            if "judge" not in entry or not fresh(entry):
                stage("vlm", done=k, total=len(to_judge))
                verdict = ask(f"juicio {c.id}", judge_candidate, client, inputs.video, c.start_s, c.anchor_s,
                              cfg.vlm_frames_judge, cfg.vlm_image_width)
                if verdict is None:
                    continue
                entry = {"model": cfg.vlm_model, "version": PROMPT_VERSION, "judge": verdict}
                cache[key] = entry
                _write_json(cache_path, cache)
                logger.info("VLM %s %s: %s (%.0f s)", c.id, hms(c.anchor_s),
                            {k2: entry["judge"].get(k2) for k2 in ("is_goal", "excitement", "skill", "worth_highlight")},
                            entry["judge"].get("_latency_s", 0.0))
            judges[c.id] = entry["judge"]
            judged.append(c)

        # 3b. Ranking relativo (solo texto, una llamada)
        if len(judged) >= 2:
            rank_key = "_ranking:" + ",".join(_cand_key(c) for c in judged)
            if rank_key not in cache or not fresh(cache[rank_key]):
                stage("vlm_rank")
                items = []
                for c in judged:
                    j = judges[c.id]
                    items.append({
                        "id": c.id, "t_s": c.anchor_s, "what_happens": j.get("what_happens"),
                        "is_goal": j.get("is_goal"), "goal_confidence": j.get("goal_confidence"),
                        "shot": c.features.get("shot", 0.0), "goal": c.features.get("goal", 0.0),
                        "audio": c.features.get("audio_sustained", 0.0), "passes": c.features.get("n_passes", 0.0),
                        "carry_m": c.features.get("drive_m", 0.0), "after": after_play_text(c.features),
                    })
                ranked = ask("ranking", rank_candidates, client, items)
                if ranked is not None:
                    cache[rank_key] = {"model": cfg.vlm_model, "version": PROMPT_VERSION, **ranked}
                    _write_json(cache_path, cache)
            ranking = cache.get(rank_key)
            if ranking is not None:
                logger.info("Ranking VLM: %s; rechazados: %s", sorted(ranking["rank"].items(), key=lambda kv: kv[1]), ranking["rejected"])

    def ensure_end(c: Candidate) -> dict:
        """Clasificación de los frames posteriores al momento clave (cacheada)."""
        key = _cand_key(c)
        entry = cache.setdefault(key, {"model": cfg.vlm_model, "version": PROMPT_VERSION})
        if not entry.get("end", {}).get("kinds") or not fresh(entry):
            stage("vlm_end")
            end = ask(f"fin {c.id}", refine_end, client, inputs.video, c.anchor_s, cfg.post_max_s, cfg.vlm_end_step_s,
                      cfg.vlm_frames_end, cfg.vlm_image_width, cfg.post_default_s)
            if end is None:
                return {}
            entry["end"] = end
            entry["model"], entry["version"] = cfg.vlm_model, PROMPT_VERSION
            _write_json(cache_path, cache)
        return entry["end"]

    # 3c. Lo que pasa después (celebración o no) se mira en todos los
    # candidatos juzgados ANTES de seleccionar: es el voto que decide los
    # goles probables (y los frames posteriores sirven luego para el corte).
    if client is not None:
        for c in judged:
            judges[c.id]["_celebration_seen"] = bool(ensure_end(c).get("celebration_seen"))

    # 4. Selección (con margen: el filtrado de solapes viene después)
    stage("select")
    chosen = select_highlights(cands, judges, ranking, cfg, audio_informative(sig))

    # 5. Límites finales (inicio del juicio, fin del VLM), duración, solapes y fusiones
    for item in chosen:
        c: Candidate = item["candidate"]
        judge = judges.get(c.id, {})
        start_s = c.start_s
        if judge.get("play_start_s") is not None:
            start_s = min(c.start_s + 10.0, max(c.start_s - 10.0, judge["play_start_s"] - 1.0))
        end_s = c.end_s
        goal_s: Optional[float] = None
        if client is not None:
            end_info = ensure_end(c)
            kinds = end_info.get("kinds") or []
            # Las repeticiones de TV solo alargan un gol; tras una parada el clip acaba con la jugada.
            vlm_end = clip_end_from_kinds(kinds, c.anchor_s, cfg.vlm_end_step_s, cfg.post_max_s, cfg.post_default_s,
                                          replay_as_reaction=item["is_goal"], replay_tail_max_s=cfg.end_replay_max_s)
            if vlm_end is not None:
                end_s = min(max(vlm_end, c.anchor_s + cfg.post_min_s), c.anchor_s + cfg.post_max_s)
            if item["is_goal"]:
                goal_s = goal_time(c.anchor_s, score_info.get(c.id), kinds, cfg)
            item["end_notes"] = end_info.get("notes")
            item["end_kinds"] = kinds
        start_s, end_s = max(0.0, start_s), min(end_s, sig.duration_s)
        item["before_fit"] = {"start_s": round(start_s, 2), "end_s": round(end_s, 2)}
        item["goal_s"] = goal_s
        item["start_s"], item["end_s"] = fit_duration(start_s, end_s, c.anchor_s, item["is_goal"], cfg,
                                                      sig.duration_s, goal_s)
    chosen = _dedupe(chosen)
    chosen.sort(key=lambda it: it["start_s"])
    chosen = _merge_close(chosen, cfg.merge_gap_s, cfg.clip_max_merged_s)

    # 6. Corte
    results = []
    for k, item in enumerate(chosen, start=1):
        c = item["candidate"]
        judge = judges.get(c.id, {})
        clip_name = f"clip_{k:02d}.mp4"
        if cut:
            stage("cut", done=k - 1, total=len(chosen))
            cut_clip(inputs.video, item["start_s"], item["end_s"], out_dir / clip_name, cfg.ffmpeg_crf, cfg.ffmpeg_preset)
        score = next((score_info[i] for i in [c.id, *(item.get("merged_from") or [])] if i in score_info), None)
        label = "goal" if item["is_goal"] else ("shot" if c.features.get("shot", 0) >= 0.08 or judge.get("is_shot") else "play")
        results.append({
            "id": f"hl-{k:02d}", "rank": k, "clip": clip_name if cut else None,
            "start_s": round(item["start_s"], 2), "end_s": round(item["end_s"], 2), "anchor_s": round(c.anchor_s, 2),
            "duration_s": round(item["end_s"] - item["start_s"], 2), "label": label, "team": c.team,
            "title": _title(label, judge, c.anchor_s, score, item.get("goal_source")),
            "description": _description(item["is_goal"], judge, score, item.get("goal_source")),
            "final_score": round(item["final_score"], 3), "rule_score": round(c.rule_score, 3),
            "goal_source": item.get("goal_source"), "vlm_rank": item.get("rank"), "vlm_reason": item.get("reason"),
            "vlm": {k2: judge.get(k2) for k2 in ("is_shot", "is_goal", "goal_confidence", "skill", "excitement", "worth_highlight")},
            "score": score, "goal_fusion": (round(gf, 3) if (gf := goal_fusion(judge, c.features)) is not None else None),
            "vjepa": {k2: round(c.features[k2], 3) for k2 in VJ_FEATURES if k2 in c.features} or None,
            "candidate_id": c.id, "merged_from": item.get("merged_from"), "sources": c.sources,
            "goal_s": (round(item["goal_s"], 2) if item.get("goal_s") is not None else None),
            "before_fit": item.get("before_fit"),
        })
    # Los clips sobrantes de una generación anterior no deben quedar colgando.
    for stale in out_dir.glob("clip_*.mp4"):
        if stale.name not in {r["clip"] for r in results if r["clip"]}:
            stale.unlink()

    report = {
        "video": str(inputs.video), "generated_at": _now(),
        "vlm": {"used": client is not None, "model": cfg.vlm_model if client is not None else None,
                "api": cfg.vlm_api if client is not None else None,
                "degraded": client is not None and vlm_state["failed"] > 0, "n_failed_calls": vlm_state["failed"]},
        "has_audio": sig.has_audio, "audio_informative": audio_informative(sig), "n_candidates": len(cands),
        "scoreboard": timeline.to_dict() if timeline is not None else None,
        "vjepa": {"used": vjepa_used},
        "highlights": results,
        "candidates": [c.to_dict() | {"vlm_score": vlm_score(judges.get(c.id, {})),
                                      "vlm_rank": (ranking or {}).get("rank", {}).get(c.id)} for c in cands],
    }
    _write_json(out_dir / HIGHLIGHTS_JSON, report)
    stage("completed", n_highlights=len(results))
    return report
