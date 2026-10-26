"""Rejilla temporal (bins de 0,5 s) con todas las señales de un partido ya
analizado, para que la búsqueda de jugadas no vuelva a tocar el vídeo:

- curvas de T-DEED de las dos cabezas (``SpotScores``), sin umbral: un gol
  que T-DEED puntúa a 0,1 no sale en ``telemetry.json`` pero aquí sí;
- excitación del audio: RMS en dB sobre la mediana móvil de 5 min y su
  media sostenida 15 s hacia delante (una celebración dura 20-30 s, un
  grito o un silbato no);
- poseedor y equipo por bin (``possession.jsonl`` + ``teams.json``) y
  posiciones en metros (``tracks_2d.jsonl``): balón, poseedor y reparto de
  jugadores por mitades (para ver un saque de centro tras un gol).

Todo se guarda en un ``.npz`` (``MatchSignals.save``) porque leer los JSONL
de tracking de un partido entero tarda minutos.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

GRID_S: float = 0.5
AUDIO_SR: int = 8000
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0
CENTER_XY: tuple[float, float] = (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2)
GOAL_CENTERS: tuple[tuple[float, float], ...] = ((0.0, PITCH_WIDTH_M / 2), (PITCH_LENGTH_M, PITCH_WIDTH_M / 2))

# Curvas de T-DEED que se conservan (máximo por bin). Clave propia -> clases
# de cualquiera de las dos cabezas (se suman, ver SpotScores.curve).
TDEED_CURVES: dict[str, tuple[str, ...]] = {
    "shot": ("SHOT",),
    "goal": ("GOAL",),
    "cross": ("CROSS",),
    "out": ("OUT",),
    "pass": ("PASS",),
    "drive": ("DRIVE",),
    "tackle": ("PLAYER SUCCESSFUL TACKLE", "BALL PLAYER BLOCK"),
    "sn_goal": ("Goal",),
    "sn_shot": ("Shots on target", "Shots off target"),
    "sn_kickoff": ("Kick-off",),
    "sn_corner": ("Corner",),
    "sn_foul": ("Foul",),
}

# Eventos de telemetry.json que son balón parado (cortan una jugada).
DEAD_BALL_EVENTS: frozenset[str] = frozenset({"throw-in", "corner", "foul", "free-kick", "kick-off", "penalty"})


@dataclass
class MatchSignals:
    duration_s: float
    fps: float
    t: np.ndarray                      # centro de cada bin (s)
    has_audio: bool
    audio_rel: np.ndarray              # dB sobre la mediana móvil (0 si no hay audio)
    audio_sustained: np.ndarray        # media de audio_rel en los 15 s siguientes
    curves: dict[str, np.ndarray]      # TDEED_CURVES -> máximo por bin (0 si no hay caché)
    possessor: np.ndarray              # tracker_id mayoritario del bin (-1 sin poseedor)
    team: np.ndarray                   # equipo del poseedor (-1 desconocido)
    ball_xy: np.ndarray                # [n, 2] metros (NaN sin homografía)
    possessor_xy: np.ndarray           # [n, 2] metros (NaN)
    players_left_frac: np.ndarray      # fracción de jugadores proyectados en la mitad izquierda (NaN)
    n_players_xy: np.ndarray           # jugadores con field_xy en el bin (media)
    actions: list[dict] = field(default_factory=list)   # telemetry.ball_actions + t_s + team
    events: list[dict] = field(default_factory=list)    # telemetry.events -> {label, t_s, score}

    @property
    def n_bins(self) -> int:
        return len(self.t)

    def bin_of(self, t_s: float) -> int:
        return int(min(max(t_s, 0.0) // GRID_S, self.n_bins - 1))

    def window(self, t0: float, t1: float) -> slice:
        """Bins cuyo centro cae en [t0, t1] (al menos uno)."""
        a = self.bin_of(t0)
        b = max(self.bin_of(t1), a) + 1
        return slice(a, b)

    def curve(self, name: str) -> np.ndarray:
        return self.curves.get(name, np.zeros(self.n_bins, dtype=np.float32))

    # --- persistencia -------------------------------------------------- #

    def save(self, path: Path) -> None:
        arrays = {f"curve__{k}": v for k, v in self.curves.items()}
        np.savez_compressed(
            path, duration_s=self.duration_s, fps=self.fps, t=self.t, has_audio=self.has_audio,
            audio_rel=self.audio_rel, audio_sustained=self.audio_sustained, possessor=self.possessor,
            team=self.team, ball_xy=self.ball_xy, possessor_xy=self.possessor_xy,
            players_left_frac=self.players_left_frac, n_players_xy=self.n_players_xy,
            actions=json.dumps(self.actions), events=json.dumps(self.events), **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> "MatchSignals":
        with np.load(path, allow_pickle=False) as z:
            curves = {k[len("curve__"):]: z[k] for k in z.files if k.startswith("curve__")}
            return cls(
                duration_s=float(z["duration_s"]), fps=float(z["fps"]), t=z["t"], has_audio=bool(z["has_audio"]),
                audio_rel=z["audio_rel"], audio_sustained=z["audio_sustained"], curves=curves,
                possessor=z["possessor"], team=z["team"], ball_xy=z["ball_xy"], possessor_xy=z["possessor_xy"],
                players_left_frac=z["players_left_frac"], n_players_xy=z["n_players_xy"],
                actions=json.loads(str(z["actions"])), events=json.loads(str(z["events"])),
            )


# --------------------------------------------------------------------------- #
#  Audio
# --------------------------------------------------------------------------- #

def _pcm_mono(video: Path, sr: int = AUDIO_SR) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        logger.warning("ffmpeg no pudo extraer audio de %s: %s", video, proc.stderr.decode(errors="replace")[-300:])
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _rolling_median(x: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    return median_filter(x, size=size, mode="nearest")


def audio_excitement(
    video: Path, n_bins: int, baseline_s: float = 300.0, sustain_s: float = 15.0,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """``(has_audio, rel_db, sustained_db)`` en la rejilla. Sin pista de audio
    (o pista muda) devuelve ceros y ``has_audio=False``."""
    x = _pcm_mono(video)
    hop = int(AUDIO_SR * GRID_S)
    n = len(x) // hop
    if n < 10:
        return False, np.zeros(n_bins, np.float32), np.zeros(n_bins, np.float32)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-10)
    db = 20.0 * np.log10(rms)
    if np.percentile(db, 99) < -80.0:       # pista presente pero muda (p.ej. ground_truth/)
        return False, np.zeros(n_bins, np.float32), np.zeros(n_bins, np.float32)
    rel = db - _rolling_median(db, max(3, int(baseline_s / GRID_S)))
    k = max(1, int(sustain_s / GRID_S))
    # media hacia delante: sustained[i] = mean(rel[i:i+k])
    padded = np.concatenate([rel, np.full(k - 1, rel[-1])])
    sustained = np.convolve(padded, np.ones(k) / k, mode="valid")
    out_rel = np.zeros(n_bins, np.float32)
    out_sus = np.zeros(n_bins, np.float32)
    m = min(n_bins, n)
    out_rel[:m] = rel[:m]
    out_sus[:m] = sustained[:m]
    return True, out_rel, out_sus


# --------------------------------------------------------------------------- #
#  T-DEED
# --------------------------------------------------------------------------- #

def tdeed_curves(npz_path: Path, n_bins: int, fps: float) -> dict[str, np.ndarray]:
    from core.ball_spotting import read_scores

    scores = read_scores(npz_path, fps=fps)
    t_rows = scores.frame_indices() / scores.fps
    idx = np.minimum((t_rows // GRID_S).astype(int), n_bins - 1)
    out: dict[str, np.ndarray] = {}
    for name, sources in TDEED_CURVES.items():
        try:
            curve = scores.curve(sources)
        except KeyError:
            logger.warning("La caché de T-DEED no tiene %s: curva '%s' a cero", sources, name)
            curve = np.zeros(len(t_rows), np.float32)
        binned = np.zeros(n_bins, np.float32)
        np.maximum.at(binned, idx, curve.astype(np.float32))
        out[name] = binned
    return out


# --------------------------------------------------------------------------- #
#  Posesión y campo
# --------------------------------------------------------------------------- #

def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_team_by_tracker(teams_json: Optional[Path]) -> dict[int, int]:
    if teams_json is None or not Path(teams_json).exists():
        return {}
    with open(teams_json) as f:
        raw = json.load(f)
    return {int(t["tracker_id"]): int(t["team_cluster"]) for t in raw.get("teams", [])}


def possession_and_field(
    possession_jsonl: Optional[Path],
    tracks_2d_jsonl: Optional[Path],
    team_by_tracker: dict[int, int],
    n_bins: int,
    fps: float,
) -> dict[str, np.ndarray]:
    """Agrega por bin: poseedor mayoritario, su equipo, posición media del
    balón y del poseedor, reparto de jugadores por mitades. Los dos JSONL
    vienen de ``tracks.jsonl`` en el mismo orden; ``tracks_2d`` es opcional
    (sin homografía todo lo métrico queda NaN)."""
    possessor = np.full(n_bins, -1, dtype=np.int64)
    team = np.full(n_bins, -1, dtype=np.int64)
    ball_xy = np.full((n_bins, 2), np.nan, dtype=np.float32)
    poss_xy = np.full((n_bins, 2), np.nan, dtype=np.float32)
    left_frac = np.full(n_bins, np.nan, dtype=np.float32)
    n_players = np.zeros(n_bins, dtype=np.float32)
    if possession_jsonl is None or not Path(possession_jsonl).exists():
        logger.warning("Sin possession.jsonl: poseedor/equipo desconocidos")
        return dict(possessor=possessor, team=team, ball_xy=ball_xy, possessor_xy=poss_xy,
                    players_left_frac=left_frac, n_players_xy=n_players)

    # Acumuladores por bin
    votes: dict[int, dict[int, int]] = {}
    ball_acc = np.zeros((n_bins, 3), np.float64)
    poss_acc = np.zeros((n_bins, 3), np.float64)
    left_acc = np.zeros((n_bins, 2), np.float64)   # [suma de fracciones, nº frames con >=4 jugadores]
    nply_acc = np.zeros((n_bins, 2), np.float64)

    tracks_iter = _iter_jsonl(tracks_2d_jsonl) if tracks_2d_jsonl and Path(tracks_2d_jsonl).exists() else None
    n_frames = 0
    for rec in _iter_jsonl(possession_jsonl):
        n_frames += 1
        frame_idx = int(rec["frame_idx"])
        b = min(int((frame_idx / fps) // GRID_S), n_bins - 1)
        pid = rec.get("possessor_tracker_id")
        if pid is not None:
            votes.setdefault(b, {})
            votes[b][int(pid)] = votes[b].get(int(pid), 0) + 1

        if tracks_iter is None:
            continue
        rec2d = next(tracks_iter, None)
        while rec2d is not None and int(rec2d["frame_idx"]) < frame_idx:
            rec2d = next(tracks_iter, None)
        if rec2d is None or int(rec2d["frame_idx"]) != frame_idx or not rec2d.get("homography_ok"):
            continue
        ball = rec2d.get("ball")
        if ball and "field_xy" in ball:
            ball_acc[b, :2] += ball["field_xy"]
            ball_acc[b, 2] += 1
        xs = []
        for tr in rec2d.get("tracks", []):
            if "field_xy" not in tr or tr.get("role") not in ("player", "goalkeeper"):
                continue
            xs.append(tr["field_xy"][0])
            if pid is not None and tr["tracker_id"] == pid:
                poss_acc[b, :2] += tr["field_xy"]
                poss_acc[b, 2] += 1
        nply_acc[b, 0] += len(xs)
        nply_acc[b, 1] += 1
        if len(xs) >= 4:
            left_acc[b, 0] += float(np.mean(np.array(xs) < PITCH_LENGTH_M / 2))
            left_acc[b, 1] += 1

    for b, v in votes.items():
        pid = max(v, key=v.get)
        possessor[b] = pid
        team[b] = team_by_tracker.get(pid, -1)
    ok = ball_acc[:, 2] > 0
    ball_xy[ok] = ball_acc[ok, :2] / ball_acc[ok, 2:3]
    ok = poss_acc[:, 2] > 0
    poss_xy[ok] = poss_acc[ok, :2] / poss_acc[ok, 2:3]
    ok = left_acc[:, 1] > 0
    left_frac[ok] = left_acc[ok, 0] / left_acc[ok, 1]
    ok = nply_acc[:, 1] > 0
    n_players[ok] = nply_acc[ok, 0] / nply_acc[ok, 1]
    logger.info("Posesión agregada: %d frames, %d bins con poseedor, %d con balón en metros",
                n_frames, int((possessor >= 0).sum()), int(np.isfinite(ball_xy[:, 0]).sum()))
    return dict(possessor=possessor, team=team, ball_xy=ball_xy, possessor_xy=poss_xy,
                players_left_frac=left_frac, n_players_xy=n_players)


# --------------------------------------------------------------------------- #
#  Telemetría
# --------------------------------------------------------------------------- #

def actions_from_telemetry(telemetry: dict, fps: float, team_by_tracker: dict[int, int]) -> list[dict]:
    out = []
    for a in telemetry.get("ball_actions", []):
        if a.get("is_background"):
            continue
        tid = int(a.get("tracker_id", -1))
        out.append({
            "t_s": a["frame_idx"] / fps, "frame_idx": int(a["frame_idx"]), "tracker_id": tid,
            "team": team_by_tracker.get(tid, -1), "label": a["label"],
            "confidence": float(a.get("confidence", 0.0)), "spot_score": a.get("spot_score"),
        })
    out.sort(key=lambda a: a["t_s"])
    return out


def events_from_telemetry(telemetry: dict, half_span_s: float = 1.0) -> list[dict]:
    """``EventSpan`` -> instante: el pico está en ``start_s + half_span``
    salvo recorte en 0 (ver analytics.engine.SPOT_EVENT_HALF_SPAN_S)."""
    out = []
    for e in telemetry.get("events", []):
        start, end = float(e["start_s"]), float(e["end_s"])
        t_s = start + half_span_s if end - start >= 2 * half_span_s - 1e-6 else (start + end) / 2
        out.append({"label": e["label"], "t_s": t_s, "score": float(e.get("peak_confidence", 0.0))})
    out.sort(key=lambda e: e["t_s"])
    return out


# --------------------------------------------------------------------------- #
#  Construcción
# --------------------------------------------------------------------------- #

def video_duration_and_fps(video: Path) -> tuple[float, float]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n / fps, fps


def build_signals(
    video: Path,
    telemetry_json: Path,
    teams_json: Optional[Path],
    tdeed_npz: Optional[Path],
    possession_jsonl: Optional[Path],
    tracks_2d_jsonl: Optional[Path],
) -> MatchSignals:
    duration_s, fps = video_duration_and_fps(video)
    with open(telemetry_json) as f:
        telemetry = json.load(f)
    n_frames = int(telemetry.get("n_frames") or 0)
    if n_frames:
        duration_s = max(duration_s, n_frames / fps)
    n_bins = int(math.ceil(duration_s / GRID_S))
    t = (np.arange(n_bins) + 0.5) * GRID_S

    team_by_tracker = load_team_by_tracker(teams_json)
    has_audio, rel, sustained = audio_excitement(video, n_bins)
    logger.info("Audio: %s", "ok" if has_audio else "ausente o mudo")
    curves = tdeed_curves(tdeed_npz, n_bins, fps) if tdeed_npz and Path(tdeed_npz).exists() else {}
    if not curves:
        logger.warning("Sin caché de T-DEED (%s): solo eventos emitidos y audio", tdeed_npz)
    field_data = possession_and_field(possession_jsonl, tracks_2d_jsonl, team_by_tracker, n_bins, fps)

    return MatchSignals(
        duration_s=duration_s, fps=fps, t=t.astype(np.float32), has_audio=has_audio,
        audio_rel=rel, audio_sustained=sustained, curves=curves,
        actions=actions_from_telemetry(telemetry, fps, team_by_tracker),
        events=events_from_telemetry(telemetry),
        **field_data,
    )


# --------------------------------------------------------------------------- #
#  Tipo de plano (solo se calcula bajo demanda, en ventanas cortas)
# --------------------------------------------------------------------------- #

def green_ratio_curve(video: Path, t0: float, t1: float, step_s: float = 1.0) -> list[tuple[float, float]]:
    """Fracción de píxeles de césped por frame en [t0, t1]. En retransmisión
    de TV separa el plano general (>= 0,5) de primeros planos, banquillo,
    grada y repeticiones cercanas (~0): la celebración de un gol es una
    racha larga de valores bajos y el partido "vuelve" cuando el plano
    general se sostiene varios segundos."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    out: list[tuple[float, float]] = []
    try:
        t = t0
        while t <= t1:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            hsv = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2HSV)
            green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255)).mean() / 255.0
            out.append((t, float(green)))
            t += step_s
    finally:
        cap.release()
    return out
