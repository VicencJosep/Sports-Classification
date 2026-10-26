"""Lector del CSV de anotación manual (``output/annotations/<partido>_Ground_Truth.csv``).

Es la hoja con la que se anota a mano, viendo el vídeo con la telemetría encima, la verdad de acciones y
eventos de los partidos propios (juveniles, cámara única). Una fila por acción o evento:

    tiempo,etiqueta,dorsal,equipo
    25:13.1,pass,3032,away
    26:16.3,header,3290,away
    27:28.1,shot,3464,home
    27:31.0,corner,3479,away

* ``tiempo``: segundos de VÍDEO como ``mm:ss.s`` o ``h:mm:ss.s`` (también vale un número de segundos). Es el
  tiempo que enseña el reproductor: la conversión a frame se hace con el fps real del vídeo (30 en el
  Espanyol, 25 en los demás), nunca con un fps fijo.
* ``etiqueta``: acción o evento (mayúsculas, guiones y guiones bajos dan igual; ver ``LABELS``). Acciones de
  balón: ``pass``, ``drive``, ``high pass``, ``header``, ``throw in``, ``cross``, ``shot``, ``out``, ``free kick``,
  ``tackle``, ``block``. Eventos (misma hoja, mismas columnas): ``shot``, ``goal``, ``corner``, ``foul``,
  ``free kick``, ``penalty``, ``kick off``, ``card``, ``offside``, ``throw in``. Un ``shot`` es a la vez acción
  (para el clasificador, cuando la tenga) y evento (para T-DEED); un ``goal`` cuenta como tiro y como gol.
* ``dorsal``: el ``tracker_id`` del pipeline del jugador que ejecuta (es lo que se ve en el vídeo anotado),
  no el dorsal real. Vacío si no se sabe.
* ``equipo``: ``home`` / ``away`` (también ``local`` / ``visitante``).

Tolerante con lo que deja una anotación a mano: filas en blanco, marcas de progreso ``---`` sueltas o pegadas al
final de un campo (``away------``), espacios de más, encabezado repetido. Lo que no entiende lo avisa con el
número de línea y lo deja fuera (equipo ``homeaway`` -> sin equipo, etiqueta desconocida -> sin acción ni evento).

Uso desde otros scripts:

    from scripts.annotation_csv import read_annotation_csv
    rows, warnings = read_annotation_csv(path)
    rows[0] -> {"line": 2, "t_s": 1513.1, "label_raw": "pass", "ball_action_label": "pass",
                "event_label": None, "tracker_id": 3032, "team": "away"}

Y como comprobación rápida de una hoja (resumen y avisos):

    python -m scripts.annotation_csv output/annotations/espanyol_damm_Ground_Truth.csv
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# etiqueta normalizada -> (acción de balón, evento). Las acciones usan el vocabulario de core/ball_actions.py y
# ground_truth/*.gt.json; los eventos, las claves de core.ball_spotting.EVENT_SOURCES.
LABELS: dict[str, tuple[Optional[str], Optional[str]]] = {
    "pass": ("pass", None),
    "drive": ("drive", None),
    "high pass": ("high_pass", None),
    "header": ("header", None),
    "throw in": ("throw_in", "throw-in"),
    "cross": ("cross", None),
    "shot": ("shot", "shot"),
    "goal": ("goal", "goal"),
    "out": ("out", None),
    "free kick": ("free_kick", "free-kick"),
    "tackle": ("player_successful_tackle", None),
    "block": ("ball_player_block", None),
    "corner": (None, "corner"),
    "foul": (None, "foul"),
    "penalty": (None, "penalty"),
    "kick off": (None, "kick-off"),
    "card": (None, "card"),
    "offside": (None, "offside"),
    "substitution": (None, "substitution"),
}
_SYNONYMS = {
    "highpass": "high pass", "throwin": "throw in", "throw": "throw in", "freekick": "free kick",
    "kickoff": "kick off", "ball player block": "block", "player successful tackle": "tackle",
    "yellow card": "card", "red card": "card", "sub": "substitution", "saque de banda": "throw in",
    "pase": "pass", "conduccion": "drive", "conducción": "drive", "cabezazo": "header", "tiro": "shot",
    "gol": "goal", "corner kick": "corner", "falta": "foul", "fuera": "out",
}
_TEAMS = {"home": "home", "away": "away", "local": "home", "visitante": "away", "h": "home", "a": "away"}
_DASHES = re.compile(r"[-_–—]+")


def normalize_label(text: str) -> str:
    s = _DASHES.sub(" ", (text or "").strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    return _SYNONYMS.get(s, s)


def parse_time(text: str) -> Optional[float]:
    """``mm:ss.s`` | ``h:mm:ss.s`` | segundos -> segundos; None si no se entiende."""
    s = (text or "").strip().strip("-").strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0].replace(",", "."))
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1].replace(",", "."))
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2].replace(",", "."))
    except ValueError:
        return None
    return None


def _clean(field: str) -> str:
    """Quita espacios y las marcas de progreso ``---`` pegadas al final (``away------``)."""
    return (field or "").strip().rstrip("-–—").strip()


def read_annotation_csv(path: Path | str) -> tuple[list[dict], list[str]]:
    """Filas válidas (ordenadas por tiempo) y avisos legibles (con número de línea)."""
    rows: list[dict] = []
    warnings: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(csv.reader(f), start=1):
            cells = [_clean(c) for c in raw]
            if not any(cells):
                continue
            if cells[0].lower().startswith("tiempo") or cells[0].lower() in ("time", "t"):
                continue                                   # encabezado (puede repetirse)
            if all(set(c) <= set("-–— ") for c in cells):
                continue                                   # marca de progreso sola
            cells += [""] * (4 - len(cells))
            t_text, label_text, dorsal_text, team_text = cells[:4]
            t_s = parse_time(t_text)
            if t_s is None:
                warnings.append(f"línea {line_no}: tiempo ilegible '{t_text}' (fila ignorada)")
                continue
            label = normalize_label(label_text)
            action, event = LABELS.get(label, (None, None))
            if label not in LABELS:
                warnings.append(f"línea {line_no}: etiqueta desconocida '{label_text}' (sin acción ni evento)")
            tracker_id: Optional[int] = None
            if dorsal_text:
                try:
                    tracker_id = int(float(dorsal_text))
                    if tracker_id < 0:
                        tracker_id = None
                except ValueError:
                    warnings.append(f"línea {line_no}: dorsal ilegible '{dorsal_text}' (sin jugador)")
            team = _TEAMS.get(team_text.lower())
            if team_text and team is None:
                warnings.append(f"línea {line_no}: equipo ilegible '{team_text}' (sin equipo)")
            rows.append({
                "line": line_no, "t_s": round(t_s, 3), "label_raw": label_text.strip(),
                "ball_action_label": action, "event_label": event, "tracker_id": tracker_id, "team": team,
            })
    for prev, cur in zip(rows, rows[1:]):
        if cur["t_s"] < prev["t_s"]:
            warnings.append(f"línea {cur['line']}: tiempo {cur['t_s']:.1f} anterior al de la línea {prev['line']} "
                            f"({prev['t_s']:.1f}): ¿error de tecleo?")
    rows.sort(key=lambda r: (r["t_s"], r["line"]))
    return rows, warnings


def summarize(rows: list[dict]) -> dict:
    return {
        "n_rows": len(rows),
        "span_s": [rows[0]["t_s"], rows[-1]["t_s"]] if rows else None,
        "ball_action_label_counts": dict(Counter(r["ball_action_label"] for r in rows if r["ball_action_label"])),
        "event_label_counts": dict(Counter(r["event_label"] for r in rows if r["event_label"])),
        "team_counts": dict(Counter(r["team"] or "?" for r in rows)),
        "n_without_tracker": sum(1 for r in rows if r["tracker_id"] is None),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        rows, warnings = read_annotation_csv(arg)
        s = summarize(rows)
        print(f"{arg}: {s['n_rows']} filas", end="")
        if s["span_s"]:
            a, b = s["span_s"]
            print(f", de {int(a // 60)}:{a % 60:04.1f} a {int(b // 60)}:{b % 60:04.1f} ({(b - a) / 60:.1f} min)")
        else:
            print()
        print(f"  acciones: {s['ball_action_label_counts']}")
        print(f"  eventos:  {s['event_label_counts']}")
        print(f"  equipos:  {s['team_counts']}   sin jugador: {s['n_without_tracker']}")
        for w in warnings:
            print(f"  AVISO {w}")


if __name__ == "__main__":
    main()
