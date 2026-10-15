"""Vocabulario canónico de etiquetas.

El pipeline mezcla grafías ("throw-in" en core/events.py, "throw_in" en
core/ball_actions.py, "kick-off"). La API expone siempre snake_case; una
etiqueta desconocida se conserva (normalizada) en vez de fallar, para que un
checkpoint nuevo no rompa el contrato.
"""

from __future__ import annotations

KNOWN_EVENT_LABELS = ("card", "corner", "foul", "goal", "kick_off", "penalty", "shot", "substitution", "throw_in")
KNOWN_BALL_ACTION_LABELS = ("pass", "drive", "high_pass", "header", "throw_in", "background")


def canonical_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")
