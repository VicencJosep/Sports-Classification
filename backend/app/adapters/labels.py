"""Vocabulario canónico de etiquetas.

El pipeline mezcla grafías ("throw-in" en core/events.py, "throw_in" en
core/ball_actions.py, "kick-off"). La API expone siempre snake_case; una
etiqueta desconocida se conserva (normalizada) en vez de fallar, para que un
checkpoint nuevo no rompa el contrato.
"""

from __future__ import annotations

KNOWN_EVENT_LABELS = (
    "card", "corner", "foul", "free_kick", "goal", "kick_off", "offside", "penalty", "shot", "substitution", "throw_in",
)
KNOWN_BALL_ACTION_LABELS = ("pass", "drive", "high_pass", "header", "throw_in", "background")

# Etiquetas que acepta la copia revisada (``review/analysis.json``): las de la hoja de anotación manual
# (``scripts/annotation_csv.py::LABELS``) en grafía canónica. Una acción o un evento corregido a mano solo puede
# llevar una de estas, para que la hoja exportada no tenga etiquetas que el lector no entienda.
REVIEW_LABELS = (
    "pass", "drive", "high_pass", "header", "throw_in", "cross", "shot", "goal", "corner", "foul", "free_kick",
    "out", "player_successful_tackle", "ball_player_block", "penalty", "kick_off", "card", "offside", "substitution",
)


def canonical_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")
