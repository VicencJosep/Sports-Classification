"""Vocabularios y utilidades compartidas por los esquemas."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Literal

PartKind = Literal["full", "first-half", "second-half"]
AnalysisStatus = Literal["pending", "processing", "completed", "error"]
AnalysisMode = Literal["clip", "full-match"]

PART_LABELS: dict[str, str] = {"full": "Full match", "first-half": "First half", "second-half": "Second half"}

# Ids generados por el servidor: 16 hex. El patrón se usa también en las rutas
# para que un id nunca pueda convertirse en una ruta de fichero arbitraria.
ID_PATTERN = r"^[0-9a-f]{16}$"


def new_id() -> str:
    return secrets.token_hex(8)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
