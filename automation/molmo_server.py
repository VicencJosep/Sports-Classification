"""
automation/molmo_server.py
==========================
Servidor de inferencia de MolmoWeb-4B, pensado para ejecutarse en un intérprete
DISTINTO al del pipeline principal.

Por qué existe
--------------
El checkpoint ``allenai/MolmoWeb-4B`` se guardó con transformers 4.57.3 y su
repo oficial pinea ``transformers>=4.51,<5``. Este repo, en cambio, fija
``transformers==5.14.1`` porque lo exige la API de Siglip que usa
``Soccer_Master`` (ver README § "Problemas conocidos"). Son incompatibles y
ninguna de las dos versiones es negociable:

  * en 5.x el código remoto del checkpoint revienta en cadena —
    ``KeyError: 'default'`` en ROPE_INIT_FUNCTIONS, luego
    ``TypeError: Unexpected keyword argument image_use_col_tokens``, luego
    ``cache_position`` a None en ``prepare_inputs_for_generation``—, y cada
    parche reimplementa a mano un trozo del delta 4.x->5.x con riesgo de
    fallar en silencio (coordenadas mal calculadas en vez de excepción);
  * bajar transformers rompería el detector de fútbol entero.

Así que el VLM vive en ``/opt/venv-molmo`` (transformers 4.57.3) y se habla con
él por tubería. El pipeline sigue en su venv de siempre.

Protocolo (JSON por líneas, stdin -> stdout)
--------------------------------------------
Entrada, una petición por línea::

    {"image_b64": "<png en base64>", "prompt": "<prompt YA montado>"}
    {"cmd": "reset"}     # olvida el historial de la trayectoria
    {"cmd": "ping"}

Salida, una respuesta por línea::

    {"ok": true, "raw": "<respuesta cruda del modelo>"}
    {"ok": false, "error": "..."}

Al arrancar, cuando el modelo ya está cargado, emite ``{"ok": true,
"ready": true}``. El cliente (``RemoteMolmoPointer`` en molmo_publisher.py)
espera esa línea antes de mandar nada.

El prompt llega YA MONTADO desde el cliente: aquí no se replica ni la plantilla
de MolmoWeb ni el parser de la respuesta, que viven (y están probados) en
``automation/molmo_publisher.py``. Este proceso solo hace imagen+texto -> texto.

Uso (no se suele lanzar a mano; lo levanta RemoteMolmoPointer):
    /opt/venv-molmo/bin/python -m automation.molmo_server --model-id allenai/MolmoWeb-4B
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import traceback


def _write(payload: dict) -> None:
    """Una respuesta = una línea, con flush inmediato (el cliente lee bloqueado)."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor de inferencia MolmoWeb-4B")
    parser.add_argument("--model-id", default="allenai/MolmoWeb-4B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    # El logging del pipeline va a stderr: stdout es EXCLUSIVAMENTE el canal de
    # respuestas JSON, y cualquier print suelto ahí rompería el protocolo.
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="[molmo_server] %(levelname)s %(message)s")

    # Import tardío y por ruta: este módulo corre bajo otro venv, pero el
    # repo está montado igual, así que reutiliza el MISMO MolmoWebPointer que
    # el pipeline (plantilla del prompt y parser incluidos).
    from automation.molmo_publisher import MolmoWebPointer

    try:
        pointer = MolmoWebPointer(
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            track_history=True,
        )
    except Exception as exc:
        _write({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()})
        raise SystemExit(1)

    _write({"ok": True, "ready": True, "device": pointer.device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _write({"ok": False, "error": f"JSON inválido: {exc}"})
            continue

        cmd = req.get("cmd")
        if cmd == "ping":
            _write({"ok": True, "pong": True})
            continue
        if cmd == "reset":
            pointer.reset_history()
            _write({"ok": True, "reset": True})
            continue
        if cmd == "shutdown":
            _write({"ok": True, "bye": True})
            break

        try:
            image_png = base64.b64decode(req["image_b64"])
            raw = pointer.describe_prompt(image_png, req["prompt"])
            _write({"ok": True, "raw": raw})
        except Exception as exc:
            _write({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})


if __name__ == "__main__":
    main()
