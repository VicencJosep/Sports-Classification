"""Backend FastAPI del proyecto Sports-Classification.

Puente entre el pipeline Python existente (``python -m main``) y el frontend
React. Este paquete no importa nada de ML: el pipeline se lanza como
subproceso y sus ficheros de salida se adaptan al dominio (ver ``adapters/``).
"""
