"""Coste computacional de los spotters (T-DEED, AdaSpot, sn-teamspotting y el
E2E-Spot de OpenSportsLab), un modelo por invocación (los repos tienen paquetes
``model``/``util``/``dataset`` con el mismo nombre y no pueden convivir en un
proceso).

Mide, con la MISMA entrada (clips de 100 frames muestreados a 796x448, uint8,
lote de ``--batch``), la inferencia tal cual la usa cada script de puntuación
(``core.ball_spotting.TDEEDSpotter._predict_heads`` / ``AdaSpot.predict``,
cada uno con su autocast):

* parámetros (M) y, si ``thop`` lo soporta, GFLOPs por clip;
* latencia por lote (mediana de ``--iters`` tras calentar) -> clips/s y, con
  el solape de inferencia y el muestreo de cada modelo (T-DEED y sn-teamspotting
  75 % y 1 de cada 2 frames, AdaSpot 50 % y 1 de cada 2, OSL 50 % y 1 de cada
  12), segundos de vídeo procesados por segundo de GPU (sin decodificar). Para
  OSL el lote de 796x448 se redimensiona a 398x224 dentro de ``predict`` (es
  parte de su preprocesado);
* VRAM pico del proceso (``torch.cuda.max_memory_allocated``);
* opcionalmente (``--video``) tiempo de pared de punta a punta sobre un vídeo
  (decodificación + resize + inferencia), con el bucle de cada script.

La GPU es compartida: los tiempos se toman con lo que haya corriendo; se
registra la memoria ocupada por otros procesos al empezar.

Uso (contenedor):
    python -m scripts.benchmark.spotter_cost --model tdeed:SoccerNetBall_test --video output/benchmark/cost/stoke_5min.mp4
    python -m scripts.benchmark.spotter_cost --model adaspot:SoccerNetBall_big --video ...
    python -m scripts.benchmark.spotter_cost --model teamspot:baseline --video ...
    python -m scripts.benchmark.spotter_cost --model osl:snbas --video ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "benchmark" / "cost"
CLIP_LEN, H, W = 100, 448, 796


def gpu_used_by_others_gb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split()[0]
        return round(int(out) / 1024, 1)
    except Exception:  # noqa: BLE001
        return float("nan")


def load(spec: str):
    """Devuelve (nombre, net para contar parámetros, función lote uint8 -> salida, solape de inferencia,
    frames de vídeo por frame muestreado, (alto, ancho) de entrada de la red)."""
    kind, name = spec.split(":", 1)
    if kind == "tdeed":
        from core.ball_spotting import TDEEDSpotter

        sp = TDEEDSpotter(model_name=name)
        return f"T-DEED {name}", sp._model._model, sp._predict_heads, 0.75, 2, (H, W)
    if kind == "adaspot":
        from scripts.benchmark.adaspot_scores import load as load_ada

        model, _, _, _ = load_ada(name)
        return f"AdaSpot {name.split('_')[1]}", model._model, model.predict, 0.5, 2, (H, W)
    if kind == "teamspot":
        from scripts.benchmark.teamspot_scores import load_model as load_team

        model, _ = load_team()
        return "sn-teamspotting baseline (T-DEED 200MF + equipo)", model._model, model.predict, 0.75, 2, (H, W)
    if kind == "osl":
        from scripts.benchmark.osl_scores import SIZE as OSL_SIZE, STRIDE as OSL_STRIDE
        from scripts.benchmark.osl_scores import load_model as load_osl, predict as osl_predict

        net = load_osl()
        return "E2E-Spot OpenSportsLab (SN-BAS-2025)", net, lambda x: osl_predict(net, x), 0.5, OSL_STRIDE, (OSL_SIZE[1], OSL_SIZE[0])
    raise ValueError(spec)


class _Inference(torch.nn.Module):
    """Envoltorio para thop: fija ``inference=True`` en los modelos T-DEED (thop no pasa kwargs)."""

    def __init__(self, net, inference_kwarg: bool) -> None:
        super().__init__()
        self.net = net
        self.inference_kwarg = inference_kwarg

    def forward(self, x):
        return self.net(x, inference=True) if self.inference_kwarg else self.net(x)


def flops_per_clip(net, kind: str, hw: tuple[int, int]) -> float | None:
    """GFLOPs de un clip (lote 1, fp32) con thop; None si el grafo no se deja perfilar."""
    try:
        from thop import profile

        x = torch.zeros(1, CLIP_LEN, 3, *hw, device="cuda")
        with torch.no_grad():
            macs, _ = profile(_Inference(net, kind != "osl"), inputs=(x,), verbose=False)
        torch.cuda.empty_cache()
        return round(2 * macs / 1e9, 1)
    except Exception as e:  # noqa: BLE001
        print(f"  (thop no pudo perfilar: {type(e).__name__}: {str(e)[:80]})")
        torch.cuda.empty_cache()
        return None


def end_to_end(spec: str, video: Path) -> tuple[float, int]:
    """Segundos de pared para puntuar ``video`` entero con el bucle de cada script; devuelve (s, frames)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    kind, name = spec.split(":", 1)
    out = OUT / f"{video.stem}.tdeed.{name}.npz"
    out.unlink(missing_ok=True)
    t0 = time.perf_counter()
    if kind == "tdeed":
        from core.ball_spotting import TDEEDSpotter

        TDEEDSpotter(model_name=name).ensure_scores(video, cache_path=out)
    else:
        # Los otros scripts escriben la caché junto al vídeo, con su propio nombre de modelo.
        if kind == "adaspot":
            module, extra, tag = "adaspot_scores", ["--model", name], f"AdaSpot_{name.split('_')[1]}"
        else:
            module, extra, tag = {"teamspot": ("teamspot_scores", [], "SNTeam_baseline"),
                                  "osl": ("osl_scores", [], "OSL_E2E_snbas")}[kind]
        subprocess.run(["python", "-m", f"scripts.benchmark.{module}", *extra, "--videos", str(video)],
                       check=True, capture_output=True)
        video.with_name(f"{video.stem}.tdeed.{tag}.npz").rename(out)
    return time.perf_counter() - t0, n_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="tdeed:<checkpoint> | adaspot:<SoccerNetBall_big|small> | teamspot:baseline | osl:snbas")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()

    others_gb = gpu_used_by_others_gb()
    kind = args.model.split(":")[0]
    name, net, predict, overlap, stride, hw = load(args.model)
    params_m = round(sum(p.numel() for p in net.parameters()) / 1e6, 2)
    gflops = flops_per_clip(net, kind, hw)

    batch = args.batch
    while True:  # si no cabe el lote en la GPU compartida, se reduce a la mitad y se anota
        try:
            x = torch.randint(0, 255, (batch, CLIP_LEN, 3, H, W), dtype=torch.uint8)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                predict(x)
            torch.cuda.synchronize()
            times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                predict(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch == 1:
                raise
            batch //= 2
            print(f"  (OOM: lote reducido a {batch})")
    lat = statistics.median(times)
    clips_s = batch / lat
    # Un clip son 100 frames muestreados = 100*stride frames de vídeo (8 s a 25 fps con stride 2, 48 s
    # con el stride 12 de OSL); con solape o cada frame se infiere 1/(1-o) veces.
    clip_video_s = CLIP_LEN * stride / 25.0
    video_s_per_s = clips_s * clip_video_s * (1 - overlap)
    vram_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    res = {"model": name, "spec": args.model, "params_M": params_m, "gflops_per_clip": gflops,
           "batch": batch, "latency_batch_s": round(lat, 4), "clips_per_s": round(clips_s, 2),
           "inference_overlap": overlap, "frame_stride": stride, "input_hw": list(hw),
           "clip_video_s": clip_video_s, "video_s_per_gpu_s": round(video_s_per_s, 1),
           "vram_peak_gb": vram_gb, "gpu_used_by_others_gb_at_start": others_gb}
    if args.video:
        # Libera el modelo medido: el bucle de punta a punta carga el suyo propio.
        del net, predict, x
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        secs, n = end_to_end(args.model, args.video)
        res.update({"e2e_video": str(args.video), "e2e_seconds": round(secs, 1), "e2e_frames": n,
                    "e2e_realtime_factor": round((n / 25.0) / secs, 2)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model.replace(':', '_')}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
