"""Embedding V-JEPA2 (3072-d) de la ventana de cada candidato a tiro (picos θ>=0.02,
NMS 4 s). Comprueba que el SVC reproduce las
probabilidades de la caché. Reanudable por partido. Uso: python - [LIMIT] (prueba sin guardar)."""
import glob, sys, time
from pathlib import Path
import numpy as np
from analytics.engine import read_event_windows
from core.spotting import find_peaks
from core.events import EventClassifier, sample_frames_window

THETA, NMS = 0.02, 4.0
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path("output/shot_benchmark/candidate_embeddings")
OUT.mkdir(parents=True, exist_ok=True)

clf = EventClassifier()
classes = [str(c) for c in clf.label_encoder.classes_]
for cache in sorted(glob.glob("datasets/SoccerNet/england_efl/2019-2020/*/720p.event_windows.vjepa2.w6s3.jsonl")):
    meta, rows = read_event_windows(Path(cache))
    video = Path(meta["source_video"])
    out = OUT / f"{video.parent.name}.npz"
    if out.exists() and LIMIT is None:
        print("ya hecho:", video.parent.name, flush=True)
        continue
    times = [(r["start_s"] + r["end_s"]) / 2 for r in rows]
    curve = [r["probabilities"]["shot"] + r["probabilities"]["goal"] for r in rows]
    idx = [times.index(t) for t, _ in find_peaks(times, curve, THETA, NMS)]
    if LIMIT:
        idx = idx[:LIMIT]
    embs, maxdiff, t0 = [], 0.0, time.perf_counter()
    for i in idx:
        r = rows[i]
        e = clf._embed(sample_frames_window(video, r["start_s"], r["end_s"], clf.num_frames))
        p = clf.clf.predict_proba(e.reshape(1, -1))[0]
        maxdiff = max(maxdiff, float(np.abs(p - np.array([r["probabilities"][c] for c in classes])).max()))
        embs.append(e.astype(np.float16))
    dt = (time.perf_counter() - t0) / len(idx)
    print(f"{video.parent.name}: {len(idx)} candidatos, {dt:.2f} s/cand, max |Δp| vs caché = {maxdiff:.4f}", flush=True)
    if LIMIT:
        break
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, idx=np.array(idx), t_s=np.array([times[i] for i in idx]), emb=np.stack(embs))
    tmp.rename(out)
print("FIN extracción", flush=True)
