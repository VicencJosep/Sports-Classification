"""Embedding V-JEPA2 SIN el recorte central 256x256 (que solo deja ver el 49% del ancho),
para los mismos 1.149 candidatos a tiro que extract_cand_emb.py. Variantes:
  squash: frame entero a 256x256 (deforma la proporción, misma resolución vertical)
  wide:   frame entero a 256x448 (proporción real, 16x28 parches; RoPE parcheado para
          rejilla no cuadrada: el código de transformers asume alto == ancho)
Uso: python - VARIANTE [LIMIT]   (con LIMIT: prueba sin guardar). Reanudable por partido."""
import glob, sys, time
from pathlib import Path
import numpy as np
import torch
from analytics.engine import read_event_windows
from core.spotting import find_peaks
from core.events import EventClassifier, sample_frames_window
from transformers.models.vjepa2 import modeling_vjepa2 as mv

THETA, NMS, PATCH = 0.02, 4.0, 16
SIZES = {"squash": (256, 256), "wide": (256, 448)}
VARIANT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
H, W = SIZES[VARIANT]


def get_position_ids(self, x, masks=None):
    """Como VJEPA2RopeAttention.get_position_ids, con rejilla grid_h x grid_w."""
    gh = getattr(self, "grid_h", self.grid_size)
    gw = getattr(self, "grid_w", self.grid_size)
    if masks is not None:
        ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
    else:
        ids = torch.arange(x.size(1), device=x.device)
    tpf = gh * gw
    frame_ids = ids // tpf
    rem = ids - tpf * frame_ids
    height_ids = rem // gw
    return frame_ids, height_ids, rem - gw * height_ids


orig_get_position_ids = mv.VJEPA2RopeAttention.get_position_ids
mv.VJEPA2RopeAttention.get_position_ids = get_position_ids

clf = EventClassifier()


def set_grid(h, w):
    for mod in clf.model.modules():
        if isinstance(mod, mv.VJEPA2RopeAttention):
            mod.grid_h, mod.grid_w = h // PATCH, w // PATCH


def embed(frames, h, w, **proc_kwargs):
    set_grid(h, w)
    inputs = clf.processor(videos=frames, return_tensors="pt", **proc_kwargs).to(clf.device)
    assert tuple(inputs["pixel_values_videos"].shape[-2:]) == (h, w), inputs["pixel_values_videos"].shape
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden = clf.model(**inputs).last_hidden_state.squeeze(0)
    assert hidden.shape[0] == (64 // 2) * (h // PATCH) * (w // PATCH), hidden.shape
    return torch.cat([hidden.mean(0), hidden.max(0)[0], hidden.std(0)]).float().cpu().numpy()


def embed_variant(frames):
    return embed(frames, H, W, do_center_crop=False, size={"height": H, "width": W})


OUT = Path(f"output/shot_benchmark/candidate_embeddings_{VARIANT}")
OUT.mkdir(parents=True, exist_ok=True)
checked = False
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
    embs, t0 = [], time.perf_counter()
    for i in idx:
        r = rows[i]
        frames = sample_frames_window(video, r["start_s"], r["end_s"], clf.num_frames)
        if not checked:
            # El parche de RoPE con rejilla cuadrada debe reproducir el embedding original
            # (recorte 256x256 de siempre) y, por tanto, las probabilidades de la caché.
            e_patch = embed(frames, 256, 256)
            p = clf.clf.predict_proba(e_patch.reshape(1, -1))[0]
            cached = np.array([r["probabilities"][str(c)] for c in clf.label_encoder.classes_])
            print(f"check parche RoPE (256x256 con recorte): max |Δp| vs caché = {np.abs(p - cached).max():.4f}", flush=True)
            checked = True
        embs.append(embed_variant(frames).astype(np.float16))
    dt = (time.perf_counter() - t0) / len(idx)
    print(f"[{VARIANT}] {video.parent.name}: {len(idx)} candidatos, {dt:.2f} s/cand", flush=True)
    if LIMIT:
        break
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, idx=np.array(idx), t_s=np.array([times[i] for i in idx]), emb=np.stack(embs))
    tmp.rename(out)
print(f"FIN extracción {VARIANT}", flush=True)
