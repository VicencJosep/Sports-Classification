"""Evalúa tramos de resumen contra la verdad de SoccerHigh (``*_intervals.srt``) con sus métricas:
P/R/F1 por fotograma (2 fps, agregadas sobre todos los partidos y media por partido), mAP por tramo y las
métricas "@T" (los mejores tramos hasta la duración del resumen real; medias por partido).

Reproduce el baseline del paper con sus propios rasgos de Hugging Face (``--source hf``: F1 0,3956 / F1@T 0,3883):

    docker exec -w /app Sports-Classification python scripts/evaluate_soccerhigh.py --split test --source hf

Con rasgos extraídos de nuestros vídeos 720p (``datasets/soccerhigh/videos/<partido>/{1,2}_720p.mkv``), el
desfase HQ -> 720p de cada mitad se toma de ``offsets.json`` (``scripts/soccerhigh_align.py``) o se estima
alineando rasgos si hay rasgos de HF del mismo partido:

    docker exec -w /app Sports-Classification python scripts/evaluate_soccerhigh.py --split test --source ours

``--predictions fichero.json`` evalúa tramos ajenos (p.ej. los clips de ``analytics/highlights``): JSON
``{"<partido>": {"1": [[start_s, end_s, score], ...], "2": [...]}}`` en segundos del vídeo HQ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.highlights.soccerhigh import (  # noqa: E402
    FPS, DEFAULT_CHECKPOINT, Segment, SoccerHighSummarizer, features_cache_path, predict, select_to_length,
)

ROOT = Path("datasets/soccerhigh")
HF_FEATURES = "HQ_224p_VideoMAEv2_Giant_K710_1408.npy"
_SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def read_split(name: str) -> list[str]:
    return [l.strip() for l in (ROOT / f"{name}.txt").read_text().splitlines() if l.strip()]


def read_srt_intervals(path: Path) -> list[tuple[int, int]]:
    """Tramos ``(inicio, fin)`` en fotogramas a 2 fps, como ``timestamp2frameidx`` de SoccerHigh (truncando)."""
    out = []
    for m in _SRT_TIME.finditer(path.read_text(encoding="utf-8-sig")):
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(v) for v in m.groups())
        t1 = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        t2 = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        out.append((int(FPS * t1), int(FPS * t2)))
    return out


def ground_truth(game: str) -> dict[int, list[tuple[int, int]]]:
    return {h: read_srt_intervals(ROOT / game / f"{h}_intervals.srt") for h in (1, 2)}


def frames_of(segments, n_frames: int) -> np.ndarray:
    m = np.zeros(n_frames, bool)
    for s, e in segments:
        m[max(s, 0): min(e, n_frames - 1) + 1] = True
    return m


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return p, r, f, iou


def evaluate(games: list[str], preds: dict[str, dict[int, list[Segment]]], n_frames: dict[str, dict[int, int]],
             frame_scores: dict[str, dict[int, np.ndarray]] | None = None) -> dict:
    from sklearn.metrics import average_precision_score

    agg = np.zeros(3, int)
    agg_raw = np.zeros(3, int)
    per_match, at_t = [], []
    y_true_all, y_score_all = [], []
    for g in games:
        gt = ground_truth(g)
        tp = fp = fn = 0
        gt_total = sum(e - s + 1 for h in (1, 2) for s, e in gt[h])
        all_segs = [(h, s) for h in (1, 2) for s in preds[g][h]]
        sel = select_to_length([s for _, s in all_segs], gt_total)
        sel_ids = {id(s) for s in sel}
        t_tp = t_fp = t_fn = 0
        t_true, t_score = [], []
        for h in (1, 2):
            n = n_frames[g][h]
            gt_m = frames_of(gt[h], n)
            pr_m = frames_of([(s.start, s.end) for s in preds[g][h]], n)
            tp += int((gt_m & pr_m).sum()); fp += int((~gt_m & pr_m).sum()); fn += int((gt_m & ~pr_m).sum())
            sc = np.zeros(n, np.float32)
            for s in preds[g][h]:
                if frame_scores is not None:
                    sc[s.start: s.end + 1] = frame_scores[g][h][s.start: s.end + 1]
                else:
                    sc[s.start: s.end + 1] = s.score
            y_true_all.append(gt_m.astype(int)); y_score_all.append(sc)
            sel_h = [s for s in preds[g][h] if id(s) in sel_ids]
            sel_m = frames_of([(s.start, s.end) for s in sel_h], n)
            t_tp += int((gt_m & sel_m).sum()); t_fp += int((~gt_m & sel_m).sum()); t_fn += int((gt_m & ~sel_m).sum())
            sc_t = np.zeros(n, np.float32)
            for s in sel_h:
                sc_t[s.start: s.end + 1] = s.score
            t_true.append(gt_m.astype(int)); t_score.append(sc_t)
        agg += (tp, fp, fn)
        if frame_scores is not None:
            # F1 de la clasificación cruda por fotograma (prob > 0,5, sin tramos ni NMS): es la cifra "F1" del paper
            for h in (1, 2):
                gt_m = frames_of(gt[h], n_frames[g][h])
                raw = frame_scores[g][h][: n_frames[g][h]] > 0.5
                agg_raw += (int((gt_m & raw).sum()), int((~gt_m & raw).sum()), int((gt_m & ~raw).sum()))
        p, r, f, iou = prf(tp, fp, fn)
        per_match.append({"game": g, "precision": p, "recall": r, "f1": f, "iou": iou,
                          "n_pred": len(all_segs), "n_gt": sum(len(gt[h]) for h in (1, 2)),
                          "pred_s": sum(s.n_frames for _, s in all_segs) / FPS, "gt_s": gt_total / FPS})
        tp_, tr_ = np.concatenate(t_true), np.concatenate(t_score)
        try:
            ap = float(average_precision_score(tp_, tr_))
        except ValueError:
            ap = 0.0
        p, r, f, iou = prf(t_tp, t_fp, t_fn)
        at_t.append({"AP": ap, "precision": p, "recall": r, "f1": f, "iou": iou})
    p, r, f, iou = prf(*agg)
    y_true, y_score = np.concatenate(y_true_all), np.concatenate(y_score_all)
    summary = {
        "n_games": len(games),
        "frame": {"precision": p, "recall": r, "f1": f, "iou": iou,
                  "average_f1": float(np.mean([m["f1"] for m in per_match])),
                  "mAP": float(average_precision_score(y_true, y_score)) if y_true.any() else 0.0},
        "frame_raw": dict(zip(("precision", "recall", "f1", "iou"), prf(*agg_raw))) if frame_scores is not None else None,
        "at_T": {k: float(np.mean([m[k] for m in at_t])) for k in ("AP", "precision", "recall", "f1", "iou")},
        "per_match": per_match,
    }
    return summary


def print_summary(s: dict) -> None:
    f = s["frame"]
    print(f"\n{s['n_games']} partidos")
    print(f"  por fotograma: P {f['precision']:.4f}  R {f['recall']:.4f}  F1 {f['f1']:.4f}  IoU {f['iou']:.4f}  "
          f"(F1 media por partido {f['average_f1']:.4f}; mAP {f['mAP']:.4f})")
    if s.get("frame_raw"):
        r = s["frame_raw"]
        print(f"  clasif. cruda: P {r['precision']:.4f}  R {r['recall']:.4f}  F1 {r['f1']:.4f}  (prob > 0,5 sin NMS)")
    t = s["at_T"]
    print(f"  @T:            P {t['precision']:.4f}  R {t['recall']:.4f}  F1 {t['f1']:.4f}  IoU {t['iou']:.4f}  mAP {t['AP']:.4f}")


def _hf_features(game: str, half: int) -> Path:
    return ROOT / game / f"{half}_{HF_FEATURES}"


def _our_features(game: str, half: int) -> Path:
    return features_cache_path(ROOT / "videos" / game / f"{half}_720p.mkv")


def load_offsets(game: str) -> dict[int, float] | None:
    path = ROOT / "videos" / game / "offsets.json"
    if path.exists():
        d = json.loads(path.read_text())
        return {1: float(d["1"]), 2: float(d["2"])}
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "validation", "test"])
    p.add_argument("--games", nargs="*", default=[])
    p.add_argument("--source", choices=["hf", "ours"], default="hf")
    p.add_argument("--predictions", type=Path, help="tramos ajenos en segundos HQ (ver cabecera)")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--window-s", type=float, default=60.0)
    p.add_argument("--out", type=Path, help="JSON con el resumen y las cifras por partido")
    p.add_argument("--only-available", action="store_true", help="salta los partidos sin rasgos")
    args = p.parse_args()

    games = list(args.games) + (read_split(args.split) if args.split else [])
    if not games:
        p.error("indica --split o --games")

    preds: dict[str, dict[int, list[Segment]]] = {}
    n_frames: dict[str, dict[int, int]] = {}
    frame_scores: dict[str, dict[int, np.ndarray]] | None = None
    if args.predictions:
        ext = json.loads(args.predictions.read_text())
        for g in games:
            preds[g] = {h: [Segment(int(round(s * FPS)), int(round(e * FPS)) - 1, float(sc)) for s, e, sc in ext.get(g, {}).get(str(h), [])]
                        for h in (1, 2)}
            n_frames[g] = {h: int(np.load(_hf_features(g, h), mmap_mode="r").shape[0]) if _hf_features(g, h).exists()
                           else max([s.end + 1 for s in preds[g][h]] + [1]) for h in (1, 2)}
    else:
        model = SoccerHighSummarizer.from_checkpoint(args.checkpoint)
        frame_scores = {}
        kept = []
        for g in games:
            paths = {h: (_hf_features(g, h) if args.source == "hf" else _our_features(g, h)) for h in (1, 2)}
            if not all(pp.exists() for pp in paths.values()):
                if args.only_available:
                    continue
                raise SystemExit(f"faltan rasgos de {g}: {paths}")
            kept.append(g)
            preds[g], n_frames[g], frame_scores[g] = {}, {}, {}
            offsets = load_offsets(g) if args.source == "ours" else None
            for h in (1, 2):
                feats = np.load(paths[h], mmap_mode="r")
                segs, prob = predict(model, feats, args.threshold, args.window_s)
                if args.source == "ours":
                    # 720p -> HQ: desplazar los tramos y la curva (sin offsets.json se asume desfase 0)
                    off = int(round((offsets or {1: 0.0, 2: 0.0})[h] * FPS))
                    hf = _hf_features(g, h)
                    n_hq = int(np.load(hf, mmap_mode="r").shape[0]) if hf.exists() else len(prob) + off
                    segs = [Segment(s.start + off, s.end + off, s.score) for s in segs if s.end + off >= 0]
                    shifted = np.zeros(n_hq, np.float32)
                    lo, hi = max(off, 0), min(n_hq, len(prob) + off)
                    shifted[lo:hi] = prob[lo - off: hi - off]
                    prob, n = shifted, n_hq
                    segs = [Segment(max(s.start, 0), min(s.end, n - 1), s.score) for s in segs if s.start < n]
                else:
                    n = len(prob)
                preds[g][h], n_frames[g][h], frame_scores[g][h] = segs, n, prob
        games = kept
    summary = evaluate(games, preds, n_frames, frame_scores)
    print_summary(summary)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
