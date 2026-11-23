"""
Evalúa variantes de posesión de equipo por fotograma con las métricas del reto
(``spell_f1_mean`` y ``macro_f1``) sobre los partidos cacheados por
``cache_possession_data.py``.

    python scripts/benchmark/eval_possession.py --dataset metrica
    python scripts/benchmark/eval_possession.py --dataset idsse --variants nearest_s50 rules soft_vit
    python scripts/benchmark/eval_possession.py --dataset all --out output/posesion/eval_all.csv

Variantes:
  nearest_raw   equipo del jugador más cercano, D si no hay balón (línea base del reto sin suavizar)
  nearest_s50   + voto mayoritario ±50 fotogramas (la línea base publicada, spellF1 0,24 en el reto)
  nearest_s125  + voto ±125
  rules         máquina de estados de control (último control válido, duelo, cambio cinemático)
  rules_msf     rules + filtro de tramos < min_len
  soft_vit      evidencias blandas + Viterbi (coste de cambio, duración mínima)
  soft_vit_msf  soft_vit + filtro de tramos
  gbm_argmax    clasificador por fotograma (--model) sin decodificar
  gbm_vit       clasificador + Viterbi
  gbm_vit_msf   clasificador + Viterbi + filtro de tramos
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.team_possession import control, decode, simulate, spell_metrics as pm  # noqa: E402
import joblib  # noqa: E402

from core.team_possession.features import CONTEXT_OFFSETS_S, FeatureConfig, compute_features, feature_matrix  # noqa: E402
from core.team_possession.io_sportec import load_npz  # noqa: E402
from core.team_possession.labels import merge_short_spells  # noqa: E402

CACHE = ROOT / "output" / "posesion" / "cache"
ALL_VARIANTS = ("nearest_raw", "nearest_s50", "nearest_s125", "rules", "rules_msf", "soft_vit", "soft_vit_msf")
_MODEL: dict = {}
_DATA: dict = {}


def predict(variant: str, feats, cfg: control.ControlConfig, args) -> np.ndarray:
    runs = feats.period_runs
    if variant.startswith("nearest"):
        base = control.nearest_labels(feats)
        if variant == "nearest_raw":
            return base
        return decode.majority_smooth(base, int(variant.split("_s")[1]), runs=runs)
    dead = control.dead_ball_evidence(feats, cfg)
    if variant.startswith("rules"):
        out = control.rule_labels(feats, cfg, dead)
        if variant == "rules_msf":
            out = decode.min_segment_filter(out, args.min_len, runs=runs)
        return out
    if variant.startswith("gbm"):
        if args.model is None:
            raise SystemExit("gbm_* necesita --model")
        bundle = _MODEL.setdefault("m", joblib.load(args.model))
        Xm = feature_matrix(feats, _DATA["d"], context_offsets_s=bundle.get("context", CONTEXT_OFFSETS_S))
        logp = np.log(np.clip(bundle["model"].predict_proba(Xm), 1e-6, 1.0))
        if variant == "gbm_argmax":
            return logp.argmax(1).astype(np.int8)
        out = decode.viterbi_decode(logp, switch_cost=args.switch_cost, min_dur=args.min_dur, runs=runs)
        if variant == "gbm_vit_msf":
            out = decode.min_segment_filter(out, args.min_len, runs=runs)
        return out
    if variant.startswith("soft_vit"):
        emit = control.log_softmax(control.emission_scores(feats, cfg, dead))
        out = decode.viterbi_decode(emit, switch_cost=args.switch_cost, min_dur=args.min_dur, runs=runs)
        if variant == "soft_vit_msf":
            out = decode.min_segment_filter(out, args.min_len, runs=runs)
        return out
    raise ValueError(variant)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dataset", default="all", help="metrica | idsse | all | prefijo de fichero")
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--switch-cost", type=float, default=8.0)
    ap.add_argument("--min-dur", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--pz", type=float, default=1.5, help="radio de la zona de posesión (m)")
    ap.add_argument("--limit", type=int, default=0, help="solo los N primeros partidos")
    ap.add_argument("--drop-ball", type=float, default=0.0, help="borra el balón en esta fracción de fotogramas (simula el reto)")
    ap.add_argument("--anon", action="store_true", help="trocea las trayectorias en slots anónimos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=Path, default=None, help="modelo joblib de train_possession_model.py para la variante gbm_vit")
    ap.add_argument("--merge-short", type=float, default=0.0, help="funde en la anterior las posesiones de la verdad más cortas que estos segundos (IDSSE es por toque)")
    args = ap.parse_args()

    prefix = "" if args.dataset == "all" else args.dataset
    files = sorted(CACHE.glob(f"{prefix}*.npz"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"sin cachés en {CACHE} para '{prefix}'")
    cfg = control.ControlConfig(pz_radius_m=args.pz)
    rows = []
    for path in files:
        data = load_npz(path)
        if args.drop_ball > 0:
            data = simulate.drop_ball(data, args.drop_ball, seed=args.seed)
        if args.anon:
            data = simulate.anonymize_slots(data, seed=args.seed)
        t0 = time.time()
        feats = compute_features(data, FeatureConfig(duel_radius_m=cfg.duel_radius_m))
        _DATA["d"] = data
        t_feat = time.time() - t0
        gt = data.labels
        if args.merge_short > 0:
            gt = merge_short_spells(gt, int(args.merge_short * data.fps))
        for v in args.variants:
            t1 = time.time()
            pred = predict(v, feats, cfg, args)
            res = pm.evaluate(gt, pred)
            rows.append({"match": data.name, "variant": v, "spell_f1": res["spell_f1_mean"],
                         "macro_f1": res["macro_f1"], "alive_dead_f1": res["alive_dead_f1"],
                         "n_gt": res["n_gt_spells"], "n_pred": res["n_pred_spells"],
                         "dur_pred_s": res["mean_spell_duration_s"], "dur_gt_s": res["gt_mean_spell_duration_s"],
                         "secs": time.time() - t1})
            r = rows[-1]
            print(f"{r['match']:<14} {v:<13} spell {r['spell_f1']:.3f}  macro {r['macro_f1']:.3f}  "
                  f"vivo/muerto {r['alive_dead_f1']:.3f}  posesiones {r['n_pred']:>4}/{r['n_gt']:<4} "
                  f"dur {r['dur_pred_s']:5.1f}/{r['dur_gt_s']:4.1f} s  {r['secs']:.1f}s", flush=True)
        print(f"  (rasgos {t_feat:.1f} s)")
    print("\nMedia por variante:")
    for v in args.variants:
        sel = [r for r in rows if r["variant"] == v]
        print(f"  {v:<13} spell {np.mean([r['spell_f1'] for r in sel]):.3f}  "
              f"macro {np.mean([r['macro_f1'] for r in sel]):.3f}  "
              f"vivo/muerto {np.mean([r['alive_dead_f1'] for r in sel]):.3f}  ({len(sel)} partidos)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("escrito", args.out)


if __name__ == "__main__":
    main()
