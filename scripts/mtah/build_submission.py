"""
Genera la entrega del reto More than a Hack 2027 (fase 1, posesión) a partir
del ATD táctico y la evalúa en la ventana pública.

    python scripts/mtah/build_submission.py --data datasets/posesion/mtah2027/data \\
        --variant gbm_vit --model output/posesion/models/hgb_possession.joblib \\
        --out output/posesion/submissions/gbm_vit.csv

Los datos del partido (BAR-ATH) no están en el repo: hay que inscribirse,
descargarlos y descomprimirlos en ``--data`` (``tactical/BAR-ATH_tactical_atd.xml``
y ``.txt``). La fila i del CSV es el fotograma i del vídeo táctico; el ATD
cuenta desde 1 (``video_frame = frame_id - 1``). Mitades evaluadas (fotogramas
de vídeo): 11185-90651 y 110808-190197; aquí el periodo 1 cubre hasta el 110807
y el periodo 2 el resto, para que todo fotograma reciba etiqueta.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.team_possession import control, decode, io_metrica, spell_metrics as pm  # noqa: E402
from core.team_possession.features import CONTEXT_OFFSETS_S, FeatureConfig, compute_features, feature_matrix  # noqa: E402
from core.team_possession.labels import write_submission_csv  # noqa: E402

N_VIDEO_FRAMES = 199319
HALF2_FIRST_VIDEO_FRAME = 110808  # primer fotograma calificado de la 2.ª parte (saque real en el 113742)
PITCH = (105.3, 68.0)  # medido por los organizadores (los metadatos dicen 105 x 68)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", type=Path, default=ROOT / "datasets" / "posesion" / "mtah2027" / "data")
    ap.add_argument("--game", default="BAR-ATH")
    ap.add_argument("--variant", default="soft_vit", choices=("nearest_s50", "soft_vit", "soft_vit_msf", "gbm_vit", "gbm_vit_msf"))
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--switch-cost", type=float, default=8.0)
    ap.add_argument("--min-dur", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=25)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=ROOT / "datasets" / "posesion" / "mtah2027" / "evaluation" / "gt" / f"BAR-ATH_possession_gt_public.csv")
    args = ap.parse_args()

    xml = args.data / "tactical" / f"{args.game}_tactical_atd.xml"
    txt = args.data / "tactical" / f"{args.game}_tactical_atd.txt"
    if not xml.exists() or not txt.exists():
        sys.exit(f"faltan {xml} o {txt}: descarga los datos del reto (ver DATA.md) y descomprímelos en {args.data}")
    # frame_id del ATD = fotograma de vídeo + 1
    periods = [(1, HALF2_FIRST_VIDEO_FRAME, 1), (HALF2_FIRST_VIDEO_FRAME + 1, 10 ** 9, 2)]
    data = io_metrica.load_epts(xml, txt, name=args.game, pitch_length=PITCH[0], pitch_width=PITCH[1], periods=periods)
    print(data.summary())
    feats = compute_features(data, FeatureConfig())
    cfg = control.ControlConfig()
    runs = feats.period_runs
    if args.variant == "nearest_s50":
        pred = decode.majority_smooth(control.nearest_labels(feats), 50, runs=runs)
    elif args.variant.startswith("soft_vit"):
        emit = control.log_softmax(control.emission_scores(feats, cfg))
        pred = decode.viterbi_decode(emit, switch_cost=args.switch_cost, min_dur=args.min_dur, runs=runs)
    else:
        import joblib
        bundle = joblib.load(args.model)
        logp = np.log(np.clip(bundle["model"].predict_proba(feature_matrix(feats, data, context_offsets_s=bundle.get("context", CONTEXT_OFFSETS_S))), 1e-6, 1.0))
        pred = decode.viterbi_decode(logp, switch_cost=args.switch_cost, min_dur=args.min_dur, runs=runs)
    if args.variant.endswith("_msf"):
        pred = decode.min_segment_filter(pred, args.min_len, runs=runs)

    # del índice del ATD al del vídeo: fila i = frame_id i + 1
    video = np.full(N_VIDEO_FRAMES, 3, dtype=np.int8)
    vf = data.frame_index - 1
    ok = (vf >= 0) & (vf < N_VIDEO_FRAMES)
    video[vf[ok]] = pred[ok]
    write_submission_csv(args.out, video, n_frames=N_VIDEO_FRAMES)
    print("entrega escrita en", args.out)
    if args.gt.exists():
        gt = pm.label_codes(pm.read_label_csv(args.gt))
        res = pm.evaluate(gt, video)
        print(f"ventana pública: spellF1 {res['spell_f1_mean']:.4f}  macroF1 {res['macro_f1']:.4f}  "
              f"posesiones {res['n_pred_spells']}/{res['n_gt_spells']}  vivo/muerto {res['alive_dead_f1']:.3f}")
        print("(17 posesiones: el ruido de esta ventana es de ±0,15 o más; no elegir variantes con ella)")


if __name__ == "__main__":
    main()
