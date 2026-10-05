"""Montaje de tiros detectados por T-DEED en un partido NO visto en su entrenamiento, para revisarlos a ojo.
Cada clip: 4 s antes y 4 s después del pico, con la puntuación y si coincide con un tiro anotado (±3 s).
Al final, los tiros anotados que T-DEED no detectó."""
import json, subprocess, sys
from pathlib import Path
import cv2, numpy as np
from core.ball_spotting import read_scores, scores_cache_path, spot
from scripts.evaluate_spotting import Prediction, match_1to1

MATCH, MODEL, TH = sys.argv[1], "SoccerNetBall_test", 0.3
d = Path(f"datasets/SoccerNet/england_efl/2019-2020/2019-10-01 - {MATCH}")
video = d / "720p.mp4"
scores = read_scores(scores_cache_path(video, MODEL))
peaks = spot(scores, "SHOT", TH, 4.0, smooth_frames=6)
ann = json.load(open(d / "Labels-ball.json"))["annotations"]
gt = sorted(int(a["position"]) / 1000 for a in ann if a["label"] == "SHOT")
preds = [Prediction(f / scores.fps, f / scores.fps, f / scores.fps, s) for f, s in peaks]
pairs = match_1to1(gt, preds, 3.0)
hit_pred = {pi for _, pi in pairs}; hit_gt = {gi for gi, _ in pairs}
near = lambda t: ", ".join(sorted({a["label"] for a in ann if abs(int(a["position"]) / 1000 - t) <= 3})) or "nada anotado"

items = [(p.t_s, f"T-DEED tiro {p.score:.2f}  |  {'ACIERTO' if i in hit_pred else 'FALSA ALARMA'}  |  anotado: {near(p.t_s)}", i in hit_pred)
         for i, p in enumerate(preds)]
items += [(g, f"TIRO NO DETECTADO (anotado SHOT)  |  max T-DEED cerca: {max(s for f, s in zip(scores.frame_indices(), scores.curve('SHOT')) if abs(f / scores.fps - g) <= 3):.2f}", None)
          for gi, g in enumerate(gt) if gi not in hit_gt]
cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = 854, 480
out_raw = Path(f"/tmp/montage_{MODEL}_raw.mp4")
w = cv2.VideoWriter(str(out_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
for k, (t, text, ok) in enumerate(items, 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((t - 4) * fps))
    for j in range(int(8 * fps)):
        ret, fr = cap.read()
        if not ret: break
        fr = cv2.resize(fr, (W, H))
        tt = t - 4 + j / fps
        color = (0, 200, 0) if ok else (0, 0, 230) if ok is False else (0, 165, 255)
        cv2.rectangle(fr, (0, 0), (W, 58), (0, 0, 0), -1)
        cv2.putText(fr, f"[{k}/{len(items)}] {int(t // 60)}:{t % 60:04.1f}  {text}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(fr, "pico de T-DEED" if abs(tt - t) < 0.15 and ok is not None else "", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # barra de progreso del clip con la marca del pico en el centro
        cv2.line(fr, (0, H - 4), (int(W * j / (8 * fps)), H - 4), color, 6); cv2.line(fr, (W // 2, H - 12), (W // 2, H), (255, 255, 255), 2)
        w.write(fr)
w.release(); cap.release()
final = Path(f"output/tdeed_demo/tiros_{MATCH.split(' ')[0]}_{MODEL}.mp4"); final.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_raw), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(final)], check=True)
print(f"{final}: {len(preds)} detecciones ({len(hit_pred)} aciertos, {len(preds) - len(hit_pred)} falsas alarmas), {len(gt) - len(hit_gt)} tiros no detectados de {len(gt)}")
