"""Comparativa de variantes del VLM sobre el banco de 150 candidatos difíciles
(output/shot_benchmark/vlm_bench/candidates.json: 60 tiros, 60 centros, 30 sin anotar ≈ repeticiones).

Variantes (todas con el mismo clip [t-3, t+5] s):
  frames_json        16 frames (mismo muestreo que ShotVerifier), pregunta "analiza y responde JSON", sin thinking
  frames_think       igual, con thinking activado (chat_template_kwargs) y solo el JSON como respuesta
  video_json         vídeo nativo 224p a 4 fps (~32 frames) en vez de frames sueltos, sin thinking
  video_think        vídeo nativo + thinking
  videoarm           orquestador VideoARM (clasificación de 10 etiquetas; tiro = shot|goal)
  omni_video         Qwen3-Omni-30B (puerto 8002), vídeo 224p + thinking (checkpoint Thinking)

Uso: python vlm_bench.py VARIANTE [--workers 6] [--limit N]     (reanudable; resultados en vlm_bench/VARIANTE.jsonl)
     python vlm_bench.py report
"""
import argparse, base64, json, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

import os
BENCH = Path(os.environ.get("VLM_BENCH_DIR", "output/shot_benchmark/vlm_bench"))
BEFORE_S, AFTER_S = 3.0, 5.0
QWEN9B = dict(url="http://localhost:8010/v1", model="Qwen/Qwen3.5-9B")
OMNI = dict(url="http://localhost:8002/v1", model="cyankiwi/Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit")

ANALYSIS = (
    "This is a football (soccer) broadcast clip of about 8 seconds, in chronological order; "
    "the key moment is around the middle.\n"
    "Analyse it step by step:\n"
    "1. LIVE or REPLAY? Signs of a replay: slow motion, a transition/wipe graphic, a camera angle "
    "different from the main wide broadcast camera, or the same action shown again.\n"
    "2. Does a player strike or head the ball? If so, where does the ball go afterwards: towards the "
    "goal (saved, blocked, into the net, wide or over the bar), across the box to a teammate (a cross), "
    "or somewhere else?\n"
    "3. A SHOT is any attempt to score: the ball is struck or headed towards the goal. It counts as a shot "
    "whether it goes in, is saved by the goalkeeper, is BLOCKED by a defender, hits the post, or goes wide or "
    "over. Crosses, passes, clearances, dribbles and balls going out of play without an attempt are NOT shots.\n"
)
JSON_LINE = 'Finish with a JSON object on its own line: {"replay": true or false, "shot": true or false, "confidence": number from 0.0 to 1.0}'
PROMPT_BRIEF = ANALYSIS + "Write at most three short sentences of analysis. " + JSON_LINE
PROMPT_THINK = ANALYSIS + "Think it through, then answer ONLY with the JSON object. " + JSON_LINE
_JSON_RE = re.compile(r"\{[^{}]*\}")


def load_bench(limit=None):
    rows = json.loads((BENCH / "candidates.json").read_text())
    return rows[:limit] if limit else rows


def frames_content(video, t):
    from core.events import sample_frames_at
    from core.events_vllm import _frame_to_data_url
    # mismo muestreo que ShotVerifier.frame_times: 8 densos en t±1s + 8 de contexto
    start, end = max(0.0, t - BEFORE_S), t + AFTER_S
    dense = np.linspace(t - 1.0, t + 1.0, 8)
    before = np.linspace(start, t - 1.0, 3)[:-1]
    after = np.linspace(t + 1.0, end, 7)[1:]
    frames = sample_frames_at(Path(video), [float(x) for x in (*before, *dense, *after)])
    return [{"type": "image_url", "image_url": {"url": _frame_to_data_url(f)}} for f in frames]


def video_content(video, t, height=224):
    start = max(0.0, t - BEFORE_S)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "clip.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{t + AFTER_S - start:.3f}",
                        "-vf", f"scale=-2:{height}", "-r", "4", "-c:v", "libx264", "-preset", "ultrafast", "-an",
                        "-loglevel", "error", str(out)], check=True, timeout=60)
        b64 = base64.b64encode(out.read_bytes()).decode()
    return [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64}}]


def ask(server, content, prompt, thinking, max_tokens):
    import requests
    payload = {
        "model": server["model"],
        "messages": [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
        "temperature": 0.6 if thinking else 0.0,
        "top_p": 0.95,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "media_io_kwargs": {"video": {"fps": 4.0}},
        "mm_processor_kwargs": {"fps": 4.0},
    }
    r = requests.post(server["url"] + "/chat/completions", json=payload, timeout=900)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return (ch["message"].get("content") or ""), ch.get("finish_reason"), r.json().get("usage", {}).get("completion_tokens")


def parse(text):
    objs = _JSON_RE.findall(text)
    for o in reversed(objs):
        try:
            d = json.loads(o)
            if "shot" in d:
                return {"shot": bool(d["shot"]), "replay": bool(d.get("replay", False)),
                        "confidence": float(d.get("confidence", 0.5))}
        except Exception:
            continue
    return None


def run_vlm(row, variant):
    video, t = row["video"], row["t_s"]
    server = OMNI if variant == "omni_video" else QWEN9B
    thinking = variant.endswith("think") or variant == "omni_video"
    content = video_content(video, t) if variant.startswith(("video", "omni")) else frames_content(video, t)
    prompt = PROMPT_THINK if thinking else PROMPT_BRIEF
    text, finish, ntok = ask(server, content, prompt, thinking, 4096 if thinking else 400)
    p = parse(text)
    return {"raw": text[-600:], "finish": finish, "tokens": ntok, "parsed": p,
            "score": (p["confidence"] if p["shot"] else 1 - p["confidence"]) if p else 0.5,
            "shot": p["shot"] if p else None, "replay": p["replay"] if p else None}


_videoarm = None
def run_videoarm(row, _variant):
    global _videoarm
    if _videoarm is None:
        from core.events_vllm import VideoARMEventClassifier
        _videoarm = VideoARMEventClassifier()
    t = row["t_s"]
    pred = _videoarm.predict_window(row["video"], max(0.0, t - BEFORE_S), t + AFTER_S)
    score = pred.probabilities.get("shot", 0.0) + pred.probabilities.get("goal", 0.0)
    return {"raw": pred.label, "parsed": pred.probabilities, "score": score,
            "shot": pred.label in ("shot", "goal"), "replay": None}


def run(variant, workers, limit):
    rows = load_bench(limit)
    out = BENCH / f"{variant}.jsonl"
    done = set()
    if out.exists():
        done = {(json.loads(l)["match"], json.loads(l)["t_s"]) for l in out.read_text().splitlines() if l.strip()}
    pending = [r for r in rows if (r["match"], r["t_s"]) not in done]
    print(f"{variant}: {len(pending)} pendientes de {len(rows)}", flush=True)
    fn = run_videoarm if variant == "videoarm" else run_vlm
    t0 = time.perf_counter()

    def work(r):
        try:
            return r, fn(r, variant), None
        except Exception as e:
            return r, None, f"{type(e).__name__}: {str(e)[:300]}"

    n = 0
    with open(out, "a") as f, ThreadPoolExecutor(workers) as ex:
        for r, res, err in ex.map(work, pending):
            n += 1
            if err:
                print(f"  ERROR {r['match'][13:30]} {r['t_s']}: {err}", flush=True)
                continue
            f.write(json.dumps({**r, **res}) + "\n"); f.flush()
            if n % 10 == 0:
                print(f"  {n}/{len(pending)} ({(time.perf_counter() - t0) / n:.1f} s/cand)", flush=True)
    print(f"FIN {variant} ({(time.perf_counter() - t0) / max(n, 1):.1f} s/cand)", flush=True)


def report():
    rows = load_bench()
    base = {(r["match"], r["t_s"]): r for r in rows}
    variants = {"v1 (sí/no, 16 frames)": {k: {**r, "score": r["vlm_v1"], "shot": r["vlm_v1"] > 0.5, "replay": None}
                                          for k, r in base.items()}}
    for p in sorted(BENCH.glob("*.jsonl")):
        variants[p.stem] = {(d["match"], d["t_s"]): d for d in map(json.loads, p.read_text().splitlines()) if d.get("shot") is not None}
    print(f"{'variante':<24}{'n':>4}  {'dice tiro en TIROS':>18}  {'en CENTROS':>11}  {'en SIN ANOTAR':>14}  "
          f"{'AUC tiro/centro':>15}  {'tiro&!repe: tiros':>18}  {'centros':>8}  {'sin anotar':>11}")
    for name, res in variants.items():
        by = {k: [r for r in res.values() if r["kind"] == k] for k in ("shot", "cross", "unannotated")}
        if not by["shot"]:
            continue
        say = {k: np.mean([r["shot"] for r in v]) if v else np.nan for k, v in by.items()}
        pos, neg = [r["score"] for r in by["shot"]], [r["score"] for r in by["cross"]]
        auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]) if pos and neg else np.nan
        stat = {k: np.mean([r["shot"] and not r.get("replay") for r in v]) if v and v[0].get("replay") is not None else np.nan
                for k, v in by.items()}
        print(f"{name:<24}{len(res):>4}  {100 * say['shot']:>17.0f}%  {100 * say['cross']:>10.0f}%  {100 * say['unannotated']:>13.0f}%  "
              f"{auc:>15.2f}  {100 * stat['shot']:>17.0f}%  {100 * stat['cross']:>7.0f}%  {100 * stat['unannotated']:>10.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    report() if a.variant == "report" else run(a.variant, a.workers, a.limit)
