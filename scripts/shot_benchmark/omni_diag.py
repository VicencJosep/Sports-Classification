"""Diagnóstico del Omni: 5 tiros + 5 centros del banco limpio, guardando el RAZONAMIENTO (reasoning_content,
que vlm_bench.py no guardaba) y probando dos resoluciones de vídeo (224p, la del banco, y 448p)."""
import json, os, sys, requests
from pathlib import Path
sys.path.insert(0, "/app/output/shot_benchmark/experiments")
os.environ["VLM_BENCH_DIR"] = "/app/output/shot_benchmark/vlm_bench"
import vlm_bench as vb

rows = json.loads(Path("/app/output/shot_benchmark/vlm_bench/candidates.json").read_text())
clean = [r for r in rows if any(k in r["match"] for k in ("Middlesbrough", "Stoke", "Reading"))]
sel = [r for r in clean if r["kind"] == "shot"][:5] + [r for r in clean if r["kind"] == "cross"][:5]
out = open("/app/output/shot_benchmark/vlm_clean/omni_diag.jsonl", "w")
for height in (224, 448):
    for r in sel:
        try:
            content = vb.video_content(r["video"], r["t_s"], height=height)
            payload = {"model": vb.OMNI["model"], "messages": [{"role": "user", "content": content + [{"type": "text", "text": vb.PROMPT_THINK}]}],
                       "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True},
                       "media_io_kwargs": {"video": {"fps": 4.0}}, "mm_processor_kwargs": {"fps": 4.0}}
            resp = requests.post(vb.OMNI["url"] + "/chat/completions", json=payload, timeout=900)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            rec = {"kind": r["kind"], "match": r["match"], "t_s": r["t_s"], "height": height, "answer": msg.get("content"),
                   "reasoning": msg.get("reasoning_content") or msg.get("reasoning"), "error": None}
        except Exception as e:
            rec = {"kind": r["kind"], "match": r["match"], "t_s": r["t_s"], "height": height, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(height, r["kind"], (rec.get("answer") or rec["error"] or "")[-60:].replace("\n", " "), flush=True)
