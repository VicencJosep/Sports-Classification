"""Cola desatendida (12-oct-2026): cuando termine el re-análisis del Espanyol con COCO a 1280 px (match_1280),
lanza el re-análisis con el detector de balón ajustado (match_snt), espera a que acabe y escribe las comparaciones
de cobertura (640 -> 1280 COCO -> SoccerNet) en output/ball_experiments/. Idempotente y reanudable.
    docker exec -d -w /app Sports-Classification /opt/venv/bin/python scripts/ball_detector/wait_and_run_snt.py
"""
import json, subprocess, time, urllib.request
from pathlib import Path

OUT = Path("output/ball_experiments")
ids = json.load(open(OUT / "espanyol_1280.json"))
STATUS = OUT / "STATUS"


def status(msg: str) -> None:
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def run_state(match: str) -> str:
    p = Path(f"backend/storage/outputs/{match}/full/run.json")
    return json.load(open(p))["status"] if p.exists() else "none"


def wait_run(match: str, limit_s: int) -> str:
    t = 0
    while run_state(match) not in ("completed", "error") and t < limit_s:
        time.sleep(120)
        t += 120
    return run_state(match)


def start(match: str) -> bool:
    req = urllib.request.Request(f"http://127.0.0.1:8100/api/matches/{match}/analysis", data=b'{"mode":"full-match"}',
                                 headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status(f"POST analysis {match} -> HTTP {r.status}")
                return True
        except Exception as exc:  # backend caído, 409 por otro análisis en curso, etc.
            status(f"POST analysis {match} intento {attempt}/3 falló: {exc}")
            time.sleep(60)
    return False


def compare(a: str, b: str, out: str) -> None:
    cmd = ["/opt/venv/bin/python", "scripts/ball_detector/distance_vs_ball.py", "--possession", a, "--compare", b,
           "--fps", "30", "--out", str(OUT / out)]
    ok = subprocess.run(cmd, capture_output=True, text=True)
    status(f"comparación {out}: {'hecha' if ok.returncode == 0 else 'FALLÓ ' + ok.stderr[-300:]}")


m1280, msnt = ids["match_1280"], ids["match_snt"]
v_old = f"backend/storage/videos/{ids['video_old']}/source.possession.jsonl"
v_1280 = f"backend/storage/videos/{ids['video_1280']}/source.possession.jsonl"
v_snt = f"backend/storage/videos/{ids['video_snt']}/source.possession.jsonl"

status(f"=== cola SoccerNet: espero a {m1280} (COCO 1280), luego lanzo {msnt} (ajustado) ===")
st = wait_run(m1280, 9 * 3600)
status(f"re-análisis COCO 1280 ({m1280}): {st}")
if Path(v_1280).exists() and not (OUT / "espanyol_640_vs_1280coco.txt").exists():
    compare(v_old, v_1280, "espanyol_640_vs_1280coco.txt")

if run_state(msnt) == "none":
    if not Path("models/ball/yolo11s_snt_ball.pt").exists():
        status("no hay models/ball/yolo11s_snt_ball.pt: no lanzo el re-análisis ajustado")
        raise SystemExit(1)
    if not start(msnt):
        status("no se pudo lanzar el re-análisis ajustado: lo lanza a mano POST /api/matches/%s/analysis" % msnt)
        raise SystemExit(1)
st = wait_run(msnt, 9 * 3600)
status(f"re-análisis ajustado ({msnt}): {st}")
if Path(v_snt).exists():
    compare(v_old, v_snt, "espanyol_640_vs_soccernet.txt")
    if Path(v_1280).exists():
        compare(v_1280, v_snt, "espanyol_1280coco_vs_soccernet.txt")
with open(OUT / "REPORT.md", "a") as f:
    f.write("\n\n## 5. Espanyol: cobertura de balón/poseedor por detector (A frente a B)\n")
    for name in ("espanyol_640_vs_1280coco.txt", "espanyol_640_vs_soccernet.txt", "espanyol_1280coco_vs_soccernet.txt"):
        p = OUT / name
        if p.exists():
            f.write(f"\n### {name}\n```\n{p.read_text()}```\n")
status("=== FIN cola SoccerNet ===")
