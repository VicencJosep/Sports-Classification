"""Datos para afinar T-DEED con nuestra cámara virtual, en el formato que espera el repo (dataset 'soccernetball'):
frames JPG a 796x448 y 25 fps por vídeo (ffmpeg), Labels-ball.json por vídeo a partir de bas/, y los json de split.
Cada MITAD renderizada es un "vídeo" (<match>_<half>_<tag>). T-DEED solo lee los eventos con gameTime de la mitad 1 en
SNB, así que se escriben todos como "1 - mm:ss"; position = ms desde el inicio de la mitad (igual que en bas/).

Uso (dentro del contenedor, cwd /app):
    python scripts/shot_benchmark/tdeed_finetune_data.py --train 118575 118576 118577 118578 128058 --val 117093 132877
                                                        [--tag vl1] [--jobs 4] [--split-name NOMBRE]
Salida: datasets/tdeed_pano/frames/<video>/frame<i>.jpg (+ fps.txt, DONE), datasets/tdeed_pano/labels/<video>/Labels-ball.json,
datasets/tdeed_pano/splits/{train,val}.json (o <split-name>.json) y models/tdeed/T-DEED/data/soccernetball/labels_path.txt
apuntando a /app/datasets/tdeed_pano/labels (del original se guarda labels_path.txt.orig). Reanudable: los vídeos ya
extraídos (DONE con el nº de frames) se saltan."""
import argparse, json, shutil, subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2

ROOT = Path("datasets/tdeed_pano")
VIEWS = Path("datasets/soccertrack-v2/views")
BAS = Path("datasets/soccertrack-v2/bas")
TDEED = Path("models/tdeed/T-DEED")
HALVES = (("1st", "1"), ("2nd", "2"))


def video_name(m, h, tag):
    return f"{m}_{h}_{tag}"


def extract(m, h, tag):
    src = VIEWS / m / f"{h}_vcam796{tag}.mp4"
    out = ROOT / "frames" / video_name(m, h, tag)
    n_src = int(cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FRAME_COUNT))
    done = out / "DONE"
    if done.exists() and abs(int(done.read_text()) - n_src) <= 2:
        return video_name(m, h, tag), int(done.read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-start_number", "0", "-q:v", "2",
                        str(out / "frame%d.jpg")])
    n = len(list(out.glob("frame*.jpg")))
    if r.returncode != 0 or abs(n - n_src) > 2:
        raise RuntimeError(f"{src}: ffmpeg {r.returncode}, {n} frames extraídos de {n_src}")
    (out / "fps.txt").write_text("25.0")
    done.write_text(str(n))
    print(f"  {video_name(m, h, tag)}: {n} frames", flush=True)
    return video_name(m, h, tag), n


def write_labels(m, h, hh, tag, classes):
    acts = json.load(open(BAS / m / f"{m}_12_class_events.json"))["actions"]
    seen, ann, skipped = set(), [], Counter()
    for a in sorted(acts, key=lambda a: int(a["position"])):
        if not a["gameTime"].startswith(f"{hh} -"):
            continue
        if a["label"] not in classes:
            skipped[a["label"]] += 1; continue
        key = (int(a["position"]), a["label"])
        if key in seen:
            skipped["duplicado"] += 1; continue
        seen.add(key)
        t = int(a["position"]) / 1000
        ann.append({"gameTime": f"1 - {int(t // 60):02d}:{int(t % 60):02d}", "label": a["label"],
                    "position": str(int(a["position"])), "team": a.get("team", ""), "visibility": "visible"})
    d = ROOT / "labels" / video_name(m, h, tag); d.mkdir(parents=True, exist_ok=True)
    json.dump({"UrlLocal": video_name(m, h, tag), "annotations": ann}, open(d / "Labels-ball.json", "w"))
    return Counter(a["label"] for a in ann), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="*", default=[]); ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--tag", default="vl1"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--split-name", default=None, help="nombre del json de split en vez de train/val (p. ej. prueba)")
    a = ap.parse_args()
    classes = [c for c in (TDEED / "data/soccernetball/class.txt").read_text().splitlines() if c.strip()]
    lp = TDEED / "data/soccernetball/labels_path.txt"
    if not lp.with_suffix(".txt.orig").exists():
        shutil.copy(lp, lp.with_suffix(".txt.orig"))
    lp.write_text("/app/datasets/tdeed_pano/labels\n")
    (ROOT / "splits").mkdir(parents=True, exist_ok=True)
    for split, matches in (("train", a.train), ("val", a.val)):
        if not matches:
            continue
        jobs = [(m, h) for m in matches for h, _ in HALVES]
        with ThreadPoolExecutor(a.jobs) as ex:
            frames = dict(ex.map(lambda j: extract(j[0], j[1], a.tag), jobs))
        total = Counter(); skipped = Counter()
        for m, (h, hh) in ((m, hh) for m in matches for hh in HALVES):
            c, s = write_labels(m, h, hh, a.tag, classes); total += c; skipped += s
        name = a.split_name or split
        entries = [{"video": video_name(m, h, a.tag), "num_frames": frames[video_name(m, h, a.tag)]} for m in matches for h, _ in HALVES]
        json.dump(entries, open(ROOT / "splits" / f"{name}.json", "w"), indent=1)
        print(f"{name}: {len(entries)} vídeos, {sum(e['num_frames'] for e in entries)} frames, {sum(total.values())} eventos "
              f"{dict(total.most_common())}" + (f" | descartados {dict(skipped)}" if skipped else ""), flush=True)


if __name__ == "__main__":
    main()
