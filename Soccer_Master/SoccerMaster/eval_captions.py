"""
Evalua el head VideoCaption de SoccerMaster (retrieval SigLIP-style) contra los
clips ya etiquetados en clips_analisis/<categoria>/*.mp4, para saber de verdad
(no por intuicion) que tan bien distingue corner/foul/goal/etc.
"""
import os
import sys
import glob
import random
import json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.multi_task import MultiTaskingModel
from configs.util import load_super_config, yaml_to_dict

CLIPS_ROOT = "/app/world_cup/clips_analisis"
CATEGORY_TO_CAPTION = {
    "corner": "corner kick",
    "foul": "foul",
    "goal": "goal",
    "kick-off": "kick-off",
    "shots_off_target": "shot off target",
    "shots_on_target": "shot on target",
    "substitution": "substitution",
    "yellow_card": "yellow card",
}
N_PER_CATEGORY = 8
SEED = 0


def sample_clip_frames(path, num_frames, image_size):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = sorted(set(int(round(i * (total - 1) / max(1, num_frames - 1))) for i in range(num_frames)))
    frames = []
    pos = 0
    idx_set_iter = iter(idxs)
    target = next(idx_set_iter, None)
    while target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        if pos == target:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(rgb).float() / 255.0
            t = (t - 0.5) / 0.5
            frames.append(t.permute(2, 0, 1))
            target = next(idx_set_iter, None)
        pos += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.stack(frames, dim=0)  # [T,3,H,W]


def main():
    cfg = yaml_to_dict("./configs/pretrain.yaml")
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiTaskingModel(config=cfg)
    model.load_checkpoint("./pretrained_models/SoccerMaster", logger=None, load_heads=True)
    model.eval().to(device)

    num_frames = cfg["NUM_FRAMES"]
    image_size = cfg.get("AUG_MAX_SIZE", 512)
    captions = list(CATEGORY_TO_CAPTION.values())
    categories = list(CATEGORY_TO_CAPTION.keys())

    random.seed(SEED)
    results = []
    confusion = {c: {c2: 0 for c2 in categories} for c in categories}

    for cat in categories:
        files = sorted(glob.glob(os.path.join(CLIPS_ROOT, cat, "*.mp4")))
        random.shuffle(files)
        files = files[:N_PER_CATEGORY]
        for fp in files:
            clip = sample_clip_frames(fp, num_frames, image_size)
            if clip is None:
                continue
            images = clip.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(images, dataset_name="VideoCaption", text=captions)
            sim = outputs["VideoCaption"]["base_similarity_matrix"][0].cpu().numpy()
            order = sim.argsort()[::-1]
            pred_caption = captions[order[0]]
            pred_cat = categories[captions.index(pred_caption)]
            top3 = [(categories[captions.index(captions[i])], float(sim[i])) for i in order[:3]]
            correct = pred_cat == cat
            confusion[cat][pred_cat] += 1
            results.append({"file": os.path.basename(fp), "true": cat, "pred": pred_cat, "correct": correct, "top3": top3})
            print(f"[{cat:18s}] pred={pred_cat:18s} {'OK' if correct else 'X '}  top3={top3}")

    total = len(results)
    correct_n = sum(r["correct"] for r in results)
    print(f"\n=== Accuracy global: {correct_n}/{total} = {correct_n/total:.1%} ===\n")
    print("Accuracy por categoria:")
    for cat in categories:
        cat_results = [r for r in results if r["true"] == cat]
        if not cat_results:
            continue
        acc = sum(r["correct"] for r in cat_results) / len(cat_results)
        print(f"  {cat:18s} {acc:.1%}  ({sum(r['correct'] for r in cat_results)}/{len(cat_results)})")

    print("\nMatriz de confusion (filas=real, columnas=predicho):")
    header = "".ljust(18) + "".join(c[:6].ljust(8) for c in categories)
    print(header)
    for cat in categories:
        row = cat.ljust(18) + "".join(str(confusion[cat][c2]).ljust(8) for c2 in categories)
        print(row)

    with open("outputs/eval_captions_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
