"""
Evalua el head CaptionClassification (clasificador softmax entrenado sobre 23
clases fijas del dataset de entrenamiento), no el VideoCaption (retrieval),
contra los clips ya etiquetados en clips_analisis/<categoria>/*.mp4.
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
from data.video_caption import keywords_list

CLIPS_ROOT = "/app/world_cup/clips_analisis"
# proxy mas cercano dentro de las 23 clases con las que se entreno el head
CATEGORY_TO_TRAINED_CLASS = {
    "corner": "corner",
    "foul": "foul with no card",
    "goal": "goal",
    "kick-off": "start of half game",
    "shots_off_target": "shot off target",
    # sin equivalente exacto en las 23 clases de entrenamiento (ver nota mas abajo)
    "shots_on_target": "saved by goal-keeper",
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
    idx_iter = iter(idxs)
    target = next(idx_iter, None)
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
            target = next(idx_iter, None)
        pos += 1
    cap.release()
    if not frames:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return torch.stack(frames, dim=0)


def main():
    cfg = yaml_to_dict("./configs/pretrain.yaml")
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultiTaskingModel(config=cfg)
    model.load_checkpoint("./pretrained_models/SoccerMaster", logger=None, load_heads=True)
    model.eval().to(device)

    num_frames = cfg["NUM_FRAMES"]
    image_size = cfg.get("AUG_MAX_SIZE", 512)

    random.seed(SEED)
    results = []
    categories = list(CATEGORY_TO_TRAINED_CLASS.keys())

    for cat in categories:
        target_class = CATEGORY_TO_TRAINED_CLASS[cat]
        target_idx = keywords_list.index(target_class)
        files = sorted(glob.glob(os.path.join(CLIPS_ROOT, cat, "*.mp4")))
        random.shuffle(files)
        files = files[:N_PER_CATEGORY]
        for fp in files:
            clip = sample_clip_frames(fp, num_frames, image_size)
            if clip is None:
                continue
            images = clip.unsqueeze(0).to(device)
            with torch.no_grad():
                # VideoCaption tambien se ejecuta (comparte dataset_name con
                # CaptionClassification) y su head exige un texto no vacio,
                # aunque aqui solo nos interese la salida de CaptionClassification.
                outputs = model(images, dataset_name="VideoCaption", text=["dummy"])
            logits = outputs["CaptionClassification"]["logits"][0]
            probs = torch.softmax(logits, dim=-1)
            top3_idx = torch.topk(probs, 3).indices.cpu().tolist()
            top3 = [(keywords_list[i], float(probs[i])) for i in top3_idx]
            pred_class = keywords_list[top3_idx[0]]
            correct = pred_class == target_class
            results.append({"file": os.path.basename(fp), "true_cat": cat, "target_class": target_class,
                             "pred_class": pred_class, "correct": correct, "top3": top3})
            print(f"[{cat:18s} -> {target_class:20s}] pred={pred_class:20s} {'OK' if correct else 'X '}  top3={top3}")

    total = len(results)
    correct_n = sum(r["correct"] for r in results)
    print(f"\n=== Accuracy global (CaptionClassification, top-1 exacto): {correct_n}/{total} = {correct_n/total:.1%} ===")
    print("(23 clases posibles -> azar ~= 4.3%)\n")
    for cat in categories:
        cat_results = [r for r in results if r["true_cat"] == cat]
        if not cat_results:
            continue
        acc = sum(r["correct"] for r in cat_results) / len(cat_results)
        in_top3 = sum(any(k == r["target_class"] for k, _ in r["top3"]) for r in cat_results) / len(cat_results)
        print(f"  {cat:18s} top1={acc:.1%}  top3={in_top3:.1%}  ({sum(r['correct'] for r in cat_results)}/{len(cat_results)})")

    with open("outputs/eval_classification_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
