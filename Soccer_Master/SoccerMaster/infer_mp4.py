"""
Inferencia de SoccerMaster sobre un video de futbol en mp4.

Este script NO usa el pipeline "Data Pipeline" (tracklab + SAM2 + Qwen2.5-VL) descrito
en el README del repo, porque a fecha de hoy ese pipeline no se puede ejecutar: al
repositorio le falta `sn_gamestate/configs/dataset/soccernet_gs.yaml` y el paquete
`tracklab/wrappers/datasets/` (incluida la clase `ExternalVideo` que permitiria pasar
un mp4 directamente a tracklab). Todo lo que este archivo usa (`models/multi_task.py`,
`models/soccer_master.py`, `models/deformable_detr`, `data/soccernet_gsr_detection.py`,
`configs/pretrain.yaml`) SI existe en el repo y SI se puede cargar con los checkpoints
publicados en https://huggingface.co/xleprime/SoccerMaster.

Que hace:
  1. Lee el mp4 con OpenCV y lo trocea en clips consecutivos de NUM_FRAMES (30 por
     defecto, fijado por el checkpoint) frames, redimensionados a 512x512.
  2. Por cada clip corre el head "SoccerNetGSR_Detection" (deteccion de personas +
     rol [player/goalkeeper/referee/other/ball] + numero de dorsal) y dibuja las
     cajas sobre los frames, generando un mp4 anotado.
  3. Por cada clip corre el head "VideoCaption" (SigLIP-style retrieval) contra una
     lista de frases candidatas (tipo "gol", "tarjeta amarilla", ...) y anota en un
     JSONL cual es la frase mas afin a ese clip.

Uso (desde dentro de codes/SoccerMaster, con el entorno conda activado):
    python infer_mp4.py --video /ruta/al/partido.mp4 --output-dir outputs/mi_video

Ver `--help` para todas las opciones.
"""
import os
import sys
import json
import argparse

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.multi_task import MultiTaskingModel
from models.deformable_detr.deformable_detr import PostProcess
from configs.util import load_super_config, yaml_to_dict
from data.soccernet_gsr_detection import role_mapping, jn_mapping

ROLE_ID_TO_NAME = {v: k for k, v in role_mapping.items() if k is not None}
JN_ID_TO_NUMBER = {v: k for k, v in jn_mapping.items() if k is not None}

ROLE_COLORS = {
    "player": (60, 200, 60),
    "goalkeeper": (60, 60, 230),
    "referee": (0, 215, 255),
    "other": (180, 180, 180),
    "ball": (255, 255, 255),
}

DEFAULT_CAPTIONS = [
    "goal",
    "shot on target",
    "shot off target",
    "corner kick",
    "foul",
    "yellow card",
    "red card",
    "substitution",
    "offside",
    "throw-in",
    "free kick",
    "penalty kick",
    "kick-off",
    "start of half game",
    "end of half game",
    "ball possession",
    "normal play, no notable event",
]


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia de SoccerMaster sobre un mp4")
    p.add_argument("--video", required=True, help="Ruta al video .mp4 de entrada")
    p.add_argument("--config", default="./configs/pretrain.yaml", help="Config yaml del modelo")
    p.add_argument("--checkpoint-dir", default="./pretrained_models/SoccerMaster",
                   help="Carpeta con backbone.pt, SoccerNetGSR_Detection.pt, VideoCaption.pt, ...")
    p.add_argument("--output-dir", default="./outputs/infer_mp4", help="Carpeta de salida")
    p.add_argument("--stride", type=int, default=1,
                   help="Procesar 1 de cada N frames del video original (usa >1 en videos largos para ir mas rapido)")
    p.add_argument("--max-frames", type=int, default=-1,
                   help="Limite de frames originales a leer del video (-1 = todo el video). Util para pruebas rapidas")
    p.add_argument("--score-threshold", type=float, default=0.5, help="Umbral de confianza para las detecciones")
    p.add_argument("--captions-file", default=None,
                   help="Fichero .txt con una frase candidata de comentario/evento por linea (si no, usa una lista por defecto)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(),
                   help="Usar autocast fp16 en GPU (mas rapido, activado por defecto si hay CUDA)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--draw-jersey", action="store_true", default=True)
    return p.parse_args()


def load_model(config_path, checkpoint_dir, device):
    cfg = yaml_to_dict(config_path)
    cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))

    model = MultiTaskingModel(config=cfg)
    model.load_checkpoint(checkpoint_dir, logger=None, load_heads=True)
    model.eval()
    model.to(device)
    return model, cfg


def preprocess_frame(frame_bgr, size):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = (tensor - 0.5) / 0.5  # AUG_MEAN/AUG_STD = 0.5 (siglip)
    tensor = tensor.permute(2, 0, 1)  # C,H,W
    return tensor, resized


def run_detection(model, images, num_frames, image_size, score_threshold, device):
    """images: tensor [1, T, 3, H, W]. Devuelve una lista de longitud T con las
    detecciones (boxes/scores/roles/jersey) de cada frame del clip."""
    with torch.no_grad():
        outputs = model(images, dataset_name="SoccerNetGSR_Detection", metas=None)
    det = outputs["SoccerNetGSR_Detection"]

    pred_logits = det["pred_logits"]  # [1, T, Q, C]
    pred_boxes = det["pred_boxes"]    # [1, T, Q, 4]
    bs, T = pred_logits.shape[:2]
    flat = {
        "pred_logits": pred_logits.reshape(bs * T, *pred_logits.shape[2:]),
        "pred_boxes": pred_boxes.reshape(bs * T, *pred_boxes.shape[2:]),
    }
    for key in ("pred_roles", "pred_jn_holistic", "pred_digit_head", "pred_digit_tail"):
        if key in det:
            flat[key] = det[key].reshape(bs * T, *det[key].shape[2:])

    target_sizes = torch.tensor([[image_size, image_size]] * (bs * T), device=device)
    results = PostProcess()(flat, target_sizes)  # lista de T dicts

    per_frame = []
    for res in results:
        keep = res["scores"] >= score_threshold
        entry = {
            "boxes": res["boxes"][keep].cpu().numpy().tolist(),
            "scores": res["scores"][keep].cpu().numpy().tolist(),
        }
        if "roles" in res:
            entry["roles"] = [ROLE_ID_TO_NAME.get(int(r), "unknown") for r in res["roles"][keep].cpu().numpy()]
        if "jersey" in res:
            entry["jersey"] = [JN_ID_TO_NUMBER.get(int(j), None) for j in res["jersey"][keep].cpu().numpy()]
        per_frame.append(entry)
    return per_frame


def run_caption_retrieval(model, images, captions):
    with torch.no_grad():
        outputs = model(images, dataset_name="VideoCaption", text=captions)
    vc = outputs["VideoCaption"]
    sim = vc["base_similarity_matrix"]  # [1, N]
    if sim is None:
        return None
    sim = sim[0].cpu().numpy()
    order = np.argsort(-sim)
    return [(captions[i], float(sim[i])) for i in order]


def draw_detections(frame_bgr, det_entry):
    frame = frame_bgr.copy()
    boxes = det_entry["boxes"]
    scores = det_entry["scores"]
    roles = det_entry.get("roles", [None] * len(boxes))
    jerseys = det_entry.get("jersey", [None] * len(boxes))
    for box, score, role, jersey in zip(boxes, scores, roles, jerseys):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = ROLE_COLORS.get(role, (0, 200, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{role or 'person'} {score:.2f}"
        if jersey is not None:
            label += f" #{jersey}"
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.captions_file:
        with open(args.captions_file) as f:
            captions = [line.strip() for line in f if line.strip()]
    else:
        captions = DEFAULT_CAPTIONS

    print(f"Cargando modelo desde {args.checkpoint_dir} ...")
    model, cfg = load_model(args.config, args.checkpoint_dir, args.device)
    num_frames = cfg["NUM_FRAMES"]
    image_size = cfg.get("AUG_MAX_SIZE", 512)
    print(f"Modelo cargado. NUM_FRAMES(clip)={num_frames}, image_size={image_size}, device={args.device}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = max(1.0, src_fps / max(1, args.stride))

    video_writer = cv2.VideoWriter(
        os.path.join(args.output_dir, "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (image_size, image_size),
    )
    jsonl_path = os.path.join(args.output_dir, "detections.jsonl")
    captions_path = os.path.join(args.output_dir, "captions.jsonl")

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (args.fp16 and args.device == "cuda") else None

    frame_idx_src = 0
    frame_idx_out = 0
    chunk_tensors = []
    chunk_resized = []

    def flush_chunk():
        nonlocal chunk_tensors, chunk_resized, frame_idx_out
        if not chunk_tensors:
            return
        n_real = len(chunk_tensors)
        tensors = list(chunk_tensors)
        resized = list(chunk_resized)
        while len(tensors) < num_frames:
            tensors.append(tensors[-1])
            resized.append(resized[-1])

        images = torch.stack(tensors, dim=0).unsqueeze(0).to(args.device)  # [1,T,3,H,W]

        if autocast_ctx is not None:
            with autocast_ctx:
                per_frame_dets = run_detection(model, images, num_frames, image_size, args.score_threshold, args.device)
                ranked_captions = run_caption_retrieval(model, images, captions)
        else:
            per_frame_dets = run_detection(model, images, num_frames, image_size, args.score_threshold, args.device)
            ranked_captions = run_caption_retrieval(model, images, captions)

        with open(jsonl_path, "a") as fdet, open(captions_path, "a") as fcap:
            for i in range(n_real):
                det = per_frame_dets[i]
                fdet.write(json.dumps({"frame": frame_idx_out, **det}) + "\n")
                annotated = draw_detections(resized[i], det)
                video_writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                frame_idx_out += 1
            fcap.write(json.dumps({
                "chunk_start_frame": frame_idx_out - n_real,
                "chunk_end_frame": frame_idx_out - 1,
                "top_caption": ranked_captions[0][0] if ranked_captions else None,
                "ranked": ranked_captions,
            }) + "\n")

        print(f"[chunk] frames {frame_idx_out - n_real}-{frame_idx_out - 1} -> "
              f"top caption: {ranked_captions[0][0] if ranked_captions else 'n/a'}")

        chunk_tensors = []
        chunk_resized = []

    while True:
        if args.max_frames > 0 and frame_idx_src >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx_src % args.stride == 0:
            tensor, resized = preprocess_frame(frame, image_size)
            chunk_tensors.append(tensor)
            chunk_resized.append(resized)
            if len(chunk_tensors) == num_frames:
                flush_chunk()
        frame_idx_src += 1

    flush_chunk()  # ultimo chunk parcial (se rellena repitiendo el ultimo frame)

    cap.release()
    video_writer.release()
    print(f"\nListo. Salidas en: {args.output_dir}")
    print(f"  - annotated.mp4    (video con cajas + rol + dorsal)")
    print(f"  - detections.jsonl (una linea por frame con las detecciones)")
    print(f"  - captions.jsonl   (una linea por clip de {num_frames} frames con el evento mas afin)")


if __name__ == "__main__":
    main()
