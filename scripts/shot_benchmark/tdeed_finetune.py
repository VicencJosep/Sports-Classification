"""Afina T-DEED (checkpoint SoccerNetBall_challenge2, doble cabeza SNB 12 + SoccerNet 17) con los clips de nuestra
cámara virtual (datos de tdeed_finetune_data.py). Reutiliza del repo el dataset de clips, el modelo, la pérdida (CE
ponderada + desplazamiento, mixup) y la evaluación (mAP a ±0,5 y ±1 s con NMS, media sobre las 12 clases); sin wandb.
Todos los clips van a la cabeza SNB; la cabeza SoccerNet no recibe gradiente (no hay datos), así que el checkpoint
resultante lo lee tdeed_scan_heads.py tal cual.

Uso (cwd models/tdeed/T-DEED):
    python /app/scripts/shot_benchmark/tdeed_finetune.py NOMBRE --store [--train-split X --val-split Y]   # 1) indexa clips
    python /app/scripts/shot_benchmark/tdeed_finetune.py NOMBRE [--epochs 8 --lr 1e-4 --epoch-frames 200000 --batch-size 2
                                                                 --workers 8 --freeze-backbone --no-mixup --no-eval-first --resume]
Salida: /app/models/tdeed/NOMBRE/checkpoint_best.pt (mejor mAP en validación), last.pt, loss.json.
--resume: si ya hay loss.json, no repite la evaluación inicial y, si hay last.pt, sigue en la época siguiente a la última
guardada (pesos de last.pt, planificador avanzado hasta ahí; el estado de AdamW empieza de cero)."""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
sys.path.insert(0, os.getcwd())
from util.io import load_json  # noqa: E402
from util.dataset import load_classes  # noqa: E402
import util.eval as ueval  # noqa: E402
from util.eval import evaluate  # noqa: E402
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset  # noqa: E402
from model.model import TDEEDModel  # noqa: E402

DATA = Path("/app/datasets/tdeed_pano")
BASE_CKPT = "/app/models/tdeed/SoccerNetBall_challenge2/checkpoint_best.pt"
STRIDE, OVERLAP = 2, 0.9   # los de 'soccernetball' en dataset/datasets.py


class Head1(Dataset):
    """Marca cada clip como del dataset 1 (cabeza SNB) para la pérdida de doble cabeza del repo."""
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, i):
        d = self.ds[i]; d["dataset"] = 1; return d

    def __len__(self):
        return len(self.ds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name"); ap.add_argument("--store", action="store_true", help="solo indexar los clips (store) y salir")
    ap.add_argument("--train-split", default="train"); ap.add_argument("--val-split", default="val")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epoch-frames", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=2); ap.add_argument("--acc-grad", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--no-mixup", action="store_true"); ap.add_argument("--no-eval-first", action="store_true")
    ap.add_argument("--init", default=BASE_CKPT); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continuar desde models/tdeed/NOMBRE/{loss.json,last.pt} si existen")
    ap.add_argument("--infer-batch", type=int, default=2, help="clips por lote en la evaluación (el repo usa 4: ~20 GB; "
                    "con 1 el repo añade una vista volteada, así que mínimo 2 para el mismo cómputo)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    ueval.INFERENCE_BATCH_SIZE = max(a.infer_batch, 2)

    args = argparse.Namespace(**load_json("config/SoccerNetBall/SoccerNetBall_challenge2.json"))
    args.crop_dim = None; args.batch_size = a.batch_size; args.learning_rate = a.lr; args.num_epochs = a.epochs
    args.warm_up_epochs = a.warmup; args.epoch_num_frames = a.epoch_frames; args.mixup = not a.no_mixup
    args.frame_dir = str(DATA / "frames"); args.store_dir = str(DATA / "store"); args.store_mode = "store" if a.store else "load"
    args.num_workers = a.workers
    classes = load_classes("data/soccernetball/class.txt"); sn_classes = load_classes("data/soccernet/class.txt")

    dataset_len = max(args.epoch_num_frames // args.clip_len, a.batch_size)
    kw = dict(stride=STRIDE, overlap=OVERLAP, radi_displacement=args.radi_displacement, dataset="soccernetball")
    train = ActionSpotDataset(classes, str(DATA / "splits" / f"{a.train_split}.json"), args.frame_dir, args.store_dir,
                              args.store_mode, "rgb", args.clip_len, dataset_len, mixup=args.mixup, **kw)
    val = ActionSpotDataset(classes, str(DATA / "splits" / f"{a.val_split}.json"), args.frame_dir, args.store_dir,
                            args.store_mode, "rgb", args.clip_len, max(dataset_len // 4, a.batch_size), mixup=False, **kw)
    print(f"clips indexados: train {train._total_len}, val {val._total_len}", flush=True)
    if a.store:
        print("Hecho; vuelve a lanzar sin --store para entrenar."); return
    val_frames = ActionSpotVideoDataset(classes, str(DATA / "splits" / f"{a.val_split}.json"), args.frame_dir, "rgb",
                                        args.clip_len, overlap_len=0, stride=STRIDE, dataset="soccernetball")

    def worker_init_fn(i):
        random.seed(i + epoch * 100)
    loader = lambda ds: DataLoader(Head1(ds), shuffle=False, batch_size=a.batch_size // a.acc_grad, pin_memory=True,
                                   num_workers=a.workers, prefetch_factor=2, worker_init_fn=worker_init_fn)
    epoch = 0
    train_loader, val_loader = loader(train), loader(val)

    out = Path(f"/app/models/tdeed/{a.name}"); out.mkdir(parents=True, exist_ok=True)
    hist, best, start_epoch, init = [], -1.0, 0, a.init
    if a.resume and (out / "loss.json").exists():
        hist = json.load(open(out / "loss.json"))
        done = [h for h in hist if h["epoch"] >= 0]
        if done and (out / "last.pt").exists():
            start_epoch = max(h["epoch"] for h in done) + 1
            best = max(h["val_mAP"] for h in done)
            init = str(out / "last.pt")
        print(f"reanudando: época inicial {start_epoch}, mejor mAP {100 * best:.2f}, pesos de {init}", flush=True)
    json.dump(vars(a), open(out / "finetune_args.json", "w"), indent=1)

    model = TDEEDModel(args=args)
    model._model.update_pred_head([len(classes) + 1, len(sn_classes) + 1])
    model._num_classes = len(classes) + 1 + len(sn_classes) + 1
    model.load(torch.load(init, map_location="cuda"))
    print(f"checkpoint inicial: {init}", flush=True)
    if a.freeze_backbone:
        for p in model._model._features.parameters():
            p.requires_grad = False
        print("backbone congelado", flush=True)
    optimizer, scaler = model.get_optimizer({"lr": args.learning_rate})
    steps_per_epoch = len(train_loader) // a.acc_grad
    sched = ChainedScheduler([LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warm_up_epochs * steps_per_epoch),
                              CosineAnnealingLR(optimizer, steps_per_epoch * (args.num_epochs - args.warm_up_epochs))])
    for _ in range(start_epoch * steps_per_epoch):   # reanudación: el planificador avanza hasta la época inicial
        sched.step()

    if not a.no_eval_first and not any(h["epoch"] == -1 for h in hist):
        t0 = time.time()
        base = evaluate(model, val_frames, "VAL", classes, printed=True, test=False)
        print(f"[base] mAP val {100 * base:.2f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        hist.append({"epoch": -1, "val_mAP": float(base)})
        json.dump(hist, open(out / "loss.json", "w"), indent=1)
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()
        tr = model.epoch(train_loader, optimizer, scaler, lr_scheduler=sched, acc_grad_iter=a.acc_grad)
        vl = model.epoch(val_loader, acc_grad_iter=a.acc_grad)
        t1 = time.time()
        m = float(evaluate(model, val_frames, "VAL", classes, printed=True, test=False))
        better = m > best; best = max(best, m)
        hist.append({"epoch": epoch, "train": tr, "val": vl, "val_mAP": m, "lr": optimizer.param_groups[0]["lr"]})
        json.dump(hist, open(out / "loss.json", "w"), indent=1)
        torch.save(model.state_dict(), out / "last.pt")
        if better:
            torch.save(model.state_dict(), out / "checkpoint_best.pt")
        print(f"[época {epoch}] pérdida train {tr:.4f} val {vl:.4f} | mAP val {100 * m:.2f} (mejor {100 * best:.2f}"
              f"{', nuevo mejor' if better else ''}) | {(t1 - t0) / 60:.1f} min entreno+val, {(time.time() - t1) / 60:.1f} min mAP"
              f" | VRAM pico {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    print(f"FIN: mejor mAP val {100 * best:.2f} -> {out / 'checkpoint_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
