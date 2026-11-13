"""Informe del experimento de RoI de AdaSpot (docs/comparativa_benchmark/7.14_adaspot_roi_gt.md): el AdaSpot original
(RoI del selector de saliencia) frente a las variantes de ``adaspot_scores.py --roi``, y cuánto acierta ese selector.

No usa GPU: lee lo que dejaron ``eval_bas.py``, ``eval_prf.py`` y ``eval_f1_pcbas.py`` y las cachés de puntuaciones, que
guardan por fila el centro de la RoI anotada (``roi_gt``) y el de la saliencia (``roi_saliency``). Escribe
``output/benchmark/adaspot_gt_roi/tabla.md`` y ``resumen.json``; lo que aún no está puntuado sale como «pendiente».

  1. Métricas: línea base frente a RoI GT, sus dos variantes de sensibilidad (centrada en los pies; saliencia
     corregida para contener al ejecutor) y los controles (recorte central y pseudo-GT del detector).
  2. Acierto del selector: en qué fracción de los frames y eventos con anotación ésta cae dentro de su recorte.
  3. Recall de los eventos que el selector tenía dentro («vistos») y fuera («ciegos»), con cada RoI.
  4. AP por clase.

Uso (contenedor):  python -m scripts.benchmark.adaspot_roi_report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.benchmark.adaspot_gt_roi import ROI
from scripts.benchmark.eval_bas import (LABELS_ROOT, OUT_ROOT, OUT_ROOT_PCBAS, OUT_ROOT_ST2, PCBAS_CLASSES, PCBAS_ROOT,
                                        ST2_ROOT, tdeed_eval_classes)

OUT = OUT_ROOT.parent / "adaspot_gt_roi"
W, H = 796, 448
ROW_MS, TOL_MS = 80.0, 500.0  # una fila de la caché = 2 frames a 25 fps; tolerancia del emparejamiento (como mAP@1)
BAS10 = [c for c in tdeed_eval_classes() if c not in ("FREE KICK", "GOAL")]
# clave, título, raíz de resultados, subcarpeta, raíz de etiquetas, media de AP que se informa, variante con coordenadas
DATASETS = [
    ("efl", "SN-BAS test (Stoke y Reading)", OUT_ROOT, "test", LABELS_ROOT, "mAP_10cls", "_detroi"),
    ("pcbas", "FOOTPASS val (3 partidos)", OUT_ROOT_PCBAS, "val_merged", OUT_ROOT_PCBAS / "labels", "mAP_pcbas8", "_gtroi"),
    ("st2vl1", "SoccerTrack v2 eval, cámara de nuestro detector (vcam796vl1)", OUT_ROOT_ST2, "vcam796vl1",
     OUT_ROOT_ST2 / "labels", "mAP_10cls", "_gtroi"),
    ("st2", "SoccerTrack v2 eval, cámara del balón anotado (vcam796)", OUT_ROOT_ST2, "vcam796",
     OUT_ROOT_ST2 / "labels", "mAP_10cls", "_gtroi"),
]
VARIANTS = {"": "Línea base (saliencia)", "_gtroi": "RoI GT", "_gtfeetroi": "RoI GT en los pies", "_fixroi": "Saliencia corregida",
            "_ctrroi": "Recorte central", "_detroi": "Pseudo-GT (detector)"}
# Variantes que no se puntúan en un dataset: sin anotación (SN-BAS), ya a la altura de los pies (SoccerTrack v2), etc.
NOT_RUN = {"_gtroi": {"efl": "no existe"}, "_gtfeetroi": {"efl": "—", "st2vl1": "—", "st2": "—"}, "_fixroi": {"efl": "—", "st2": "—"},
           "_detroi": {"pcbas": "—", "st2vl1": "—", "st2": "—"}}
MODELS = {"big": "AdaSpot-b", "small": "AdaSpot-s"}
ROWS = {"efl": [("mAP@0,5 s", "at1"), ("mAP@1 s", "at2"), ("F1", "F1")],
        "pcbas": [("Macro-F1@0,15", "macroF1"), ("mAP@0,5 s", "at1"), ("mAP@1 s", "at2"), ("F1", "F1")],
        "st2vl1": [("mAP@0,5 s", "at1"), ("mAP tight", "tight"), ("F1", "F1")],
        "st2": [("mAP@0,5 s", "at1"), ("mAP tight", "tight"), ("F1", "F1")]}


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def num(v, digits: int = 1) -> str:
    return "pendiente" if v is None else f"{v:.{digits}f}".replace(".", ",")


def metrics(root: Path, model: str, sub: str, map_key: str) -> dict | None:
    s, p, f = (read(root / model / sub / n) for n in ("summary.json", "prf.json", "f1_pcbas.json"))
    if not s or not p:
        return None
    res = {k.split("/")[1]: v for k, v in s["results"].items() if k.startswith("snms")}
    best = p["best"]
    return {"at1": res.get("at1", {}).get(map_key), "at2": res.get("at2", {}).get(map_key),
            "tight": res.get("tight", {}).get(map_key), "per_class": res.get("at1", {}).get("per_class", {}),
            "F1": best["overall"]["F1"], "P": best["overall"]["P"], "R": best["overall"]["R"], "thr": best["threshold"],
            "macroF1": f["macro_f1"] if f else p["at_0.15"]["macro_F1"]}


def cache(key: str, sub: str, game: str, model: str) -> Path:
    if key == "efl":
        return LABELS_ROOT / game / f"720p.tdeed.{model}.npz"
    if key == "pcbas":
        return PCBAS_ROOT / "videos" / f"{game}.tdeed.{model}.npz"
    match, half = game.split("/")
    return ST2_ROOT / "views" / match / f"{half}_{sub}.tdeed.{model}.npz"


def matched(gt_ms: np.ndarray, pred: list[tuple[float, float]]) -> np.ndarray:
    """Como ``eval_prf.match`` (1 a 1, por confianza, ±0,5 s), pero devuelve qué eventos quedaron emparejados."""
    used = np.zeros(len(gt_ms), bool)
    for t, _ in sorted(pred, key=lambda p: -p[1]):
        if not len(gt_ms):
            break
        d = np.abs(gt_ms - t)
        d[used] = np.inf
        i = int(np.argmin(d))
        if d[i] <= TOL_MS:
            used[i] = True
    return used


def selector(key: str, root: Path, sub: str, label_root: Path, size: str, given: str) -> dict | None:
    """Acierto del selector de saliencia frente a la anotación (frames y eventos) y recall de cada variante en los
    eventos que el selector tenía dentro de su recorte, fuera, o sin anotación con la que decidirlo."""
    models = {"base": f"AdaSpot_{size}", "given": f"AdaSpot_{size}{given}", "fix": f"AdaSpot_{size}_fixroi",
              "center": f"AdaSpot_{size}_ctrroi"}
    summary = read(root / models["given"] / sub / "summary.json")
    if not summary:
        return None
    classes = list(PCBAS_CLASSES.values()) if key == "pcbas" else BAS10
    # Mismo umbral para todas (el óptimo de la línea base): son los mismos pesos, y con umbrales distintos el recall de un
    # grupo cambiaría aunque la RoI no lo hiciera.
    base_thr = (read(root / models["base"] / sub / "prf.json") or {}).get("best", {}).get("threshold")
    thr = {k: base_thr for k in models}
    n_frames = n_valid = n_inside = 0
    dist, events = [], []  # events: (clase, estado, emparejado por cada variante)
    for game in summary["games"]:
        path = cache(key, sub, game, models["given"])
        z = np.load(path) if path.exists() else None
        if z is None or "roi_gt" not in z.files or "roi_saliency" not in z.files:
            return None
        n = min(len(z["roi_gt"]), len(z["roi_saliency"]))
        gt, sal = z["roi_gt"][:n].astype(float), z["roi_saliency"][:n].astype(float)
        covered = z["roi_used"][:n].any(1)  # la cola del vídeo no la cubre ningún clip
        valid = np.isfinite(gt).all(1) & covered
        # Recorte real de la saliencia: get_rois trunca el centro a píxel y lo mantiene dentro de la imagen.
        cy = np.clip(np.floor(sal[:, 0] * H), ROI // 2, H - ROI // 2 - 1)
        cx = np.clip(np.floor(sal[:, 1] * W), ROI // 2, W - ROI // 2 - 1)
        with np.errstate(invalid="ignore"):
            dy, dx = gt[:, 0] * H - cy, gt[:, 1] * W - cx
            inside = valid & (np.abs(dy) <= ROI / 2) & (np.abs(dx) <= ROI / 2)
        n_frames, n_valid, n_inside = n_frames + int(covered.sum()), n_valid + int(valid.sum()), n_inside + int(inside.sum())
        dist.append(np.hypot(dy, dx)[valid])
        ann = read(label_root / game / "Labels-ball.json")["annotations"]
        preds = {k: (read(root / m / sub / "snms12" / "preds" / game / "results_spotting.json") or {}).get("predictions")
                 for k, m in models.items()}
        for c in classes:
            t = np.array(sorted(float(a["position"]) for a in ann if a["label"] == c))
            if not len(t):
                continue
            rows = np.minimum(np.round(t / ROW_MS).astype(int), n - 1)
            state = np.where(~valid[rows], "sin dato", np.where(inside[rows], "vista", "ciega"))
            hit = {k: None if pr is None or thr[k] is None else
                   matched(t, [(float(p["position"]), float(p["confidence"])) for p in pr
                               if p["label"] == c and float(p["confidence"]) >= thr[k]]) for k, pr in preds.items()}
            events += [(c, state[i], *(None if hit[k] is None else bool(hit[k][i]) for k in models)) for i in range(len(t))]
    dist = np.concatenate(dist)
    with_data = [e for e in events if e[1] != "sin dato"]
    groups = {}
    for g in ("vista", "ciega", "sin dato"):
        ev = [e for e in events if e[1] == g]
        groups[g] = {"n": len(ev), **{k: (round(100 * float(np.mean([e[2 + j] for e in ev])), 1)
                                          if ev and ev[0][2 + j] is not None else None) for j, k in enumerate(models)}}
    per_class = {c: {"n": sum(e[0] == c for e in with_data),
                     "hit": round(100 * float(np.mean([e[1] == "vista" for e in with_data if e[0] == c])), 1)}
                 for c in classes if any(e[0] == c for e in with_data)}
    return {"frames_with_annotation": round(100 * n_valid / max(n_frames, 1), 1),
            "frame_hit": round(100 * n_inside / max(n_valid, 1), 1), "median_dist_px": round(float(np.median(dist)), 1),
            "events": len(events), "events_with_annotation": round(100 * len(with_data) / max(len(events), 1), 1),
            "event_hit": round(100 * float(np.mean([e[1] == "vista" for e in with_data])), 1) if with_data else None,
            "per_class": per_class, "groups": groups, "thresholds": thr}


def main() -> None:
    summary: dict = {}
    t1 = ["| Dataset | Modelo | Métrica | " + " | ".join(VARIANTS.values()) + " | Δ RoI GT − base |",
          "|---|---|---|" + "---:|" * (len(VARIANTS) + 1)]
    t2 = ["| Dataset | Modelo | Frames con anotación | …dentro del recorte de saliencia | Distancia mediana | "
          "Eventos con anotación | …dentro del recorte en su instante |", "|---|---|---:|---:|---:|---:|---:|"]
    t3 = ["| Dataset | Modelo | Eventos | n | Recall línea base | Recall RoI dada | Recall saliencia corregida | "
          "Recall recorte central |", "|---|---|---|---:|---:|---:|---:|---:|"]
    t4 = []
    for key, title, root, sub, label_root, map_key, given in DATASETS:
        for size, name in MODELS.items():
            m = {v: metrics(root, f"AdaSpot_{size}{v}", sub, map_key) for v in VARIANTS}
            summary[f"{key}/{size}"] = {"metrics": m}
            for label, field in ROWS[key]:
                cells = [NOT_RUN.get(v, {}).get(key) or num(m[v][field] if m[v] else None) for v in VARIANTS]
                ref = "_gtroi" if key != "efl" else "_detroi"
                delta = (f"{m[ref][field] - m[''][field]:+.1f}".replace(".", ",")
                         if m[ref] and m[""] and None not in (m[ref][field], m[""][field]) else "pendiente")
                t1.append(f"| {title} | {name} | {label} | " + " | ".join(cells) + f" | {delta}{' (detector)' if key == 'efl' else ''} |")
            sel = selector(key, root, sub, label_root, size, given)
            summary[f"{key}/{size}"]["selector"] = sel
            if sel:
                t2.append(f"| {title} | {name} | {num(sel['frames_with_annotation'])} % | {num(sel['frame_hit'])} % | "
                          f"{num(sel['median_dist_px'], 0)} px | {num(sel['events_with_annotation'])} % | {num(sel['event_hit'])} % |")
                for g, row in sel["groups"].items():
                    t3.append(f"| {title} | {name} | {g} | {row['n']} | {num(row['base'])} | {num(row['given'])} | "
                              f"{NOT_RUN['_fixroi'].get(key) or num(row['fix'])} | {num(row['center'])} |")
            classes = list(PCBAS_CLASSES.values()) if key == "pcbas" else BAS10
            if any(m.values()):
                t4 += ["", f"**{title}, {name}** (AP a ±0,5 s; entre paréntesis, % de eventos de la clase que el selector "
                       "tenía dentro de su recorte)", "", "| Variante | " + " | ".join(
                           c.title() + (f" ({num(sel['per_class'][c]['hit'], 0)} %)" if sel and c in sel["per_class"] else "")
                           for c in classes) + " |", "|---|" + "---:|" * len(classes)]
                t4 += [f"| {VARIANTS[v]} | " + " | ".join(num(m[v]["per_class"].get(c)) for c in classes) + " |"
                       for v in VARIANTS if m[v]]
    t5 = ["| Modelo | Métrica | Caché original (lotes de 4) | Re-puntuada (lotes de 2) |", "|---|---|---:|---:|"]
    for size, name in MODELS.items():  # mismo modelo y misma RoI: la diferencia es el ruido numérico de la re-puntuación
        a, b = (metrics(OUT_ROOT, f"AdaSpot_{size}{v}", "test", "mAP_10cls") for v in ("", "_salroi"))
        summary[f"efl/{size}"]["batch_check"] = b
        t5 += [f"| {name} | {label} | {num(a[field] if a else None, 2)} | {num(b[field] if b else None, 2)} |"
               for label, field in (("mAP@0,5 s", "at1"), ("mAP@1 s", "at2"), ("F1", "F1"))]
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(["# AdaSpot: RoI del selector de saliencia frente a RoI anotada", "",
                      "Generado por `scripts/benchmark/adaspot_roi_report.py`; método y lectura en "
                      "`docs/comparativa_benchmark/7.14_adaspot_roi_gt.md`. Pesos congelados, misma inferencia y mismo "
                      "post-proceso en todas las columnas; solo cambia dónde cae el recorte de 112x112.", "",
                      "## 1. Métricas", "", *t1, "",
                      "mAP sobre las 10 clases de AdaSpot (8 en FOOTPASS); F1 micro al umbral global que la maximiza; "
                      "Macro-F1@0,15 a ±12 frames y solo por clase (sin equipo ni dorsal).", "",
                      "## 2. Acierto del selector de saliencia", "", *t2, "",
                      "En SN-BAS la «anotación» es el balón de nuestro detector (pseudo-GT), no una etiqueta.", "",
                      "## 3. Recall según si el selector tenía el evento dentro de su recorte", "", *t3, "",
                      "Todas las variantes al umbral óptimo de la línea base. «RoI dada» = RoI GT (pseudo-GT del detector en SN-BAS). «Vista» = el "
                      "ejecutor estaba dentro del recorte de saliencia en el instante del evento; «ciega» = fuera.", "",
                      "## 4. AP por clase", *t4, "",
                      "## 5. Control: ruido de re-puntuar con otro tamaño de lote (SN-BAS, RoI de saliencia)", "", *t5, "",
                      "Las variantes de AdaSpot-b en FOOTPASS y SoccerTrack v2 se puntuaron con lotes de 4, como las cachés "
                      "originales (comparación exacta); el resto, con lotes de 2 para ocupar 4 GB de VRAM en vez de 7.", ""])
    (OUT / "tabla.md").write_text(text)
    (OUT / "resumen.json").write_text(json.dumps(summary, indent=1))
    print(text)


if __name__ == "__main__":
    main()
