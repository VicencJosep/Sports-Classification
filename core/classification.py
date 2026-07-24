"""
core/classification.py
========================
Asignación de equipo (2 clusters por apariencia) sobre un vídeo ya trackeado
por ``core/tracking.py``, portando el algoritmo de
``Soccer_Master/sn-gamestate/sn_gamestate/team/tracklet_team_clustering_api.py``
(``TrackletTeamClustering``) fuera de tracklab/Hydra/pandas.

Pipeline: tracks.jsonl + vídeo -> recorte por bbox (solo role=="player") ->
embedding de apariencia (PRTReId, 256-d) -> media por tracker_id -> K-Means
(k=2) -> teams.json.

El extractor de embeddings por defecto es un puerto de
``Soccer_Master/sn-gamestate/sn_gamestate/reid/prtreid_api.py`` (``PRTReId``,
BPBReID + backbone HRNet32). El paquete real (``prtreid``/``bpbreid``) NO está
vendorizado en este repo, se instala vía git sin versión fijada (ver
requirements.txt y README § "Problemas conocidos") — es la MISMA categoría de
riesgo que ya rompió ``transformers``/``torch`` sin pin en este proyecto.

El clustering en sí (``TeamClusterer``) no depende de ``prtreid`` en absoluto:
se puede probar end-to-end con ``StubEmbedder`` (color medio del crop) sin GPU
ni esa dependencia instalada.

Explícitamente fuera de alcance de este módulo (ver plan / README):
  - Etiquetado left/right (``TrackletTeamSideLabeling``) — necesita
    ``field_xy``/homografía (``core/homography.py``, dependencia externa
    PnLCalib sin verificar end-to-end todavía).
  - OCR de dorsales (la otra mitad del roadmap de este fichero).
  - Voto de mayoría por tracklet (``MajorityVoteTrackletFilter``) — este
    proyecto ya obtiene ``role`` estable por detección desde el propio
    detector (``TrackRecord.role``), no hace falta una capa de voto aparte
    para el filtro "player" que usa el clustering de equipos.

Uso rápido (standalone, desde la raíz del proyecto):
    python -m core.classification --source partido.mp4 --tracks tracks.jsonl --output teams.json
    python -m core.classification --source partido.mp4 --tracks tracks.jsonl --output teams.json --embedder stub
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
_DEFAULT_REID_DIR: Path = _PROJECT_ROOT / "models" / "reid"
DEFAULT_REID_WEIGHTS_PATH: Path = _DEFAULT_REID_DIR / "prtreid-soccernet-baseline.pth.tar"
DEFAULT_REID_BACKBONE_PATH: Path = _DEFAULT_REID_DIR / "hrnetv2_w32_imagenet_pretrained.pth"
_REID_WEIGHTS_URL: str = "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
_REID_WEIGHTS_MD5: str = "9633825232bc89f23a94522c5561650e"
_REID_BACKBONE_URL: str = "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1"
_REID_BACKBONE_MD5: str = "58ea12b0420aa3adaa2f74114c9f9721"

DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 128)  # (height, width), fijo por el checkpoint publicado
DEFAULT_BATCH_SIZE: int = 64                       # = prtreid.yaml test.batch_size
DEFAULT_N_CLUSTERS: int = 2
DEFAULT_RANDOM_STATE: int = 0

_PLAYER_ROLE: str = "player"


# --------------------------------------------------------------------------- #
#  Interfaz de embedding (inyección de dependencias, mismo patrón que
#  core/tracking.py::Detector)
# --------------------------------------------------------------------------- #

class Embedder(Protocol):
    """Contrato que debe cumplir cualquier extractor de apariencia inyectado
    en ``assign_teams``.

    ``embed_batch`` recibe 1..N crops BGR (``np.ndarray`` HxWx3 uint8, tamaños
    variables, SIN redimensionar — el propio embedder decide su resize y
    normalización internos, igual que ``prtreid.tools.feature_extractor.
    FeatureExtractor`` hace para PRTReId) y devuelve un array ``(N,
    embedding_dim)`` float32, en el mismo orden que la entrada.
    """

    @property
    def embedding_dim(self) -> int: ...

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...


def _stub_prtreid_random_occlusion() -> None:
    """Sustituye ``prtreid.data.data_augmentation.random_occlusion`` por un
    módulo vacío en ``sys.modules`` ANTES de que ``import prtreid`` intente
    cargar el fichero real.

    Ese fichero (aumento de datos de ENTRENAMIENTO, que no usamos — nosotros
    solo llamamos a ``FeatureExtractor`` para inferencia) hace
    ``from albumentations import (DualTransform, functional)``. ``functional``
    como submódulo importable directamente desde la raíz del paquete es una
    API de versiones de `albumentations` muy anteriores a las reorganizaciones
    internas por tipo de efecto (blur/crops/geometric/...); ninguna versión de
    `albumentations` razonablemente reciente la expone así, y bajar a una
    versión antigua arriesga romper OTRA cosa en la cadena de dependencias.
    Confirmado en ejecución real (17 jul 2026):
    ``ImportError: cannot import name 'functional' from 'albumentations'``.

    ``RandomOcclusion`` sí necesita existir como nombre importable (otros
    módulos de `prtreid` hacen ``from prtreid.data.data_augmentation import
    RandomOcclusion`` incondicionalmente al cargar el paquete), así que el stub
    define una clase dummy en vez de dejar el módulo completamente vacío — no
    se instancia nunca en el camino de inferencia que usa este fichero.

    Vive en nuestro propio código (no parchea `site-packages`), así que
    sobrevive a cualquier reconstrucción de la imagen Docker.
    """
    import sys
    import types

    name = "prtreid.data.data_augmentation.random_occlusion"
    if name in sys.modules:
        return

    stub = types.ModuleType(name)

    class RandomOcclusion:  # dummy: nunca se instancia en nuestro camino de inferencia
        def __init__(self, *args, **kwargs) -> None:
            pass

    stub.RandomOcclusion = RandomOcclusion
    sys.modules[name] = stub


def _register_prtreid_soccernet_dataset() -> None:
    """Registra la clase ``SoccerNet`` que YA trae ``prtreid`` (``prtreid.data.
    datasets.image.soccernet.SoccerNet``) bajo el nombre ``"SoccerNet"`` —
    el mismo que usan ``cfg.data.sources``/``targets`` en ``_build_cfg``.

    ``prtreid.scripts.main.build_config`` llama internamente a
    ``compute_parts_num_and_names(cfg)``, que necesita ALGÚN dataset
    registrado bajo ``cfg.data.sources[0]`` para poder consultar
    ``get_masks_config(cfg.model.bpbreid.masks.dir)`` — aunque el resultado no
    dependa realmente de qué dataset sea: con
    ``masks.dir="pose_on_img_crops"`` (valor de ``prtreid.yaml``), tanto la
    ``SoccerNet`` real de `prtreid` como la ``ReidDataset`` propia de
    `sn-gamestate` devuelven ``None`` (ninguna reconoce esa clave en su
    ``masks_dirs``), así que da igual cuál se registre.

    `prtreid` importa esta clase pero no la registra por defecto — solo
    `sn-gamestate` la registra, con su propia `ReidDataset` (fuertemente
    acoplada a `tracklab`: necesita un `TrackingDataset` completo con splits
    de train/query/gallery, inviable de replicar para pura inferencia). Usar
    la `SoccerNet` real de `prtreid` en su lugar es más simple y NO menos
    fiel: es la clase que el propio paquete define para este dataset.

    Solo se registra la CLASE (``register_image_dataset`` no la instancia),
    así que nunca dispara el escaneo de disco de un dataset que no tenemos.
    Confirmado en ejecución real (17 jul 2026):
    ``ValueError: Invalid dataset name. Received "SoccerNet", but expected to
    be one of [...]`` sin este registro.
    """
    from prtreid.data.datasets import SoccerNet, get_image_dataset, register_image_dataset

    try:
        get_image_dataset("SoccerNet")
    except ValueError:
        register_image_dataset("SoccerNet", SoccerNet)


_TORCH_LOAD_PATCHED: bool = False


def _patch_torch_load_weights_only_false() -> None:
    """``prtreid`` (``load_checkpoint`` dentro de ``build_config``,
    ``load_pretrained_weights`` dentro de ``FeatureExtractor``) llama a
    ``torch.load(fpath, map_location=...)`` sin especificar ``weights_only``
    — desde PyTorch 2.6 el valor por defecto cambió de ``False`` a ``True``, y
    el checkpoint publicado de PRTReId (pickle con objetos
    ``numpy.core.multiarray.scalar`` de una versión de numpy antigua) no pasa
    esa comprobación de seguridad por defecto. Confirmado en ejecución real
    (17 jul 2026): ``_pickle.UnpicklingError: Weights only load failed ...
    Unsupported global: GLOBAL numpy.core.multiarray.scalar``.

    El checkpoint viene de una URL de Zenodo fija, verificada por MD5 antes de
    usarse (ver ``_download_if_missing``) — no es un pickle arbitrario de
    origen desconocido, así que forzar ``weights_only=False`` aquí es un
    riesgo aceptado y acotado (solo afecta a este proceso, mientras se usa
    ``PRTReIdEmbedder``), no una desactivación general de la comprobación de
    seguridad de PyTorch para el resto del proyecto. Idempotente: solo
    parchea `torch.load` una vez por proceso.
    """
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return

    import torch

    _original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_load
    _TORCH_LOAD_PATCHED = True


class PRTReIdEmbedder:
    """Embedder de apariencia (256-d) vía PRTReId (BPBReID + backbone HRNet32).

    Puerto directo de ``prtreid_api.py::PRTReId.preprocess``/``.process`` sin
    tracklab/Hydra/pandas: construye el ``cfg`` (yacs ``CfgNode``) con los
    valores fijos de ``sn-gamestate/sn_gamestate/configs/modules/reid/
    prtreid.yaml`` en vez de resolverlo vía Hydra, y usa solo 3 símbolos del
    paquete externo ``prtreid``: ``FeatureExtractor``, ``build_config``,
    ``extract_test_embeddings``.

    ✅ Verificado en ejecución real (17 jul 2026, ver README § "Problemas
    conocidos" para el detalle de los cinco incidentes que hubo que resolver:
    dos imports rotos de ``prtreid`` sin relación con nuestro código, un
    dataset a registrar a mano, y un ``torch.load`` que había que forzar a
    ``weights_only=False``). El paquete ``prtreid`` sigue sin estar
    vendorizado (se instala vía git sin versión fijada, ver requirements.txt)
    — si tras reconstruir la imagen algo vuelve a fallar al construir el
    modelo, ese README es el primer sitio a revisar.

    Import perezoso: `torch`/`prtreid`/`yacs`/`requests` solo se importan aquí
    dentro, nunca a nivel de módulo — el resto de core/classification.py
    (Embedder, TeamClusterer, StubEmbedder, CLI) sigue siendo importable sin
    `prtreid` instalado, igual que core/tracking.py sigue siendo importable
    sin `ultralytics`.

    Args:
        weights_path / backbone_weights_path: rutas locales de los checkpoints
            (se descargan de Zenodo, con verificación MD5, si faltan).
        device: "cuda" | "cpu" (None -> cuda si está disponible).
        image_size: (height, width) de entrada al modelo — (256, 128), fijo
            por el checkpoint publicado; no cambiar salvo que cambies de
            checkpoint.
    """

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_REID_WEIGHTS_PATH,
        backbone_weights_path: Path | str = DEFAULT_REID_BACKBONE_PATH,
        device: Optional[str] = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        _stub_prtreid_random_occlusion()
        _register_prtreid_soccernet_dataset()
        _patch_torch_load_weights_only_false()
        import torch

        weights_path = Path(weights_path)
        backbone_weights_path = Path(backbone_weights_path)

        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self._torch = torch
        self._feature_extractor = None  # construido perezosamente en el primer embed_batch

        self._download_if_missing(weights_path, _REID_WEIGHTS_URL, _REID_WEIGHTS_MD5)
        self._download_if_missing(backbone_weights_path, _REID_BACKBONE_URL, _REID_BACKBONE_MD5)

        logger.info("Construyendo config de PRTReId (weights=%s) ...", weights_path)
        self.cfg = self._build_cfg(weights_path, backbone_weights_path)

    @property
    def embedding_dim(self) -> int:
        return 256  # cfg.model.bpbreid.dim_reduce_output en prtreid.yaml

    # ------------------------------------------------------------------ #

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        from prtreid.utils.tools import extract_test_embeddings

        if self._feature_extractor is None:
            from prtreid.tools.feature_extractor import FeatureExtractor

            logger.info("Cargando PRTReId (FeatureExtractor) en %s ...", self.device)
            self._feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=self.image_size,
                model=None,
                verbose=False,
            )

        # prtreid/torchreid entrenan en RGB; los crops de core/tracking.py son
        # BGR (OpenCV). Sin verificar empíricamente contra el modelo real
        # (paquete no instalado aquí) — si la similitud imágen-misma-persona
        # sale mal en la verificación manual, es el primer sitio a revisar.
        rgb_crops = [crop[:, :, ::-1] for crop in crops]

        reid_result = self._feature_extractor(rgb_crops, external_parts_masks=None)
        embeddings, _visibility, _masks, _, _role_scores = extract_test_embeddings(
            reid_result, self.cfg.model.bpbreid.test_embeddings
        )
        return embeddings.cpu().detach().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #

    def _build_cfg(self, weights_path: Path, backbone_weights_path: Path):
        """Reconstruye el ``cfg`` yacs que ``prtreid.scripts.main.build_config``
        espera, a partir de los valores literales de ``prtreid.yaml`` (sección
        ``cfg:``). Ver aviso en el docstring de la clase.

        La referencia (prtreid_api.py, vía Hydra) construye este cfg pasando
        por OmegaConf (``CN(OmegaConf.to_container(cfg, resolve=True))``) porque
        Hydra ya le entrega un `DictConfig` con interpolaciones (`${...}`) que
        resolver. Aquí no hace falta: `raw_cfg` es un dict Python plano sin
        interpolaciones, así que `yacs.CfgNode` lo acepta directamente — y
        `omegaconf` ni siquiera es una dependencia real de `prtreid`/
        `torchreid` (es cosa de la capa Hydra de sn-gamestate, que no usamos).
        Confirmado en ejecución real: sin este cambio, falla con
        ``ModuleNotFoundError: No module named 'omegaconf'``."""
        from prtreid.scripts.main import build_config
        from yacs.config import CfgNode as CN

        raw_cfg = {
            "project": {
                "name": "TrackLab", "experiment_name": "", "notes": "", "tags": [],
                "logger": {"use_tensorboard": False, "use_wandb": False},
            },
            "data": {
                "root": "", "type": "image", "sources": ["SoccerNet"], "targets": ["SoccerNet"],
                "height": self.image_size[0], "width": self.image_size[1],
                "combineall": False, "transforms": ["rc", "re"], "save_dir": "", "workers": 4,
            },
            "sampler": {
                "train_sampler": "PrtreidSampler", "train_sampler_t": "PrtreidSampler",
                "num_instances": 4,
            },
            "model": {
                "name": "bpbreid", "pretrained": True, "save_model_flag": False,
                "load_config": True, "load_weights": str(weights_path),
                "bpbreid": {
                    "pooling": "gwap", "normalization": "identity",
                    "mask_filtering_training": False, "mask_filtering_testing": False,
                    "training_binary_visibility_score": True, "testing_binary_visibility_score": True,
                    "last_stride": 1, "learnable_attention_enabled": False,
                    "dim_reduce": "after_pooling", "dim_reduce_output": 256,
                    "backbone": "hrnet32", "test_embeddings": ["globl"],
                    "test_use_target_segmentation": "none", "shared_parts_id_classifier": False,
                    "hrnet_pretrained_path": str(backbone_weights_path.parent),
                    "masks": {"type": "disk", "dir": "pose_on_img_crops", "preprocess": "id"},
                },
            },
            "loss": {
                "name": "part_based",
                "part_based": {
                    "name": "part_averaged_triplet_loss", "ppl": "cl",
                    "weights": {
                        "globl": {"id": 1.0, "tr": 1.0}, "foreg": {"id": 0.0, "tr": 0.0},
                        "conct": {"id": 0.0, "tr": 0.0}, "parts": {"id": 0.0, "tr": 0.0},
                        "pixls": {"ce": 0.0},
                    },
                },
            },
            "train": {"batch_size": 32, "max_epoch": 20},
            "test": {
                "evaluate": False, "detailed_ranking": False, "start_eval": 40,
                "batch_size": DEFAULT_BATCH_SIZE, "batch_size_pairwise_dist_matrix": 5000,
                "normalize_feature": True, "dist_metric": "euclidean",
                "visrank": False, "visrank_per_body_part": False,
                "vis_embedding_projection": False, "vis_feature_maps": False,
                "visrank_topk": 10, "visrank_count": 4, "visrank_q_idx_list": [],
                "part_based": {"dist_combine_strat": "mean"},
            },
        }

        yacs_cfg = CN(raw_cfg)
        yacs_cfg.use_gpu = self.device.startswith("cuda")
        return build_config(config=yacs_cfg)

    @staticmethod
    def _download_if_missing(path: Path, url: str, md5: str) -> None:
        if path.exists():
            return
        import hashlib

        import requests

        logger.info("Descargando %s -> %s ...", url, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".part")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

        digest = hashlib.md5(tmp_path.read_bytes()).hexdigest()
        if digest != md5:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"MD5 no coincide para {url}: esperado {md5}, obtenido {digest}")
        tmp_path.rename(path)


class StubEmbedder:
    """Embedder falso (color medio BGR del crop, 3-d) para probar
    ``TeamClusterer``/``assign_teams`` sin GPU ni ``prtreid`` instalado (ver
    README § verificación). Expuesto vía ``--embedder stub``, no es solo para
    pruebas desechables: es un modo de funcionamiento real y barato, aunque
    de calidad muy inferior a PRTReId (separa por color medio de la camiseta,
    no por apariencia aprendida)."""

    @property
    def embedding_dim(self) -> int:
        return 3

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack(
            [crop.reshape(-1, 3).mean(axis=0) for crop in crops]
        ).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Clustering de equipos (sin I/O, puerto directo de TrackletTeamClustering)
# --------------------------------------------------------------------------- #

class TeamClusterer:
    """Agrega embeddings de apariencia por tracklet y separa en ``n_clusters``
    equipos con K-Means sobre esas medias — puerto directo de
    ``TrackletTeamClustering.process()`` (Soccer_Master/sn-gamestate) sin
    pandas/tracklab.

    A diferencia del original (que guarda cada embedding individual en un
    DataFrame y hace ``groupby(...).mean()`` al final), aquí se acumula de
    forma incremental (suma + contador por ``tracker_id``) — memoria
    ``O(nº tracklets)`` en vez de ``O(nº detecciones)``, resultado
    numéricamente idéntico.

    No hace I/O ni recorta crops: solo agrega vectores ya calculados. Qué
    detecciones son "player" y cómo se extraen sus embeddings vive fuera de
    esta clase (``assign_teams``/``Embedder``), a propósito, para poder
    testear esta lógica con vectores sintéticos sin vídeo ni GPU.

    Args:
        n_clusters: nº de equipos (2, igual que el original).
        random_state: semilla de K-Means (0, igual que el original — determina
            de forma estable, para las mismas entradas, qué mitad se llama "0").
    """

    def __init__(
        self, n_clusters: int = DEFAULT_N_CLUSTERS, random_state: int = DEFAULT_RANDOM_STATE
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._sum: dict[int, np.ndarray] = {}
        self._count: dict[int, int] = {}

    def add_embedding(self, tracker_id: int, embedding: np.ndarray) -> None:
        """Acumula una detección más (ya embebida) de este ``tracker_id``."""
        embedding = np.asarray(embedding, dtype=np.float64)
        if tracker_id in self._sum:
            self._sum[tracker_id] += embedding
        else:
            self._sum[tracker_id] = embedding.copy()
        self._count[tracker_id] = self._count.get(tracker_id, 0) + 1

    def fit(self) -> dict[int, int]:
        """Cierra la acumulación y devuelve ``{tracker_id: team_cluster}``.

        Replica los casos borde de ``TrackletTeamClustering.process()``
        exactamente:
          - 0 tracklets acumulados -> ``{}`` (el original pone
            ``team_cluster=NaN`` en todas las filas; aquí, sin "filas" fuera
            de este dict, un dict vacío es el equivalente correcto).
          - 1 tracklet -> ``{ese_id: 0}`` (hardcodeado, igual que el original).
          - >=2 tracklets -> ``KMeans(n_clusters, random_state).fit(medias)
            .labels_``, SIN normalizar los embeddings antes (igual que el
            original).
        Extensión propia (no existe en el original, que fija ``n_clusters=2``
        a fuego): si hay menos tracklets que ``n_clusters``, asigna un cluster
        por tracklet en vez de fallar (KMeans exige ``n_samples >= n_clusters``).
        """
        if not self._sum:
            return {}

        tracker_ids = list(self._sum.keys())
        if len(tracker_ids) == 1:
            return {tracker_ids[0]: 0}

        means = np.vstack([self._sum[tid] / self._count[tid] for tid in tracker_ids])

        if len(tracker_ids) < self.n_clusters:
            logger.warning(
                "Solo %d tracklets de jugador para %d clusters -- asignando un cluster por tracklet",
                len(tracker_ids), self.n_clusters,
            )
            return {tid: i for i, tid in enumerate(tracker_ids)}

        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(means).labels_
        return {tid: int(label) for tid, label in zip(tracker_ids, labels)}


# --------------------------------------------------------------------------- #
#  Registros serializables (telemetría)
# --------------------------------------------------------------------------- #

@dataclass
class TeamAssignment:
    tracker_id: int
    team_cluster: int  # 0 o 1


@dataclass
class ExcludedTracklet:
    tracker_id: int
    reason: str  # "never_player" | "no_valid_crops"


@dataclass
class TeamClusteringResult:
    source_video: str
    tracks_jsonl: str
    embedder: str            # "prtreid" | "stub"
    embedding_dim: int
    n_clusters: int
    random_state: int
    teams: list[TeamAssignment]
    excluded: list[ExcludedTracklet]
    n_frames_processed: int
    n_crops_embedded: int
    n_crops_skipped_degenerate: int

    def to_telemetry(self) -> dict:
        """Registro serializable para el JSON de salida (recursa listas de
        dataclasses anidadas: teams/excluded)."""
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Orquestación
# --------------------------------------------------------------------------- #

def _iter_tracks_jsonl(path: Path) -> Iterator[dict]:
    """Lee un JSONL de tracking línea a línea (mismo patrón que
    core/homography.py y core/possession.py)."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _clamp_bbox(
    xyxy: Sequence[float], frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(int(round(x1)), frame_w))
    y1 = max(0, min(int(round(y1)), frame_h))
    x2 = max(0, min(int(round(x2)), frame_w))
    y2 = max(0, min(int(round(y2)), frame_h))
    return x1, y1, x2, y2


def assign_teams(
    source_video: Path | str,
    tracks_jsonl_path: Path | str,
    embedder: Embedder,
    embedder_name: Optional[str] = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TeamClusteringResult:
    """Orquesta el pipeline completo de asignación de equipos sobre un vídeo
    ya trackeado (``tracks.jsonl`` de ``core/tracking.py``).

    1. Lee ``tracks_jsonl_path`` entero en memoria (``frame_idx -> record``,
       mismo patrón que ``core/homography.py::process()``).
    2. Recorre el vídeo SECUENCIALMENTE con ``cv2.VideoCapture`` (sin seek —
       no fiable frame a frame con muchos encoders con B-frames). Para frames
       sin ninguna detección ``role=="player"`` usa ``cap.grab()`` (avanza sin
       decodificar, barato) en vez de ``cap.read()``.
    3. Por cada frame decodificado, por cada track ``role=="player"``: clampa
       ``xyxy`` a los límites del frame; descarta (cuenta, no aborta) si el
       resultado tiene anchura o altura <= 0.
    4. Batchea los crops válidos; al llenar ``batch_size`` (o al terminar el
       vídeo) llama ``embedder.embed_batch`` una vez y descarga el resultado
       en ``TeamClusterer.add_embedding`` — nunca se retienen embeddings por
       detección más allá de ese flush.
    5. Al agotar el vídeo, ``TeamClusterer.fit()`` una única vez.

    Lanza ``FileNotFoundError`` si ``source_video``/``tracks_jsonl_path`` no
    existen. NO lanza si hay 0 tracklets de jugador — el resultado tendrá
    ``teams=[]`` y todo ``tracker_id`` visto en ``excluded``.
    """
    source_video = Path(source_video)
    tracks_jsonl_path = Path(tracks_jsonl_path)
    if not source_video.exists():
        raise FileNotFoundError(f"Vídeo no encontrado: {source_video}")
    if not tracks_jsonl_path.exists():
        raise FileNotFoundError(f"tracks.jsonl no encontrado: {tracks_jsonl_path}")

    records_by_frame: dict[int, dict] = {
        rec["frame_idx"]: rec for rec in _iter_tracks_jsonl(tracks_jsonl_path)
    }

    all_ids: set[int] = set()
    ever_player_ids: set[int] = set()
    for rec in records_by_frame.values():
        for t in rec.get("tracks", []):
            all_ids.add(t["tracker_id"])
            if t.get("role") == _PLAYER_ROLE:
                ever_player_ids.add(t["tracker_id"])

    clusterer = TeamClusterer(n_clusters=n_clusters, random_state=random_state)

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {source_video}")

    n_frames_processed = 0
    n_crops_embedded = 0
    n_crops_skipped_degenerate = 0
    batch_crops: list[np.ndarray] = []
    batch_tracker_ids: list[int] = []

    def _flush() -> None:
        nonlocal n_crops_embedded
        if not batch_crops:
            return
        embeddings = embedder.embed_batch(batch_crops)
        for tid, emb in zip(batch_tracker_ids, embeddings):
            clusterer.add_embedding(tid, emb)
        n_crops_embedded += len(batch_crops)
        batch_crops.clear()
        batch_tracker_ids.clear()

    try:
        frame_idx = 0
        while True:
            rec = records_by_frame.get(frame_idx)
            has_player = rec is not None and any(
                t.get("role") == _PLAYER_ROLE for t in rec.get("tracks", [])
            )

            if not has_player:
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ok, frame = cap.read()
            if not ok:
                break
            n_frames_processed += 1
            frame_h, frame_w = frame.shape[:2]

            for t in rec.get("tracks", []):
                if t.get("role") != _PLAYER_ROLE:
                    continue
                x1, y1, x2, y2 = _clamp_bbox(t["xyxy"], frame_w, frame_h)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    n_crops_skipped_degenerate += 1
                    logger.debug(
                        "Crop degenerado para tracker_id=%s en frame %d", t["tracker_id"], frame_idx
                    )
                    continue
                batch_crops.append(frame[y1:y2, x1:x2])
                batch_tracker_ids.append(t["tracker_id"])
                if len(batch_crops) >= batch_size:
                    _flush()

            frame_idx += 1

        _flush()
    finally:
        cap.release()

    team_by_id = clusterer.fit()

    no_valid_crops_ids = ever_player_ids - set(team_by_id.keys())
    never_player_ids = all_ids - ever_player_ids
    excluded = [
        ExcludedTracklet(tracker_id=tid, reason="never_player") for tid in sorted(never_player_ids)
    ] + [
        ExcludedTracklet(tracker_id=tid, reason="no_valid_crops") for tid in sorted(no_valid_crops_ids)
    ]

    return TeamClusteringResult(
        source_video=str(source_video),
        tracks_jsonl=str(tracks_jsonl_path),
        embedder=embedder_name or type(embedder).__name__,
        embedding_dim=embedder.embedding_dim,
        n_clusters=n_clusters,
        random_state=random_state,
        teams=[TeamAssignment(tracker_id=tid, team_cluster=team) for tid, team in team_by_id.items()],
        excluded=excluded,
        n_frames_processed=n_frames_processed,
        n_crops_embedded=n_crops_embedded,
        n_crops_skipped_degenerate=n_crops_skipped_degenerate,
    )


# --------------------------------------------------------------------------- #
#  Ejecución standalone
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asignación de equipos (2 clusters) sobre un vídeo ya trackeado (PRTReId + K-Means)"
    )
    parser.add_argument("--source", required=True, type=Path, help="Vídeo .mp4 (el mismo usado por core/tracking.py)")
    parser.add_argument("--tracks", required=True, type=Path, help="tracks.jsonl de core/tracking.py")
    parser.add_argument("--output", required=True, type=Path, help="Salida: teams.json")
    parser.add_argument(
        "--annotated-output", default=None, type=Path,
        help="Si se indica, escribe además un .mp4 coloreado por equipo (rojo/azul; "
             "gris para quien no tenga equipo asignado) -- ver utils/visualizers.py",
    )
    parser.add_argument("--embedder", choices=["prtreid", "stub"], default="prtreid")
    parser.add_argument("--reid-weights", type=Path, default=DEFAULT_REID_WEIGHTS_PATH)
    parser.add_argument("--reid-backbone-weights", type=Path, default=DEFAULT_REID_BACKBONE_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.source.exists():
        parser.error(f"No existe: {args.source}")
    if not args.tracks.exists():
        parser.error(f"No existe: {args.tracks}")

    embedder: Embedder
    if args.embedder == "prtreid":
        embedder = PRTReIdEmbedder(
            weights_path=args.reid_weights,
            backbone_weights_path=args.reid_backbone_weights,
            device=args.device,
        )
    else:
        embedder = StubEmbedder()

    result = assign_teams(
        source_video=args.source,
        tracks_jsonl_path=args.tracks,
        embedder=embedder,
        embedder_name=args.embedder,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        batch_size=args.batch_size,
    )

    with open(args.output, "w") as f:
        json.dump(result.to_telemetry(), f, indent=2)

    logger.info(
        "Equipos asignados: %d tracklets (%d excluidos) -- %d frames procesados, "
        "%d crops embebidos (%d degenerados descartados)",
        len(result.teams), len(result.excluded),
        result.n_frames_processed, result.n_crops_embedded, result.n_crops_skipped_degenerate,
    )
    logger.info("Resultado escrito en %s", args.output)

    if args.annotated_output:
        from utils.visualizers import render_annotated_video  # import perezoso: solo si se pide --annotated-output

        team_by_tracker_id = {t.tracker_id: t.team_cluster for t in result.teams}
        logger.info("Renderizando vídeo anotado por equipo en %s ...", args.annotated_output)
        render_annotated_video(
            source_video=args.source,
            tracks_jsonl_path=args.tracks,
            output_path=args.annotated_output,
            team_by_tracker_id=team_by_tracker_id,
        )
        logger.info("Vídeo anotado escrito en %s", args.annotated_output)


if __name__ == "__main__":
    main()
