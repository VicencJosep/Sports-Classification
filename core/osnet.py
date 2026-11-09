"""
core/osnet.py
=============
OSNet x1.0 (Zhou et al., ICCV 2019) como extractor de apariencia para el ReID
del tracker (``core.tracking.BotSortTracker``).

Copia recortada de ``torchreid/models/osnet.py`` (Kaiyang Zhou, licencia MIT):
solo la arquitectura ``osnet_x1_0`` en modo inferencia, sin descarga de pesos
ImageNet ni cabeza de clasificación. El ``torchreid`` instalado en el
contenedor no se puede importar (su ``__init__`` arrastra ``skimage``) y el
clon de Deep-EIoU bajo ``models/`` no es un sitio del que importar en
producción; de ahí la copia.

Pesos por defecto: ``models/reid/osnet_x1_0_sports.pth.tar-60``, el OSNet
"sports" que distribuyen Deep-EIoU y GTATrack (entrenado en SportsMOT). Sus
repos no tienen licencia: vale para benchmark interno, revisar antes de
producto (docs/comparativa_benchmark.md §5.4). Cualquier checkpoint de
torchreid con la misma arquitectura (p. ej. ``osnet_x1_0`` Market-1501, MIT)
carga igual.

``OSNetEmbedder`` cumple el protocolo ``core.classification.Embedder``
(crops BGR de tamaño variable -> ``(N, 512)`` float32), así que también
sirve como embedder de equipos (``core.classification``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_OSNET_WEIGHTS_PATH: Path = Path(__file__).resolve().parents[1] / "models" / "reid" / "osnet_x1_0_sports.pth.tar-60"
OSNET_IMAGE_SIZE: tuple[int, int] = (256, 128)  # (alto, ancho), el de torchreid
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Capas (torchreid/models/osnet.py, MIT)
# --------------------------------------------------------------------------- #

class ConvLayer(nn.Module):
    """conv + bn + relu."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 conv + bn + relu."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 conv + bn (sin no linealidad)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 (lineal) + 3x3 en profundidad (no lineal)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Puertas por canal condicionadas a la entrada (sigmoide, reducción 16)."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    """Bloque omni-escala: cuatro ramas de 1 a 4 convoluciones ligeras, fusionadas por puertas."""

    def __init__(self, in_channels: int, out_channels: int, bottleneck_reduction: int = 4) -> None:
        super().__init__()
        mid = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid),
                                    LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return torch.relu(x3 + identity)


class OSNet(nn.Module):
    """Omni-Scale Network en modo inferencia: ``forward`` devuelve el vector de 512-d (``fc``)."""

    def __init__(self, layers: Sequence[int], channels: Sequence[int], feature_dim: int = 512) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # mismos nombres que torchreid (fc.0 Linear, fc.1 BatchNorm1d) para que el state_dict case
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))

    @staticmethod
    def _make_layer(layer: int, in_channels: int, out_channels: int, reduce_spatial_size: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_channels, out_channels)]
        for _ in range(1, layer):
            layers.append(OSBlock(out_channels, out_channels))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        v = self.global_avgpool(x).flatten(1)
        return self.fc(v)


def osnet_x1_0() -> OSNet:
    """OSNet de anchura 1.0 (la de ``sports_model.pth.tar-60`` y de los pesos públicos de torchreid)."""
    return OSNet(layers=[2, 2, 2], channels=[64, 256, 384, 512])


def load_osnet_weights(model: nn.Module, path: Path | str) -> tuple[int, int]:
    """Carga un checkpoint de torchreid (``{'state_dict': ...}`` o un state_dict a secas).

    Como ``torchreid.utils.load_pretrained_weights``: quita el prefijo ``module.`` y descarta
    las capas que no casan en nombre o forma (la cabeza ``classifier``). Devuelve
    (capas cargadas, capas del checkpoint).
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # checkpoints antiguos con objetos en el pickle (optimizador, scheduler)
        logger.warning("OSNet: %s no carga con weights_only=True; se reintenta sin esa comprobación", path.name)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    own = model.state_dict()
    matched = {}
    for k, v in state.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k in own and own[k].shape == v.shape:
            matched[k] = v
    own.update(matched)
    model.load_state_dict(own)
    if not matched:
        raise RuntimeError(f"OSNet: ningún tensor de {path} casa con la arquitectura")
    logger.info("OSNet: %d/%d tensores cargados de %s", len(matched), len(state), path.name)
    return len(matched), len(state)


# --------------------------------------------------------------------------- #
#  Embedder (protocolo core.classification.Embedder)
# --------------------------------------------------------------------------- #

class OSNetEmbedder:
    """Extractor de apariencia OSNet x1.0: crops BGR -> ``(N, 512)`` float32.

    Preprocesado de ``torchreid.utils.FeatureExtractor``: RGB, redimensionado a
    256x128 y normalización ImageNet. El modelo se entrenó en RGB; el demo de
    Deep-EIoU le pasaba BGR y la variante corregida (``rgb``) medía mejor
    (docs/comparativa_benchmark.md §7.4), así que aquí se convierte.
    Un crop vacío produce un vector de ceros (sin norma): BoT-SORT lo trata como
    "sin apariencia" y asocia solo por movimiento.

    El lote sube a la GPU en uint8 y se normaliza allí: en numpy, normalizar
    22 crops costaba ~220 ms por frame con la CPU compartida (medido el
    6-nov-2026) frente a 0,4 ms en la GPU. Con ``fp16`` (autocast, por defecto
    en CUDA) el forward de 22 crops baja de 21 a 9 ms.
    """

    def __init__(
        self,
        weights_path: Path | str = DEFAULT_OSNET_WEIGHTS_PATH,
        device: Optional[str] = None,
        image_size: tuple[int, int] = OSNET_IMAGE_SIZE,
        batch_size: int = 128,
        fp16: Optional[bool] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.batch_size = batch_size
        self.fp16 = (self.device.startswith("cuda")) if fp16 is None else fp16
        self.model = osnet_x1_0()
        load_osnet_weights(self.model, weights_path)
        self.model.eval().to(self.device)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def embedding_dim(self) -> int:
        return self.model.feature_dim

    def _preprocess(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        """Crops BGR -> tensor (N, 3, H, W) float normalizado, en ``self.device``."""
        h, w = self.image_size
        batch = np.zeros((len(crops), h, w, 3), dtype=np.uint8)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)
            batch[i] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        return x.sub_(self._mean).div_(self._std)

    @torch.inference_mode()
    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        empty = np.array([c is None or c.size == 0 for c in crops])
        x = self._preprocess(crops)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16 and x.is_cuda):
            feats = torch.cat([self.model(x[i:i + self.batch_size]) for i in range(0, len(x), self.batch_size)])
        out = feats.float().cpu().numpy()
        out[empty] = 0.0
        return out
