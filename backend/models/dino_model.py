"""
models/dino_model.py
--------------------
Owns the DINOv2 model — loaded exactly once at process startup.

Every other module that needs the model imports MODEL, TRANSFORM, and DEVICE
from here. This eliminates the double-load that existed when both main.py and
imageHash.py each called torch.hub.load() independently.

Public names
------------
MODEL    : torch.nn.Module  — the DINOv2 ViT-B/14 in eval mode
TRANSFORM: torchvision.transforms.Compose  — preprocessing pipeline
DEVICE   : str  — "cuda" or "cpu"
"""

import logging

import torch
import torchvision.transforms as T

from config import (
    DINO_MODEL_NAME,
    DINO_INPUT_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

logger = logging.getLogger(__name__)


def _build_transform() -> T.Compose:
    """
    Preprocessing pipeline applied to every image before DINOv2 inference.

    Images are converted to grayscale (3-channel) so the model is insensitive
    to colour-mode differences (light/dark theme screenshots, colour filters,
    etc.).  The normalisation constants are the standard ImageNet values used
    by all DINOv2 torchvision checkpoints.
    """
    return T.Compose([
        T.Resize((DINO_INPUT_SIZE, DINO_INPUT_SIZE)),
        # Grayscale mode: model sees luminance only, ignores hue/saturation.
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def _load_model(model_name: str, device: str) -> torch.nn.Module:
    """Download (or use cached) DINOv2 checkpoint and move to device."""
    logger.info("Loading DINOv2 model '%s' on device '%s'...", model_name, device)
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()
    logger.info("DINOv2 model loaded successfully.")
    return model


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once when this module is first imported.
# ---------------------------------------------------------------------------

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TRANSFORM: T.Compose = _build_transform()
MODEL: torch.nn.Module = _load_model(DINO_MODEL_NAME, DEVICE)
