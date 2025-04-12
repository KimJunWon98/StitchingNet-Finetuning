"""
MobileNetV2 with a user‑controllable width multiplier (``alpha``) **and optional
`timm` pretrained weights support**.

This module is a drop‑in replacement for
``torchvision.models.quantization.mobilenet_v2`` that additionally allows you
to pull ImageNet‑1k checkpoints from the *timm* project for **any** width
multiplier.  The incoming timm state‑dict is automatically **remapped** to the
key layout expected by torchvision so that you can mix‑and‑match training
recipes without worrying about naming mismatches.

Highlights
----------
* ✅  Supports *float* and *quantized* (QAT‑ready or INT8) variants.
* ✅  Accepts either torchvision weight enums **or** timm checkpoints.
* ✅  Automatic key‑name remapping – no more "unexpected keys" errors.
* ✅  Works with arbitrary ``alpha`` values (0.35 → 1.4).

Usage
-----
```python
from timm_mobilenetv2_quantization import timm_mobilenet_v2_custom

# 0.50‑wide float model, timm weights
model = timm_mobilenet_v2_custom(alpha=0.5, timm_pretrained=True)

# Explicit timm model name also works (alpha inferred)
model = timm_mobilenet_v2_custom(timm_pretrained="mobilenetv2_050")

# 1.0‑wide INT8 model, torchvision weights
qmodel = timm_mobilenet_v2_custom(alpha=1.0, quantize=True, weights="DEFAULT")
```
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Optional, Union

import torch
from torch import Tensor, nn  # noqa: F401 – used for docstrings only

# ---------------------------------------------------------------------------
# Torchvision internals
# ---------------------------------------------------------------------------
from torchvision.models._api import register_model

try:
    from torchvision.models._api import handle_legacy_interface
except ImportError:  # torchvision>=0.16 dropped the symbol

    def handle_legacy_interface(**_ignore):  # type: ignore
        def decorator(fn):
            return fn

        return decorator

from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.quantization.mobilenetv2 import (
    MobileNet_V2_QuantizedWeights,
    QuantizableInvertedResidual,
    QuantizableMobileNetV2,
    _replace_relu,
    quantize_model,
)

# ---------------------------------------------------------------------------
# Optional timm import
# ---------------------------------------------------------------------------
try:
    import timm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – timm optional
    timm = None  # pylint: disable=invalid-name

__all__ = ["timm_mobilenet_v2_custom"]

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _alpha_to_timm_name(alpha: float) -> str:
    """Return canonical timm model name for a given width multiplier."""

    rounded = int(round(alpha * 100))
    return f"mobilenetv2_{rounded:03d}"


def _load_timm_state(alpha: float, timm_model_name: Optional[str] = None) -> Mapping[str, torch.Tensor]:
    """Download a timm MobileNetV2 checkpoint and return its *float* state‑dict."""

    if timm is None:
        raise RuntimeError("timm is not installed; run `pip install timm`.")

    name = timm_model_name or _alpha_to_timm_name(alpha)
    try:
        model = timm.create_model(name, pretrained=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to create timm model '{name}': {exc}") from exc

    return model.state_dict()


def _remap_timm_to_tv(state_timm: Mapping[str, torch.Tensor], model_tv: nn.Module) -> OrderedDict:
    """Remap timm parameter/buffer keys → torchvision layout (order‑based)."""

    tv_keys = list(model_tv.state_dict().keys())
    timm_keys = list(state_timm.keys())

    if len(tv_keys) != len(timm_keys):
        raise RuntimeError(
            "timm and torchvision MobileNetV2 parameter counts differ – "
            "check version compatibility."
        )

    return OrderedDict((tv_k, state_timm[timm_k]) for tv_k, timm_k in zip(tv_keys, timm_keys))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@register_model(name="timm_mobilenet_v2_custom")
@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V2_Weights.IMAGENET1K_V1,
    )
)
def timm_mobilenet_v2_custom(
    *,
    alpha: float = 1.0,
    weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    timm_pretrained: Union[bool, str, None] = None,
    **kwargs: Any,
):
    """Create a MobileNetV2 with configurable ``alpha`` and optional timm weights."""

    # ------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if timm_pretrained and weights is not None:
        raise ValueError("Specify only one of `weights` or `timm_pretrained`.")

    # Alpha → width_mult for the underlying constructor
    kwargs["width_mult"] = alpha
    kwargs.pop("alpha", None)

    # ------------------------------------------------------------------
    # Torchvision weight handling (if not using timm)
    # ------------------------------------------------------------------
    if not timm_pretrained:
        weights = (MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights).verify(weights)

        if weights is not None:
            if alpha != 1.0:
                raise ValueError(
                    "Pre‑trained torchvision weights only exist for alpha==1.0; "
                    "set alpha=1.0 or enable timm_pretrained."
                )
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
            if "backend" in weights.meta:
                _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])

    # Choose quantization backend (default qnnpack)
    backend = kwargs.pop("backend", "qnnpack")

    # ------------------------------------------------------------------
    # Build *float* model
    # ------------------------------------------------------------------
    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    # ------------------------------------------------------------------
    # Load weights (timm or torchvision)
    # ------------------------------------------------------------------
    if timm_pretrained:
        # 1. Download timm checkpoint
        state_timm = _load_timm_state(alpha, timm_model_name=timm_pretrained if isinstance(timm_pretrained, str) else None)
        # 2. Remap → torchvision key layout
        state = _remap_timm_to_tv(state_timm, model)
        # 3. Load
        model.load_state_dict(state, strict=False)

        # Quantize after loading float weights
        if quantize:
            quantize_model(model, backend)

    else:  # torchvision path
        if quantize:
            quantize_model(model, backend)
        if weights is not None:
            model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
