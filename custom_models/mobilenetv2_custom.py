# mobilenetv2_custom.py
"""
MobileNetV2 with a user‑controllable width multiplier (``alpha``).

This file provides a drop‑in replacement for the stock
``torchvision.models.quantization.mobilenet_v2`` factory, exposing the *width
multiplier* (a.k.a. ``alpha``) as a dedicated argument while preserving the
original API – including support for quantized inference, pretrained weights
(`alpha == 1.0`), and model registration via
``torchvision.models.quantization.mobilenet_v2_custom``.

Examples
--------
>>> from mobilenetv2_custom import mobilenet_v2_custom
>>> # 0.5× wide floating‑point model
>>> model = mobilenet_v2_custom(alpha=0.5, quantize=False)
>>> # 1.0× wide INT8 model for CPU inference
>>> qmodel = mobilenet_v2_custom(alpha=1.0, quantize=True, weights="DEFAULT")

Parameters
----------
alpha : float, default 1.0
    Width multiplier that uniformly scales the number of channels. Typical
    values are 0.35, 0.5, 0.75, 1.0, 1.3.
    Pretrained checkpoints are only available for ``alpha == 1.0``.
**kwargs : Any
    Additional arguments accepted by the original
    :pyfunc:`torchvision.models.quantization.mobilenet_v2` factory, e.g.
    ``num_classes``, ``backend``, etc.

Notes
-----
* Passing ``alpha`` is equivalent to passing ``width_mult`` to the underlying
  ``MobileNetV2`` constructor.  This wrapper merely renames the argument for
  convenience.
* If you request pretrained weights while ``alpha`` is not 1.0, a
  ``ValueError`` is raised because no such checkpoints exist.
"""

from functools import partial
from typing import Any, Optional, Union

from torch import nn, Tensor  # noqa: F401 (needed for type hints in docstrings)

# --- 기존 ---
# from torchvision.models._api import register_model, handle_legacy_interface

# --- 수정 ---
from torchvision.models._api import register_model
try:
    from torchvision.models._api import handle_legacy_interface
except ImportError:
    # torchvision>=0.16 에서는 사라진 심볼 → no‑op 데코레이터로 대체
    def handle_legacy_interface(**_ignore):
        def decorator(fn):
            return fn
        return decorator

from torchvision.models._utils import _ovewrite_named_param
from torchvision.transforms._presets import ImageClassification

from torchvision.models.mobilenetv2 import (
    MobileNet_V2_Weights,
)
from torchvision.models.quantization.mobilenetv2 import (
    QuantizableMobileNetV2,
    QuantizableInvertedResidual,
    MobileNet_V2_QuantizedWeights,
    _replace_relu,
    quantize_model,
)

__all__ = ["mobilenet_v2_custom"]


@register_model(name="mobilenet_v2_custom")
@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V2_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v2_custom(
    *,
    alpha: float = 1.0,
    weights: Optional[Union[MobileNet_V2_QuantizedWeights, MobileNet_V2_Weights]] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
):
    """Factory function for *MobileNetV2* with configurable ``alpha``.

    See the module‑level docstring for details.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha != 1.0 and weights is not None:
        raise ValueError(
            "Pre‑trained weights are only available for alpha == 1.0. "
            "Set weights=None or use alpha=1.0."
        )

    # Map user‑friendly name to the underlying argument expected by MobileNetV2.
    kwargs["width_mult"] = alpha
    # Avoid duplicate keyword if user also passed width_mult directly.
    kwargs.pop("alpha", None)

    # Resolve weight enum according to the quantization flag.
    weights = (MobileNet_V2_QuantizedWeights if quantize else MobileNet_V2_Weights).verify(weights)

    # Align num_classes / backend with checkpoint metadata when using weights.
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])

    # Default quantization backend if user did not specify one explicitly.
    backend = kwargs.pop("backend", "qnnpack")

    # Build the model.
    model = QuantizableMobileNetV2(block=QuantizableInvertedResidual, **kwargs)
    _replace_relu(model)

    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model
