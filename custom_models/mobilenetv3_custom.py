# mobilenetv3_custom.py
"""
MobileNetV3 (Large) with user‑controllable width multiplier ``alpha``.

This module is a thin wrapper around `torchvision.models.quantization.mobilenetv3`
that exposes the *width multiplier* (commonly denoted ``alpha`` in the MobileNet
papers) as a first‑class argument while keeping full compatibility with the
original API – including support for quantized inference, pretrained weights
(`alpha == 1.0`), TorchScript, and the internal registration mechanism that
allows you to call the model via ``torchvision.models.quantization.mobilenet_v3_custom``.

Examples
--------
>>> from mobilenetv3_custom import mobilenet_v3_custom
>>> # 0.75× wide floating‑point model
>>> model = mobilenet_v3_custom(alpha=0.75, quantize=False)
>>> # 1.0× wide INT8 model for CPU inference
>>> qmodel = mobilenet_v3_custom(alpha=1.0, quantize=True, weights='DEFAULT')

Parameters
----------
alpha : float, default 1.0
    Width multiplier that scales the number of channels in every layer.
    Typical values are 0.35, 0.5, 0.75, 1.0, 1.25.
    Pre‑trained weights are only available for ``alpha == 1.0``.
**kwargs :
    Any additional keyword arguments accepted by
    :pyfunc:`torchvision.models.quantization.mobilenet_v3_large`, e.g.
    ``num_classes``, ``backend``, ``weights``, …

Note
----
* Passing ``alpha`` is equivalent to passing ``width_mult`` to the original
  torchvision implementation.  This wrapper simply renames the argument for
  convenience and readability.
* If ``alpha`` is different from 1.0 *and* ``weights`` is not ``None`` we will
  raise a ``ValueError`` because no matching checkpoints exist.

"""

from functools import partial
from typing import Any, Optional, Union

import torchvision
from torchvision.models._api import register_model
try:
    from torchvision.models._api import handle_legacy_interface
except ImportError:          # torchvision ≥ 0.16
    def handle_legacy_interface(**_ignored):
        # 그냥 원본 함수를 그대로 반환하는 no‑op 데코레이터
        def decorator(fn):
            return fn
        return decorator
    
from torchvision.models._utils import _ovewrite_named_param
from torchvision.transforms._presets import ImageClassification

# Re‑export the original quantization helpers so that users do not need to dig
# into private torchvision modules.
from torchvision.models.quantization.mobilenetv3 import (  # noqa: F401
    QuantizableMobileNetV3,
    QuantizableInvertedResidual,
    QuantizableSqueezeExcitation,
)

from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision.models.quantization.mobilenetv3 import (
    _mobilenet_v3_model as _tv_mobilenet_v3_model,
    MobileNet_V3_Large_QuantizedWeights,
    MobileNet_V3_Large_Weights,
)
from torchvision.models._meta import _IMAGENET_CATEGORIES

__all__ = ["mobilenet_v3_custom"]


@register_model(name="mobilenet_v3_custom")
@handle_legacy_interface(
    weights=(
        "pretrained",
        lambda kwargs: MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        if kwargs.get("quantize", False)
        else MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    )
)
def mobilenet_v3_custom(
    *,
    alpha: float = 1.0,
    weights: Optional[
        Union[MobileNet_V3_Large_QuantizedWeights, MobileNet_V3_Large_Weights]
    ] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
):
    """
    Construct a MobileNetV3‑Large model with a user‑defined width multiplier.

    See the module‑level docstring for a full description of the arguments.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha != 1.0 and weights is not None:
        raise ValueError(
            "Pre‑trained weights are only available for alpha == 1.0. "
            "Please set ``weights=None`` or use alpha=1.0."
        )

    # Rename for underlying torchvision implementation.
    kwargs["width_mult"] = alpha
    # Ensure we do not accidentally pass duplicate keys.
    kwargs.pop("alpha", None)

    # If weights are provided, align the number of classes with the checkpoint.
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])

    # Build the layer configuration and create the model using torchvision's
    # internal helper (which also handles QAT / conversion when quantize=True).
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large", **kwargs
    )
    model = _tv_mobilenet_v3_model(
        inverted_residual_setting,
        last_channel,
        weights,
        progress,
        quantize,
        **kwargs,
    )

    return model
