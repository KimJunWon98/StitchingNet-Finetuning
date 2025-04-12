# custom_models/mobilenetv3_small_custom.py
from functools import partial
from typing import Any, Optional, Union
from torchvision.models._api import register_model
try:
    from torchvision.models._api import handle_legacy_interface
except ImportError:            # torchvision ≥0.16
    def handle_legacy_interface(**_):
        def decorator(fn): return fn
        return decorator

from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
# from torchvision.models.quantization.mobilenetv3 import (
#     _mobilenet_v3_model as _tv_model,
#     QuantizableMobileNetV3, QuantizableInvertedResidual,
#     MobileNet_V3_Large_QuantizedWeights,  # 사용 X, but needed for typing
#     MobileNet_V3_Small_Weights,
# )


# ① quantization 서브모듈에서 가져올 항목
from torchvision.models.quantization.mobilenetv3 import (
    _mobilenet_v3_model as _tv_model,
    QuantizableMobileNetV3,
    QuantizableInvertedResidual,
)

# ② *가중치* 열거형은 **mobilenetv3 원본 모듈**에서 가져와야 함
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

from torchvision.models._utils import _ovewrite_named_param

__all__ = ["mobilenet_v3_small_custom"]

@register_model("mobilenet_v3_small_custom")
@handle_legacy_interface(weights=("pretrained",
                                  lambda kw: MobileNet_V3_Small_Weights.IMAGENET1K_V1))
def mobilenet_v3_small_custom(
    *, alpha: float = 1.0,
    weights: Optional[MobileNet_V3_Small_Weights] = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if alpha != 1.0 and weights is not None:
        raise ValueError("α≠1.0 에는 pretrained 가중치가 없습니다.")

    kwargs["width_mult"] = alpha
    kwargs.pop("alpha", None)

    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    inv_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)

    model = _tv_model(inv_setting, last_channel,
                      weights, progress, quantize, **kwargs)
    return model
