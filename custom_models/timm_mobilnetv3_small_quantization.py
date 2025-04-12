"""
MobileNetV3‑Small wrapper that keeps the torchvision API while
(optionally) loading timm pretrained weights.

Example
-------
>>> from mobilenetv3_small_custom import timm_mobilenet_v3_small_custom
>>> model = timm_mobilenet_v3_small_custom(alpha=0.75, pretrained="timm")
>>> x = torch.randn(1, 3, 224, 224)
>>> logits = model(x)                     # (1, 1000)
"""
from typing import Any, Optional
import warnings

import timm          # pip install timm>=0.9
import torch
from torchvision.models._api import register_model
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision.models.quantization.mobilenetv3 import (
    _mobilenet_v3_model as _tv_mobilenet_v3_model,
)

__all__ = ["timm_mobilenet_v3_small_custom"]

# α 값 ↔ timm 모델 이름
_TIMM_VARIANTS = {
    0.50: "mobilenetv3_small_050",
    0.75: "mobilenetv3_small_075",
    1.00: "mobilenetv3_small_100",
}

def _copy_by_order_excluding_classifier(src: torch.nn.Module, dst: torch.nn.Module) -> float:
    """
    Copy parameters from timm's src to torchvision's dst, ignoring parameter names.

    The twist here is that we exclude any parameters whose key starts with
    'classifier.' from the fraction calculation (both from total and matched).
    However, we still perform the actual copy if the shapes match.

    Returns
    -------
    fraction : float
        Percentage of *non‑classifier* src parameters that got successfully copied.
    """
    src_state = src.state_dict()
    dst_state = dst.state_dict()

    # We'll do a single pass in definition order.
    src_items = list(src_state.items())  # [(key, tensor), ...]
    dst_items = list(dst_state.items())

    total_elems = 0  # total number of elements (excluding classifier)
    matched_elems = 0

    with torch.no_grad():
        for (k_s, v_s), (k_d, v_d) in zip(src_items, dst_items):
            # We define 'classifier' param to be anything whose key starts with 'classifier.'
            # (That holds for timm mobilenetv3_small: 'classifier.weight', 'classifier.bias', etc.)
            is_classifier_param = k_s.startswith('classifier.')

            # For fraction calc, skip classifier from total
            if not is_classifier_param:
                total_elems += v_s.numel()

            # Always copy if shapes match (including classifier), so user can still get it
            if v_s.shape == v_d.shape:
                v_d.copy_(v_s)
                # If it's not classifier, add to matched_elems
                if not is_classifier_param:
                    matched_elems += v_s.numel()

    if total_elems == 0:
        # edge case: if there's no non‑classifier param
        fraction = 100.0
    else:
        fraction = matched_elems / total_elems * 100.0
    return fraction

@register_model("timm_mobilenet_v3_small_custom")
def timm_mobilenet_v3_small_custom(
    *,
    alpha: float = 1.0,
    pretrained: Optional[str] = None,   # "timm" | None
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> torch.nn.Module:
    # ---------- 기본 검증 ----------
    if alpha not in _TIMM_VARIANTS:
        raise ValueError(f"Unsupported alpha {alpha}. Choose from {list(_TIMM_VARIANTS)}")
    if quantize:
        raise NotImplementedError("MobileNetV3‑Small INT8 checkpoints are not available.")

    # ---------- 1) torchvision 구조 ----------
    kwargs["width_mult"] = alpha
    inv_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    # _mobilenet_v3_model 는 width_mult 를 이미 소비했으므로 중복 전달되지 않게 pop
    kwargs.pop("width_mult", None)

    tv_model = _tv_mobilenet_v3_model(
        inv_setting,
        last_channel,
        weights=None,            # timm 쪽에서 로드
        progress=progress,
        quantize=False,
        **kwargs,
    )

    # ---------- 2) timm 가중치 → copy‑by‑order ----------
    if pretrained == "timm":
        timm_name = _TIMM_VARIANTS[alpha]
        tm_model = timm.create_model(timm_name, pretrained=True)

        # 분류 클래스 수가 1000이 아니라면 마지막 FC 레이어는 스킵(자동 초기화)
        if kwargs.get("num_classes", 1000) != 1000:
            warnings.warn(
                "num_classes ≠ 1000 → classifier weights will be randomly initialized",
                stacklevel=2,
            )

        fraction = _copy_by_order_excluding_classifier(tm_model, tv_model)
        print(f"[timm_mobilenet_v3_small_custom] {fraction:.2f}% of non‑classifier timm parameters were copied.")

    return tv_model
