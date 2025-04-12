from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.quantization import fuse_modules

# -----------------------------------------------------------------------------
# 1.  MobileNet V2 / V3 helpers
# -----------------------------------------------------------------------------
from timm.models._efficientnet_blocks import InvertedResidual, DepthwiseSeparableConv


def _fuse_mobilenetv2(model: nn.Module) -> None:
    """Fuse Conv+BN(+ReLU) blocks inside a timm MobileNetV2 model
    (handles both old separate‑ReLU and new BatchNormAct2d layouts)."""
    # 1) Stem --------------------------------------------------------------
    stem_has_act = hasattr(model, "act1")
    fuse_list = ["conv_stem", "bn1"] + (["act1"] if stem_has_act else [])
    fuse_modules(model, fuse_list, inplace=True)

    # 2) Blocks ------------------------------------------------------------
    for stage in model.blocks:
        for blk in stage:
            if isinstance(blk, DepthwiseSeparableConv):
                # conv_dw path
                has_act = hasattr(blk, "act1")
                fuse_modules(blk, ["conv_dw", "bn1"] + (["act1"] if has_act else []), inplace=True)
                # conv_pw path (always BN only)
                fuse_modules(blk, ["conv_pw", "bn2"], inplace=True)

            elif isinstance(blk, InvertedResidual):
                # conv_pw
                has_act = hasattr(blk, "act1")
                fuse_modules(blk, ["conv_pw", "bn1"] + (["act1"] if has_act else []), inplace=True)
                # conv_dw
                has_act = hasattr(blk, "act2")
                fuse_modules(blk, ["conv_dw", "bn2"] + (["act2"] if has_act else []), inplace=True)
                # conv_pwl
                fuse_modules(blk, ["conv_pwl", "bn3"], inplace=True)

    # 3) Head --------------------------------------------------------------
    head_has_act = hasattr(model, "act2")
    fuse_list = ["conv_head", "bn2"] + (["act2"] if head_has_act else [])
    fuse_modules(model, fuse_list, inplace=True)
    

def _fuse_mobilenetv3(model: nn.Module) -> None:
    """Fuse Conv+BN(+Activation) for timm MobileNetV3 (large/small)."""

    for m in model.modules():
        # timm MobileNetV3 blocks are `timm.models.mobilenetv3.InvertedResidual`.
        # They have attributes: conv, bn, act.
        if isinstance(m, nn.Sequential):
            # naive attempt: fuse consecutive (Conv, BN, Act) triples
            submods = list(m._modules.keys())
            for i in range(len(submods) - 2):
                trio = submods[i : i + 3]
                types = [type(m._modules[k]) for k in trio]
                if issubclass(types[0], nn.Conv2d) and issubclass(types[1], nn.BatchNorm2d):
                    fuse_modules(m, trio[:2] + ([] if not issubclass(types[2], nn.ReLU) else [trio[2]]), inplace=True)


# -----------------------------------------------------------------------------
# 2.  ResNet‑style helpers (BasicBlock & Bottleneck)
# -----------------------------------------------------------------------------
from timm.models.resnet import BasicBlock, Bottleneck


def _fuse_resnet(model: nn.Module) -> None:
    """Fuse Conv+BN+ReLU inside ResNet/ResNeXt/RegNet blocks."""

    for m in model.modules():
        if isinstance(m, BasicBlock):
            fuse_modules(m, ["conv1", "bn1", "act1"], inplace=True)
            fuse_modules(m, ["conv2", "bn2"], inplace=True)
        elif isinstance(m, Bottleneck):
            fuse_modules(m, ["conv1", "bn1", "act1"], inplace=True)
            fuse_modules(m, ["conv2", "bn2", "act2"], inplace=True)
            fuse_modules(m, ["conv3", "bn3"], inplace=True)


# -----------------------------------------------------------------------------
# 3.  Registry & public API
# -----------------------------------------------------------------------------

_FUSER_REGISTRY: Dict[str, Callable[[nn.Module], None]] = {
    "mobilenetv2": _fuse_mobilenetv2,
    "mobilenetv3": _fuse_mobilenetv3,
    "resnet": _fuse_resnet,
    "resnext": _fuse_resnet,
    "regnet": _fuse_resnet,
}


def fuse_model(model: nn.Module) -> None:
    
    name = model.__class__.__name__.lower()  # e.g. 'MobileNetV2' → 'mobilenetv2'

    # simple heuristic: find first registry key that is a substring of class name
    for key, fuser in _FUSER_REGISTRY.items():
        if key in name:
            fuser(model)
            return

    # If no key matched, silently skip (e.g., ViT) – user can add later.
    return
