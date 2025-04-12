"""fuse_timm_mobile.py
================================
Utilities to **fuse Conv‑BN(+Act) blocks** inside *timm* MobileNet family
(V2 & V3) so the resulting model mimics torchvision's fused layout and can be
fed into QAT / PTQ pipelines without extra hassle.

Public API
----------
```python
from fuse_timm_mobile import (
    fuse_conv_bnact,        # in‑place fuse of an existing model
    load_fused_timm_model,  # create + fuse in one call (helper)
)
```

Example
~~~~~~~
```python
model = load_fused_timm_model("mobilenetv3_small_050", pretrained=True)
print(sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()))  # 0
```
"""
from __future__ import annotations

import copy
import re
from typing import Tuple

import torch
from torch import nn

# ---------------------------------------------------------------------------
# 0)  fuse_conv_bn_eval – import‑with‑fallback (PyTorch 1.13 → 2.6+)
# ---------------------------------------------------------------------------
try:
    # ≤ 2.1
    from torch.ao.quantization.utils import fuse_conv_bn_eval  # type: ignore
except ImportError:  # pragma: no cover
    try:
        # 2.2 – 2.5
        from torch.ao.quantization.fuse_modules import fuse_conv_bn_eval  # type: ignore
    except ImportError:  # pragma: no cover
        try:
            # 2.6+
            from torch.ao.quantization._equalize import fuse_conv_bn_eval  # type: ignore
        except ImportError:  # pragma: no cover

            def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:  # noqa: D401 – simple desc
                """Fallback implementation that merges *conv* + *bn* weights (eval‑mode)."""
                with torch.no_grad():
                    fused = copy.deepcopy(conv)

                    w_conv = conv.weight
                    b_conv = conv.bias if conv.bias is not None else torch.zeros(
                        w_conv.size(0), device=w_conv.device
                    )

                    w_bn = bn.weight / torch.sqrt(bn.running_var + bn.eps)
                    b_bn = bn.bias - bn.running_mean * w_bn
                    w_bn = w_bn.reshape(-1, 1, 1, 1)

                    fused.weight.copy_(w_conv * w_bn)
                    fused.bias = nn.Parameter(b_conv * w_bn.flatten() + b_bn)

                return fused

# ---------------------------------------------------------------------------
# 1)  Conv‑BN‑Act fuse for timm MobileNet blocks
# ---------------------------------------------------------------------------
from timm.models.layers import BatchNormAct2d  # type: ignore

_ACT_TYPES = (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.SiLU, nn.Identity)

class _ActOnly(nn.Sequential):
    """A wrapper that holds only the activation layer (for post‑fuse graph)."""

    def __init__(self, act: nn.Module):
        super().__init__(act)


def _fuse_block_pair(parent: nn.Module, idx: int, kids: list[Tuple[str, nn.Module]]) -> bool:
    """Try fusing kids[idx] (Conv2d) + kids[idx+1] (BatchNormAct2d)."""
    n1, m1 = kids[idx]
    n2, m2 = kids[idx + 1]

    if not (isinstance(m1, nn.Conv2d) and isinstance(m2, BatchNormAct2d)):
        return False

    bn = next((c for c in m2.modules() if isinstance(c, nn.BatchNorm2d)), None)
    if bn is None:
        return False

    act = next((c for c in m2.modules() if isinstance(c, _ACT_TYPES)), nn.Identity())

    fused_conv = fuse_conv_bn_eval(m1, bn)
    setattr(parent, n1, fused_conv)
    setattr(parent, n2, _ActOnly(copy.deepcopy(act)))
    return True


def fuse_conv_bnact(model: nn.Module) -> int:  # noqa: D401
    """Fuse *all* Conv2d + BatchNormAct2d pairs **in‑place**.

    Returns
    -------
    int
        Number of fused pairs.
    """
    fused_cnt = 0

    def _recursive(mod: nn.Module) -> None:
        nonlocal fused_cnt
        kids = list(mod.named_children())
        i = 0
        while i < len(kids) - 1:
            if _fuse_block_pair(mod, i, kids):
                fused_cnt += 1
                i += 2
            else:
                i += 1
        for _, child in mod.named_children():
            _recursive(child)

    _recursive(model)
    return fused_cnt


# ---------------------------------------------------------------------------
# 2)  Helper: create + fuse in one shot
# ---------------------------------------------------------------------------
import timm  # type: ignore

def _parse_width(model_name: str) -> float:
    m = re.search(r"_([0-9]{3})$", model_name)
    if not m:
        raise ValueError(f"Cannot parse width from '{model_name}'.")
    return int(m.group(1)) / 100.0


def load_fused_timm_model(
    timm_name: str,
    *,
    pretrained: bool = True,
    **timm_kwargs,
) -> nn.Module:
    """Create a *timm* MobileNet, fuse Conv‑BN‑Act, and return the model."""
    model = timm.create_model(timm_name, pretrained=pretrained, **timm_kwargs).eval()
    pairs = fuse_conv_bnact(model)
    if pairs == 0:
        raise RuntimeError(
            f"No Conv‑BN pairs were fused – '{timm_name}' may use an unsupported layout."
        )
    return model


# ---------------------------------------------------------------------------
# 3)  Self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        description="Fuse Conv+BN(+Act) blocks inside timm MobileNet models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """Examples
            --------
            python fuse_timm_mobile.py mobilenetv3_small_050  --pretrained
            python fuse_timm_mobile.py mobilenetv2_075
            """,
        ),
    )
    parser.add_argument("model", help="timm model name, e.g. mobilenetv3_small_050")
    parser.add_argument("--pretrained", action="store_true", help="load ImageNet weights")
    args = parser.parse_args()

    m = load_fused_timm_model(args.model, pretrained=args.pretrained)
    bn_left = sum(isinstance(x, nn.BatchNorm2d) for x in m.modules())
    print(f"Model '{args.model}' fused successfully → BN layers left: {bn_left}")
    x = torch.randn(1, 3, 224, 224)
    print("Output shape:", m(x).shape)
