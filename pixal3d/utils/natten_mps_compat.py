from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any


def install() -> bool:
    try:
        import natten_mps
        import natten_mps.functional as mps_functional
    except Exception:
        return False

    natten_mps.set_backend("metal")

    functional = types.ModuleType("natten.functional")
    functional.__spec__ = ModuleSpec("natten.functional", loader=None)

    def _to_mps_layout(tensor):
        return tensor.permute(0, 2, 3, 1, 4).contiguous()

    def _from_mps_layout(tensor):
        return tensor.permute(0, 3, 1, 2, 4).contiguous()

    def na2d_qk(query, key, kernel_size, dilation=1, **kwargs: Any):
        return _from_mps_layout(
            mps_functional.na2d_qk(
                _to_mps_layout(query),
                _to_mps_layout(key),
                kernel_size=kernel_size,
                dilation=dilation,
                stride=kwargs.get("stride", 1),
                is_causal=kwargs.get("is_causal", False),
                scale=kwargs.get("scale"),
            )
        )

    def na2d_av(attn, value, kernel_size, dilation=1, **kwargs: Any):
        return _from_mps_layout(
            mps_functional.na2d_av(
                _to_mps_layout(attn),
                _to_mps_layout(value),
                kernel_size=kernel_size,
                dilation=dilation,
                stride=kwargs.get("stride", 1),
                is_causal=kwargs.get("is_causal", False),
            )
        )

    def na2d(query, key, value, kernel_size, dilation=1, **kwargs: Any):
        return _from_mps_layout(
            mps_functional.na2d(
                _to_mps_layout(query),
                _to_mps_layout(key),
                _to_mps_layout(value),
                kernel_size=kernel_size,
                dilation=dilation,
                stride=kwargs.get("stride", 1),
                is_causal=kwargs.get("is_causal", False),
                scale=kwargs.get("scale"),
            )
        )

    functional.na2d_qk = na2d_qk
    functional.na2d_av = na2d_av
    functional.na2d = na2d

    natten_module = types.ModuleType("natten")
    natten_module.__spec__ = ModuleSpec("natten", loader=None, is_package=True)
    natten_module.na2d_qk = na2d_qk
    natten_module.na2d_av = na2d_av
    natten_module.na2d = na2d
    natten_module.functional = functional
    natten_module.__version__ = getattr(natten_mps, "__version__", "0")

    sys.modules["natten"] = natten_module
    sys.modules["natten.functional"] = functional
    return True
