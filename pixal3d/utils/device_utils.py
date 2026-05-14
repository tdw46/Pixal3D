from __future__ import annotations

import os
import platform
from typing import Any

import torch


def mps_is_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(requested: str | torch.device | None = "auto") -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device(value)
    if value == "mps":
        if not mps_is_available():
            raise RuntimeError("Metal/MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if value == "metal":
        if not mps_is_available():
            raise RuntimeError("Metal/MPS was requested, but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    return torch.device(value)


def configure_backend_for_device(device: torch.device, *, enable_mps_fallback: bool = True) -> None:
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if enable_mps_fallback else "0"
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_CONV_BACKEND", "metal")
        os.environ.setdefault("PIXAL3D_DEVICE", "mps")
    elif device.type == "cuda":
        os.environ.setdefault("ATTN_BACKEND", "flash_attn_3")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", os.environ.get("ATTN_BACKEND", "flash_attn_3"))
        os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
        os.environ.setdefault("PIXAL3D_DEVICE", str(device))
    else:
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
        os.environ.setdefault("SPARSE_CONV_BACKEND", "metal")
        os.environ.setdefault("PIXAL3D_DEVICE", device.type)


def default_device() -> torch.device:
    return resolve_device(os.environ.get("PIXAL3D_DEVICE", "auto"))


def module_device(module: Any, fallback: str | torch.device | None = None) -> torch.device:
    if hasattr(module, "parameters"):
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass
    if fallback is not None:
        return resolve_device(fallback)
    return default_device()


def synchronize(device: torch.device | str | None = None) -> None:
    resolved = resolve_device(device or default_device())
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_cache(device: torch.device | str | None = None) -> None:
    resolved = resolve_device(device or default_device())
    if resolved.type == "cuda":
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"CUDA ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "CUDA"
    if device.type == "mps":
        return "Apple Metal/MPS"
    if platform.system().lower() == "darwin" and device.type == "cpu":
        return "CPU on macOS"
    return device.type.upper()
