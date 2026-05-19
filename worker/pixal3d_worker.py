from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from pathlib import Path


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "_vendor"
    for path in (root, vendor):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if platform.system().lower() == "windows":
        triton_root = vendor / "triton"
        bundled_cc = triton_root / "runtime" / "tcc" / "tcc.exe"
        if bundled_cc.is_file():
            os.environ.setdefault("CC", str(bundled_cc))
        bundled_cuda = triton_root / "backends" / "nvidia"
        if (
            (bundled_cuda / "bin" / "ptxas.exe").is_file()
            and (bundled_cuda / "include" / "cuda.h").is_file()
            and (bundled_cuda / "lib" / "x64" / "cuda.lib").is_file()
        ):
            os.environ.setdefault("CUDA_PATH", str(bundled_cuda))
            os.environ.setdefault("CUDA_HOME", str(bundled_cuda))
            cuda_bin = str(bundled_cuda / "bin")
            path_entries = [entry.lower() for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
            if cuda_bin.lower() not in path_entries:
                os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        try:
            from dependency_manager import configure_windows_triton_environment, patch_windows_triton_python_dev_paths

            configure_windows_triton_environment(os.environ)
            patch_windows_triton_python_dev_paths()
        except Exception:
            pass
    return root


def _log(message: str) -> None:
    print(message, flush=True)


def _resolved_backend(requested_device: str) -> str:
    value = (requested_device or "auto").strip().lower()
    if value.startswith("cuda"):
        return "cuda"
    if value in {"mps", "metal"}:
        return "metal"
    if value == "cpu":
        return "cpu"
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        return "metal"
    return "cuda" if system in {"linux", "windows"} else "cpu"


def main() -> int:
    _bootstrap()
    parser = argparse.ArgumentParser(description="Run Pixal3D inference from the Blender extension.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cuda:0", "mps", "metal", "cpu"])
    parser.add_argument("--decimation_target", type=int, default=300000)
    parser.add_argument("--target_resolution", type=int, default=1536, choices=[1024, 1536])
    parser.add_argument("--max_num_tokens", type=int, default=32768)
    parser.add_argument("--ss_sampling_steps", type=int, default=16)
    parser.add_argument("--shape_sampling_steps", type=int, default=16)
    parser.add_argument("--tex_sampling_steps", type=int, default=16)
    parser.add_argument("--texture_size", type=int, default=4096)
    parser.add_argument("--low_vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable_mps_fallback", action="store_true")
    args = parser.parse_args()

    _log("[Worker] Pixal3D worker starting")
    _log(f"[Worker] Python: {sys.executable}")
    _log(f"[Worker] Platform: {platform.platform()}")
    _log(f"[Worker] Working directory: {Path.cwd()}")
    _log(f"[Worker] Input image: {args.image}")
    _log(f"[Worker] Output GLB: {args.output}")
    _log(f"[Worker] Model: {args.model_path}")
    _log(f"[Worker] Seed: {args.seed}")
    _log(f"[Worker] Decimation target: {args.decimation_target or 'full detail'}")
    _log(f"[Worker] Target resolution: {args.target_resolution}")
    _log(f"[Worker] Max sparse tokens: {args.max_num_tokens}")
    _log(f"[Worker] Sampling steps: {args.ss_sampling_steps}/{args.shape_sampling_steps}/{args.tex_sampling_steps}")
    _log(f"[Worker] Texture size: {args.texture_size}")
    _log(f"[Worker] Low-VRAM staged placement: {args.low_vram}")
    _log(f"[Worker] Device request: {args.device}")
    backend = _resolved_backend(args.device)
    _log(f"[Worker] Resolved backend: {backend.upper()}")
    if backend == "metal":
        _log("[Worker] Metal GPU backend: PyTorch MPS")
        _log(f"[Worker] Unsupported-op CPU fallback: {not args.disable_mps_fallback}")
    elif backend == "cuda":
        _log("[Worker] CUDA runtime profile: Torch 2.7 / CUDA 12.8 with native Pixal3D wheels")

    try:
        _log("[Worker] Importing Pixal3D inference runtime...")
        from inference import run_inference
    except Exception as error:
        print(f"Could not import Pixal3D inference runtime: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2

    try:
        _log("[Worker] Running Pixal3D inference...")
        run_inference(
            image_path=args.image,
            output_path=args.output,
            seed=args.seed,
            model_path=args.model_path,
            device=args.device,
            decimation_target=args.decimation_target,
            target_resolution=args.target_resolution,
            max_num_tokens=args.max_num_tokens,
            ss_sampling_steps=args.ss_sampling_steps,
            shape_slat_sampling_steps=args.shape_sampling_steps,
            tex_slat_sampling_steps=args.tex_sampling_steps,
            texture_size=args.texture_size,
            low_vram=args.low_vram,
            enable_mps_fallback=not args.disable_mps_fallback,
        )
    except Exception as error:
        print(f"Pixal3D generation failed: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1

    _log(f"Generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
