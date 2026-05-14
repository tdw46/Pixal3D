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
    return root


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    _bootstrap()
    parser = argparse.ArgumentParser(description="Run Pixal3D inference from the Blender extension.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cuda:0", "mps", "metal", "cpu"])
    parser.add_argument("--decimation_target", type=int, default=0)
    parser.add_argument("--texture_size", type=int, default=2048)
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
    _log(f"[Worker] Texture size: {args.texture_size}")
    _log(f"[Worker] Device request: {args.device}")
    _log("[Worker] Metal GPU backend: PyTorch MPS when device resolves to Metal")
    _log(f"[Worker] Unsupported-op CPU fallback: {not args.disable_mps_fallback}")

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
            texture_size=args.texture_size,
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
