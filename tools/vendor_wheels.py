from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


WEBVIEW_REQUIREMENTS = [
    "pywebview==3.4",
    "pyobjc-core==11.1; sys_platform == 'darwin'",
    "pyobjc-framework-Cocoa==11.1; sys_platform == 'darwin'",
    "pyobjc-framework-WebKit==11.1; sys_platform == 'darwin'",
    "cffi==2.0.0",
    "pycparser==2.23",
]

PIXAL3D_REQUIREMENTS = [
    "torch>=2.6.0",
    "torchvision>=0.21.0",
    "pillow==12.0.0",
    "imageio==2.37.2",
    "imageio-ffmpeg==0.6.0",
    "tqdm==4.67.1",
    "easydict==1.13",
    "opencv-python-headless==4.12.0.88",
    "trimesh==4.10.1",
    "transformers==4.57.3",
    "zstandard==0.25.0",
    "kornia==0.8.2",
    "timm==1.0.22",
    "diffusers==0.37.1",
    "accelerate==1.13.0",
    "plyfile==1.1.3",
    "gradio==6.0.1",
    "utils3d @ https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl",
]

MOGE_WHEEL_SOURCES = [
    "git+https://github.com/microsoft/MoGe.git",
    "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6",
]

CUDA_EXTENSION_REQUIREMENTS = [
    "natten==0.21.0",
]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and install extension-local wheels.")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--include-pixal3d", action="store_true")
    parser.add_argument("--include-cuda-extensions", action="store_true")
    parser.add_argument("--include-moge", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    wheels = root / "wheels"
    vendor = root / "_vendor"
    wheels.mkdir(parents=True, exist_ok=True)
    vendor.mkdir(parents=True, exist_ok=True)

    requirements = list(WEBVIEW_REQUIREMENTS)
    if args.include_pixal3d:
        requirements.extend(PIXAL3D_REQUIREMENTS)
    if args.include_cuda_extensions:
        requirements.extend(CUDA_EXTENSION_REQUIREMENTS)

    if args.download:
        run([
            sys.executable,
            "-m",
            "pip",
            "download",
            "--dest",
            str(wheels),
            "--only-binary=:all:",
            *requirements,
        ])
        if args.include_moge:
            run([
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
                *MOGE_WHEEL_SOURCES,
            ])

    if args.install:
        wheel_files = sorted(str(path) for path in wheels.rglob("*.whl"))
        if not wheel_files:
            raise SystemExit("No wheels found to install.")
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheels),
            "--target",
            str(vendor),
            "--upgrade",
            *wheel_files,
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
