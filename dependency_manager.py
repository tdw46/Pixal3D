from __future__ import annotations

import importlib.machinery
import json
import os
import platform
import site
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


STATE_VERSION = 1

WEBVIEW_MODULES = (
    "webview",
)

PIXAL3D_MODULES = (
    "torch",
    "torchvision",
    "PIL",
    "cv2",
    "numpy",
    "diffusers",
    "accelerate",
    "transformers",
    "trimesh",
    "fast_simplification",
    "xatlas",
    "plyfile",
    "zstandard",
    "kornia",
    "timm",
    "einops",
    "natten_mps",
    "moge",
    "utils3d",
    "natten",
    "o_voxel.convert",
    "cumesh",
    "flex_gemm",
    "flash_attn_interface",
)

CUDA_ONLY_MODULES = (
    "natten",
    "o_voxel",
    "cumesh",
    "flex_gemm",
    "flash_attn_interface",
)

METAL_GENERATION_MODULES = (
    "torch",
    "torchvision",
    "PIL",
    "cv2",
    "numpy",
    "diffusers",
    "accelerate",
    "transformers",
    "trimesh",
    "fast_simplification",
    "xatlas",
    "plyfile",
    "zstandard",
    "kornia",
    "timm",
    "einops",
    "natten_mps",
    "moge",
    "utils3d",
    "o_voxel",
)

WEBVIEW_PACKAGES = (
    "pywebview==3.4",
    "pyobjc-core==11.1",
    "pyobjc-framework-Cocoa==11.1",
    "pyobjc-framework-WebKit==11.1",
    "cffi>=1.17",
    "pycparser==2.23",
)

METAL_PACKAGES = (
    "torch",
    "torchvision",
    "pillow",
    "numpy==2.2.6",
    "imageio",
    "imageio-ffmpeg",
    "tqdm",
    "easydict",
    "opencv-python-headless",
    "trimesh",
    "fast-simplification==0.1.13",
    "xatlas==0.0.11",
    "transformers",
    "zstandard",
    "kornia",
    "timm",
    "einops",
    "natten-mps==0.3.0",
    "diffusers",
    "accelerate",
    "plyfile",
    "gradio",
    "utils3d",
    "moge",
    "pipeline",
)

OPEN_MODEL_ASSET_REPOS = (
    "TencentARC/Pixal3D",
    "ZhengPeng7/BiRefNet",
    "Ruicheng/moge-2-vitl",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
)

NAF_CHECKPOINT_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_CHECKPOINT_NAME = "naf_release.pth"


@dataclass
class RuntimeStatus:
    webview_ready: bool
    generation_ready: bool
    missing_webview_modules: list[str]
    missing_generation_modules: list[str]
    platform_key: str
    python_executable: str
    vendor_dir: Path
    wheels_dir: Path
    install_log_path: Path
    state_path: Path
    last_updated: str
    last_error: str
    unsupported_notes: list[str]


def extension_root() -> Path:
    return Path(__file__).resolve().parent


def vendor_dir() -> Path:
    return extension_root() / "_vendor"


def wheels_dir() -> Path:
    return extension_root() / "wheels"


def wheels_cache_dir() -> Path:
    return wheels_dir() / "cache"


def install_log_path() -> Path:
    return wheels_dir() / "install.log"


def install_state_path() -> Path:
    return wheels_dir() / "state.json"


def ensure_runtime_paths() -> None:
    root_path = str(extension_root())
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    vendor_path = vendor_dir()
    if vendor_path.is_dir():
        site.addsitedir(str(vendor_path))
        vendor_value = str(vendor_path)
        if vendor_value in sys.path:
            sys.path.remove(vendor_value)
        sys.path.insert(0, vendor_value)


def ensure_runtime_directories() -> None:
    vendor_dir().mkdir(parents=True, exist_ok=True)
    wheels_dir().mkdir(parents=True, exist_ok=True)
    wheels_cache_dir().mkdir(parents=True, exist_ok=True)


def blender_python() -> str:
    if bpy is not None:
        python_path = getattr(getattr(bpy, "app", None), "binary_path_python", "")
        if python_path:
            return python_path
    return sys.executable


def platform_key() -> str:
    machine = platform.machine().lower().replace("amd64", "x86_64")
    if machine == "aarch64":
        machine = "arm64"
    return f"{platform.system().lower()}-{machine}-cp{sys.version_info.major}{sys.version_info.minor}"


def _read_state() -> dict:
    path = install_state_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(data: dict) -> None:
    ensure_runtime_directories()
    install_state_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _module_is_available(module_name: str) -> bool:
    ensure_runtime_paths()
    search_paths = [str(vendor_dir())] + list(sys.path)
    return importlib.machinery.PathFinder.find_spec(module_name, search_paths) is not None


def missing_modules(module_names: tuple[str, ...]) -> list[str]:
    return [module_name for module_name in module_names if not _module_is_available(module_name)]


def unsupported_notes() -> list[str]:
    notes: list[str] = []
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        notes.append(
            "Pixal3D/Trellis.2 upstream inference is documented for Linux with NVIDIA CUDA; "
            "the CUDA extension wheels are not available for this macOS Blender runtime."
        )
    if sys.version_info[:2] != (3, 10):
        notes.append(
            "The published Hugging Face demo wheels for o_voxel, flex_gemm, cumesh, and flash_attn_3 "
            "target CPython 3.10, while Blender 5.0 uses CPython "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        notes.append(
            "Apple Metal/MPS can run some torch models, but Pixal3D calls CUDA-only modules and .cuda() paths."
        )
    return notes


def resolved_generation_backend(requested_device: str = "auto") -> str:
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


def get_runtime_status(requested_device: str = "auto") -> RuntimeStatus:
    ensure_runtime_paths()
    state = _read_state()
    missing_webview = missing_modules(WEBVIEW_MODULES)
    backend = resolved_generation_backend(requested_device)
    required_generation = PIXAL3D_MODULES if backend == "cuda" else METAL_GENERATION_MODULES
    missing_generation = missing_modules(required_generation)
    cuda_missing = [module_name for module_name in CUDA_ONLY_MODULES if module_name in missing_generation]
    notes = unsupported_notes()
    if backend == "metal":
        notes = [
            note for note in notes
            if "CUDA extension wheels are not available" not in note
            and "CPython 3.10" not in note
            and "calls CUDA-only modules" not in note
        ]
        if "o_voxel.convert" in missing_generation:
            notes.append(
                "Missing o_voxel. Pixal3D/TRELLIS.2 shape decoding requires o_voxel up front."
            )
        else:
            notes.append(
                "Metal support uses PyTorch MPS on the Apple GPU, SDPA attention, natten-mps Metal neighborhood attention for NAF, "
                "a Metal sparse-conv compatibility backend, and the bundled pure-Python o_voxel.convert compatibility layer for shape decoding."
            )
        notes.append(
            "Use Prepare Open Model Assets to cache the open macOS model stack up front: "
            + ", ".join((*OPEN_MODEL_ASSET_REPOS, "valeoai/NAF"))
        )
    generation_ready = not missing_generation
    if cuda_missing and not notes:
        notes.append("CUDA runtime modules are missing: " + ", ".join(cuda_missing))
    return RuntimeStatus(
        webview_ready=not missing_webview,
        generation_ready=generation_ready,
        missing_webview_modules=missing_webview,
        missing_generation_modules=missing_generation,
        platform_key=platform_key(),
        python_executable=blender_python(),
        vendor_dir=vendor_dir(),
        wheels_dir=wheels_dir(),
        install_log_path=install_log_path(),
        state_path=install_state_path(),
        last_updated=str(state.get("last_updated", "")),
        last_error=str(state.get("last_error", "")),
        unsupported_notes=notes,
    )


def install_bundled_wheels() -> tuple[bool, str]:
    ensure_runtime_directories()
    wheel_files = sorted(str(path) for path in wheels_dir().glob("*.whl"))
    if not wheel_files:
        return False, f"No wheel files found in {wheels_dir()}"

    command = [
        blender_python(),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(wheels_dir()),
        "--find-links",
        str(wheels_dir() / "cache_metal"),
        "--target",
        str(vendor_dir()),
        "--upgrade",
        *WEBVIEW_PACKAGES,
        *METAL_PACKAGES,
    ]
    with install_log_path().open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== Install bundled wheels {datetime.now(timezone.utc).isoformat()} ===\n")
        log_file.write("Command: " + " ".join(command) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        log_file.write(result.stdout or "")
        log_file.write(result.stderr or "")

    state = {
        "state_version": STATE_VERSION,
        "platform_key": platform_key(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "last_error": "" if result.returncode == 0 else (result.stderr or result.stdout or "pip failed"),
    }
    _write_state(state)
    if result.returncode != 0:
        return False, state["last_error"]
    return True, f"Installed {len(wheel_files)} bundled wheel(s) into {vendor_dir()}"


def prepare_open_model_assets() -> tuple[bool, str]:
    ensure_runtime_paths()
    try:
        from huggingface_hub import snapshot_download
    except Exception as error:
        return False, f"huggingface_hub is not installed in the Pixal3D runtime: {error}"

    ensure_runtime_directories()
    with install_log_path().open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== Prepare open model assets {datetime.now(timezone.utc).isoformat()} ===\n")
        for repo_id in OPEN_MODEL_ASSET_REPOS:
            try:
                log_file.write(f"Downloading/checking {repo_id}\n")
                path = snapshot_download(repo_id=repo_id)
                log_file.write(f"Ready: {repo_id} -> {path}\n")
            except Exception as error:
                message = f"Could not prepare {repo_id}: {error}"
                log_file.write(message + "\n")
                state = {
                    "state_version": STATE_VERSION,
                    "platform_key": platform_key(),
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "last_error": message,
                }
                _write_state(state)
                return False, message
        try:
            path = _prepare_naf_assets(log_file)
            log_file.write(f"Ready: valeoai/NAF -> {path}\n")
        except Exception as error:
            message = f"Could not prepare valeoai/NAF: {error}"
            log_file.write(message + "\n")
            state = {
                "state_version": STATE_VERSION,
                "platform_key": platform_key(),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "last_error": message,
            }
            _write_state(state)
            return False, message
    state = {
        "state_version": STATE_VERSION,
        "platform_key": platform_key(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "last_error": "",
    }
    _write_state(state)
    return True, "Prepared open Pixal3D model assets: " + ", ".join((*OPEN_MODEL_ASSET_REPOS, "valeoai/NAF"))


def _prepare_naf_assets(log_file) -> Path:
    import certifi
    import requests
    import torch

    try:
        from pixal3d.utils.natten_mps_compat import install as install_natten_mps_compat

        install_natten_mps_compat()
    except Exception as error:
        log_file.write(f"natten-mps alias unavailable during NAF prep: {error}\n")

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / NAF_CHECKPOINT_NAME
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        log_file.write(f"Downloading/checking valeoai/NAF checkpoint {NAF_CHECKPOINT_URL}\n")
        response = requests.get(NAF_CHECKPOINT_URL, stream=True, timeout=60, verify=certifi.where())
        response.raise_for_status()
        tmp_path = checkpoint_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(checkpoint_path)
    log_file.write("Downloading/checking valeoai/NAF torch hub repo\n")
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    torch.hub.load("valeoai/NAF", "naf", pretrained=False, device=device, trust_repo=True)
    return checkpoint_path


def run_worker(image_path: str, output_path: str, seed: int, model_path: str, device: str = "auto", enable_mps_fallback: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    vendor_path = str(vendor_dir())
    root_path = str(extension_root())
    env["PYTHONPATH"] = os.pathsep.join([vendor_path, root_path, env.get("PYTHONPATH", "")])
    env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    command = [
        blender_python(),
        str(extension_root() / "worker" / "pixal3d_worker.py"),
        "--image",
        image_path,
        "--output",
        output_path,
        "--seed",
        str(seed),
        "--model_path",
        model_path,
        "--device",
        device,
    ]
    if not enable_mps_fallback:
        command.append("--disable_mps_fallback")
    return subprocess.run(command, capture_output=True, text=True, check=False, env=env)
