from __future__ import annotations

import importlib.machinery
import functools
import json
import os
import platform
import queue
import shutil
import site
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None


STATE_VERSION = 1
INSTALL_PROGRESS_STALE_SECONDS = 180

WEBVIEW_MODULES = ("webview",)

COMMON_GENERATION_MODULES = (
    "torch",
    "torchvision",
    "PIL",
    "cv2",
    "numpy",
    "diffusers",
    "accelerate",
    "transformers",
    "trimesh",
    "plyfile",
    "zstandard",
    "kornia",
    "timm",
    "einops",
    "moge",
    "utils3d",
)

CUDA_GENERATION_MODULES = (
    *COMMON_GENERATION_MODULES,
    "natten",
    "o_voxel",
    "cumesh",
    "flex_gemm",
    "triton",
    "flash_attn",
    "nvdiffrast",
    "nvdiffrec_render",
)

CUDA_ONLY_MODULES = (
    "natten",
    "o_voxel",
    "cumesh",
    "flex_gemm",
    "triton",
    "flash_attn",
    "nvdiffrast",
    "nvdiffrec_render",
)

WINDOWS_CUDA_OPTIONAL_MISSING_MODULES: tuple[str, ...] = ()

METAL_GENERATION_MODULES = (
    *COMMON_GENERATION_MODULES,
    "fast_simplification",
    "xatlas",
    "natten_mps",
    "o_voxel.convert",
)

WEBVIEW_PACKAGES = ("pywebview==3.4",)

WEBVIEW_DARWIN_PACKAGES = (
    "pyobjc-core==11.1",
    "pyobjc-framework-Cocoa==11.1",
    "pyobjc-framework-WebKit==11.1",
    "cffi>=1.17",
    "pycparser==2.23",
)
WEBVIEW_WINDOWS_PACKAGES = (
    "pywebview==5.4",
    "pythonnet==3.0.5",
)
WEBVIEW_WINDOWS_REQUIRED_DISTRIBUTIONS = {
    "pywebview": "5.4",
    "pythonnet": "3.0.5",
}

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

WINDOWS_CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
WINDOWS_CUDA_CACHE_PROFILE = "windows-cuda-torch270-cu128"
WINDOWS_CUDA_TORCH_PACKAGES = ("torch==2.7.0", "torchvision==0.22.0")
WINDOWS_CUDA_PACKAGES = (
    "pillow==12.0.0",
    "numpy==2.2.6",
    "matplotlib",
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
    "einops==0.8.1",
    "diffusers==0.37.1",
    "accelerate==1.13.0",
    "triton-windows==3.2.0.post21",
    "plyfile==1.1.3",
    "gradio==6.0.1",
    "utils3d @ https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl",
)
WINDOWS_CUDA_RECOMMENDED_PACKAGES = (
    "hf_xet==1.5.0",
)
WINDOWS_CUDA_RECOMMENDED_MODULES = (
    "hf_xet",
)
WINDOWS_CUDA_WHEEL_SOURCES = (
    "git+https://github.com/microsoft/MoGe.git",
    "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6",
)
WINDOWS_CUDA_NATIVE_BASE_URL = "https://raw.githubusercontent.com/visualbruno/ComfyUI-Trellis2/main/wheels/Windows/Torch270"
WINDOWS_CUDA_FLASH_ATTN_URL = "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4.post1%2Bcu128torch2.7.0cxx11abiFALSE-cp311-cp311-win_amd64.whl"
WINDOWS_CUDA_NATTEN_BASE_URL = "https://huggingface.co/lldacing/NATTEN-windows/resolve/main"
WINDOWS_CUDA_KEEP_PREFIXES = (
    "numpy-2.2.6",
    "pillow-12.0.0",
    "torch-2.7.0+cu128",
    "torchvision-0.22.0+cu128",
    "triton_windows-3.2.0.post21",
    "natten-0.17.5+torch270cu128",
    "cumesh-1.0",
    "flex_gemm-0.0.1",
    "nvdiffrast-0.4.0",
    "nvdiffrec_render-0.0.0",
    "o_voxel-0.0.1",
    "flash_attn-2.7.4.post1",
)
WINDOWS_CUDA_CONTROLLED_PREFIXES = (
    "numpy-",
    "pillow-",
    "torch-",
    "torchvision-",
    "triton_windows-",
    "natten-",
    "cumesh-",
    "flex_gemm-",
    "nvdiffrast-",
    "nvdiffrec_render-",
    "o_voxel-",
    "flash_attn-",
    "flash_attn_3-",
)
WINDOWS_CUDA_CONSTRAINTS = (
    "torch==2.7.0",
    "torchvision==0.22.0",
    "triton-windows==3.2.0.post21",
    "numpy==2.2.6",
    "pillow==12.0.0",
)
WINDOWS_CUDA_REQUIRED_DISTRIBUTIONS = {
    "torch": "2.7.0",
    "torchvision": "0.22.0",
    "triton-windows": "3.2.0",
    "natten": "0.17.5",
    "flash-attn": "2.7.4",
}

OPEN_MODEL_ASSET_REPOS = (
    "TencentARC/Pixal3D",
    "ZhengPeng7/BiRefNet",
    "Ruicheng/moge-2-vitl",
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
)
NAF_CHECKPOINT_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_CHECKPOINT_NAME = "naf_release.pth"

_INSTALL_THREAD: threading.Thread | None = None
_RUNTIME_STATUS_CACHE: dict[str, RuntimeStatus] = {}
_WINDOWS_DLL_DIRECTORY_HANDLES: list[object] = []
_WINDOWS_DLL_DIRECTORY_VALUES: set[str] = set()


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


def windows_cuda_wheels_cache_dir() -> Path:
    return wheels_dir() / "cache_windows_cuda"


def windows_webview_wheels_cache_dir() -> Path:
    return wheels_dir() / "cache_windows_webview"


def windows_cuda_cache_profile_path() -> Path:
    return windows_cuda_wheels_cache_dir() / "profile.txt"


def install_log_path() -> Path:
    return wheels_dir() / "install.log"


def install_state_path() -> Path:
    return wheels_dir() / "state.json"


def install_progress_path() -> Path:
    return wheels_dir() / "install_progress.json"


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
    configure_windows_triton_environment(os.environ)
    configure_windows_native_dll_directories()
    patch_windows_triton_python_dev_paths()


def configure_windows_native_dll_directories() -> None:
    if platform.system().lower() != "windows" or not hasattr(os, "add_dll_directory"):
        return
    dll_dirs = (
        vendor_dir(),
        vendor_dir() / "torch" / "lib",
        vendor_dir() / "triton" / "backends" / "nvidia" / "bin",
    )
    existing_paths = [entry.lower() for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    for dll_dir in dll_dirs:
        if not dll_dir.is_dir():
            continue
        dll_value = str(dll_dir)
        if dll_value.lower() not in existing_paths:
            os.environ["PATH"] = dll_value + os.pathsep + os.environ.get("PATH", "")
            existing_paths.insert(0, dll_value.lower())
        if dll_value.lower() in _WINDOWS_DLL_DIRECTORY_VALUES:
            continue
        try:
            _WINDOWS_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_value))
            _WINDOWS_DLL_DIRECTORY_VALUES.add(dll_value.lower())
        except OSError:
            pass


def configure_windows_triton_environment(env) -> None:
    if platform.system().lower() != "windows":
        return
    triton_root = vendor_dir() / "triton"
    bundled_cc = triton_root / "runtime" / "tcc" / "tcc.exe"
    if bundled_cc.is_file():
        env.setdefault("CC", str(bundled_cc))
    bundled_cuda = triton_root / "backends" / "nvidia"
    if (
        (bundled_cuda / "bin" / "ptxas.exe").is_file()
        and (bundled_cuda / "include" / "cuda.h").is_file()
        and (bundled_cuda / "lib" / "x64" / "cuda.lib").is_file()
    ):
        env.setdefault("CUDA_PATH", str(bundled_cuda))
        env.setdefault("CUDA_HOME", str(bundled_cuda))
        cuda_bin = str(bundled_cuda / "bin")
        path_entries = [entry.lower() for entry in env.get("PATH", "").split(os.pathsep) if entry]
        if cuda_bin.lower() not in path_entries:
            env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    python_include, python_lib = windows_python_dev_paths()
    if python_include and python_lib:
        env.setdefault("PIXAL3D_TRITON_PYTHON_INCLUDE", str(python_include))
        env.setdefault("PIXAL3D_TRITON_PYTHON_LIB", str(python_lib))


def windows_python_dev_paths() -> tuple[Path | None, Path | None]:
    if platform.system().lower() != "windows":
        return None, None
    roots: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(Path(local_app_data) / "Programs" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}")
    roots.extend(
        [
            Path.home() / "AppData" / "Local" / "Programs" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}",
            Path(f"C:/Python{sys.version_info.major}{sys.version_info.minor}"),
        ]
    )
    for root in roots:
        include_dir = root / "include"
        lib_dir = root / "libs"
        if (include_dir / "Python.h").is_file() and (lib_dir / "python3.lib").is_file():
            return include_dir, lib_dir
    return None, None


def patch_windows_triton_python_dev_paths() -> None:
    if platform.system().lower() != "windows":
        return
    include_value = os.environ.get("PIXAL3D_TRITON_PYTHON_INCLUDE")
    lib_value = os.environ.get("PIXAL3D_TRITON_PYTHON_LIB")
    if not include_value or not lib_value:
        return
    import sysconfig

    if not getattr(sysconfig, "_beyond_pixal3d_triton_patched", False):
        original_get_paths = sysconfig.get_paths

        def patched_get_paths(*args, **kwargs):
            paths = dict(original_get_paths(*args, **kwargs))
            paths["include"] = include_value
            return paths

        sysconfig.get_paths = patched_get_paths
        sysconfig._beyond_pixal3d_triton_patched = True
    try:
        import triton.windows_utils as windows_utils

        @functools.cache
        def find_python_override() -> list[str]:
            return [lib_value]

        windows_utils.find_python = find_python_override
        build_module = sys.modules.get("triton.runtime.build")
        if build_module is not None:
            build_module.find_python = find_python_override
    except Exception:
        pass


def ensure_runtime_directories() -> None:
    vendor_dir().mkdir(parents=True, exist_ok=True)
    wheels_dir().mkdir(parents=True, exist_ok=True)
    wheels_cache_dir().mkdir(parents=True, exist_ok=True)
    windows_cuda_wheels_cache_dir().mkdir(parents=True, exist_ok=True)
    windows_webview_wheels_cache_dir().mkdir(parents=True, exist_ok=True)


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


def _status_to_snapshot(status: RuntimeStatus, requested_device: str) -> dict:
    return {
        "requested_device": requested_device,
        "platform_key": status.platform_key,
        "webview_ready": status.webview_ready,
        "generation_ready": status.generation_ready,
        "missing_webview_modules": status.missing_webview_modules,
        "missing_generation_modules": status.missing_generation_modules,
        "python_executable": status.python_executable,
        "vendor_dir": str(status.vendor_dir),
        "wheels_dir": str(status.wheels_dir),
        "install_log_path": str(status.install_log_path),
        "state_path": str(status.state_path),
        "last_updated": status.last_updated,
        "last_error": status.last_error,
        "unsupported_notes": status.unsupported_notes,
    }


def _status_from_snapshot(snapshot: dict, requested_device: str) -> RuntimeStatus | None:
    if not snapshot or snapshot.get("platform_key") != platform_key():
        return None
    if snapshot.get("requested_device") != requested_device:
        return None
    try:
        return RuntimeStatus(
            webview_ready=bool(snapshot.get("webview_ready", False)),
            generation_ready=bool(snapshot.get("generation_ready", False)),
            missing_webview_modules=list(snapshot.get("missing_webview_modules", [])),
            missing_generation_modules=list(snapshot.get("missing_generation_modules", [])),
            platform_key=str(snapshot.get("platform_key", platform_key())),
            python_executable=str(snapshot.get("python_executable", blender_python())),
            vendor_dir=Path(str(snapshot.get("vendor_dir", vendor_dir()))),
            wheels_dir=Path(str(snapshot.get("wheels_dir", wheels_dir()))),
            install_log_path=Path(str(snapshot.get("install_log_path", install_log_path()))),
            state_path=Path(str(snapshot.get("state_path", install_state_path()))),
            last_updated=str(snapshot.get("last_updated", "")),
            last_error=str(snapshot.get("last_error", "")),
            unsupported_notes=list(snapshot.get("unsupported_notes", [])),
        )
    except Exception:
        return None


def _unchecked_runtime_status(requested_device: str) -> RuntimeStatus:
    state = _read_state()
    return RuntimeStatus(
        webview_ready=False,
        generation_ready=False,
        missing_webview_modules=[],
        missing_generation_modules=[],
        platform_key=platform_key(),
        python_executable=blender_python(),
        vendor_dir=vendor_dir(),
        wheels_dir=wheels_dir(),
        install_log_path=install_log_path(),
        state_path=install_state_path(),
        last_updated=str(state.get("last_updated", "")),
        last_error=str(state.get("last_error", "")),
        unsupported_notes=["Runtime status has not been refreshed in this Blender session. Press Refresh Pixal3D Runtime."],
    )


def get_cached_runtime_status(requested_device: str = "auto") -> RuntimeStatus:
    key = requested_device or "auto"
    cached = _RUNTIME_STATUS_CACHE.get(key)
    if cached is not None:
        return cached
    state = _read_state()
    snapshot = state.get("runtime_status") if isinstance(state, dict) else None
    from_snapshot = _status_from_snapshot(snapshot if isinstance(snapshot, dict) else {}, key)
    if from_snapshot is not None:
        _RUNTIME_STATUS_CACHE[key] = from_snapshot
        return from_snapshot
    unchecked = _unchecked_runtime_status(key)
    _RUNTIME_STATUS_CACHE[key] = unchecked
    return unchecked


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_install_progress() -> dict:
    path = install_progress_path()
    if not path.is_file():
        return {"running": False, "progress": 0.0, "stage": "", "message": "", "last_error": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"running": False, "progress": 0.0, "stage": "", "message": "", "last_error": ""}
    data["running"] = bool(data.get("running", False))
    data["progress"] = max(0.0, min(1.0, float(data.get("progress", 0.0) or 0.0)))
    data["stage"] = str(data.get("stage", ""))
    data["message"] = str(data.get("message", ""))
    data["last_error"] = str(data.get("last_error", ""))
    if data["running"]:
        progress_pid = int(data.get("pid") or -1)
        same_process = progress_pid == os.getpid()
        last_updated = _parse_iso_datetime(str(data.get("last_updated", "")))
        heartbeat_age = (
            datetime.now(timezone.utc) - last_updated
        ).total_seconds() if last_updated is not None else INSTALL_PROGRESS_STALE_SECONDS + 1
        thread_alive = _INSTALL_THREAD is not None and _INSTALL_THREAD.is_alive()
        if not same_process or (not thread_alive and heartbeat_age > INSTALL_PROGRESS_STALE_SECONDS):
            data["running"] = False
            data["stage"] = "Install interrupted"
            data["message"] = ""
            data["last_error"] = (
                "The previous Blender session closed during dependency install. "
                "Press the install button again to resume from cached wheels."
            )
            data["completed_at"] = datetime.now(timezone.utc).isoformat()
            data["last_updated"] = data["completed_at"]
            try:
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass
    return data


def _write_install_progress(
    *,
    running: bool,
    progress: float,
    stage: str,
    message: str = "",
    last_error: str = "",
) -> None:
    ensure_runtime_directories()
    current = get_install_progress()
    data = {
        "running": running,
        "progress": max(0.0, min(1.0, float(progress))),
        "stage": stage,
        "message": message,
        "last_error": last_error,
        "pid": os.getpid(),
        "started_at": current.get("started_at") if current.get("running") else datetime.now(timezone.utc).isoformat(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    if not running:
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
    install_progress_path().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _console(message: str) -> None:
    print(f"Beyond Pixal3D install: {message}", flush=True)


def _module_is_available(module_name: str, *, allow_local: bool = True) -> bool:
    ensure_runtime_paths()
    search_paths = [str(vendor_dir())] + list(sys.path)
    if not allow_local:
        root_value = str(extension_root())
        search_paths = [value for value in search_paths if value and str(Path(value).resolve()) != root_value]
    try:
        spec = importlib.machinery.PathFinder.find_spec(module_name, search_paths)
    except Exception:
        return False
    if spec is None:
        return False
    if not allow_local:
        root_path = extension_root().resolve()
        vendor_path = vendor_dir().resolve()
        origin = str(spec.origin or "")
        if origin:
            origin_path = Path(origin).resolve()
            if origin_path.is_relative_to(root_path) and not origin_path.is_relative_to(vendor_path):
                return False
        for location in spec.submodule_search_locations or []:
            location_path = Path(location).resolve()
            if location_path.is_relative_to(root_path) and not location_path.is_relative_to(vendor_path):
                return False
    return True


def missing_modules(module_names: tuple[str, ...], *, native_modules: tuple[str, ...] = ()) -> list[str]:
    native_set = set(native_modules)
    return [
        module_name
        for module_name in module_names
        if not _module_is_available(module_name, allow_local=module_name not in native_set)
    ]


def windows_cuda_native_import_failures() -> dict[str, str]:
    if not windows_cuda_runtime_available():
        return {}
    ensure_runtime_paths()
    failures: dict[str, str] = {}
    probes = {
        "nvdiffrast": "nvdiffrast.torch",
        "o_voxel": "o_voxel.postprocess",
        "nvdiffrec_render": "nvdiffrec_render",
        "cumesh": "cumesh",
        "flex_gemm": "flex_gemm",
        "flash_attn": "flash_attn",
    }
    for module_name, import_name in probes.items():
        try:
            __import__(import_name, fromlist=["*"])
        except Exception as error:
            failures[module_name] = str(error)
    return failures


def module_availability(module_names: tuple[str, ...], *, native_modules: tuple[str, ...] = ()) -> dict[str, bool]:
    native_set = set(native_modules)
    return {
        module_name: _module_is_available(module_name, allow_local=module_name not in native_set)
        for module_name in module_names
    }


def bundled_metal_runtime_available() -> bool:
    system = platform.system().lower()
    machine = platform.machine().lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}


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


def open_model_asset_prep_available(requested_device: str = "auto") -> bool:
    return bundled_metal_runtime_available() and resolved_generation_backend(requested_device) == "metal"


def windows_cuda_runtime_available(requested_device: str = "auto") -> bool:
    return platform.system().lower() == "windows" and resolved_generation_backend(requested_device) == "cuda"


def bundled_install_packages() -> tuple[str, ...]:
    if platform.system().lower() == "windows":
        return WEBVIEW_WINDOWS_PACKAGES
    packages = list(WEBVIEW_PACKAGES)
    if platform.system().lower() == "darwin":
        packages.extend(WEBVIEW_DARWIN_PACKAGES)
    if bundled_metal_runtime_available():
        packages.extend(METAL_PACKAGES)
    return tuple(packages)


def webview_modules_for_platform() -> tuple[str, ...]:
    modules = list(WEBVIEW_MODULES)
    if platform.system().lower() == "windows":
        modules.append("clr")
    return tuple(modules)


def webview_version_mismatches() -> list[str]:
    if platform.system().lower() != "windows":
        return []
    mismatches = []
    for distribution_name, expected_version in WEBVIEW_WINDOWS_REQUIRED_DISTRIBUTIONS.items():
        installed_version = _installed_distribution_version(distribution_name)
        if not installed_version:
            mismatches.append(f"{distribution_name}=={expected_version} is not installed")
        elif not installed_version.startswith(expected_version):
            mismatches.append(f"{distribution_name}=={expected_version} is required; installed {installed_version}")
    return mismatches


def bundled_install_label() -> str:
    if windows_cuda_runtime_available():
        return "Install Windows CUDA Wheels"
    if bundled_metal_runtime_available():
        return "Install Bundled Metal Wheels"
    return "Install Bundled Webview Wheels"


def runtime_install_packages(requested_device: str = "auto") -> tuple[str, ...]:
    if windows_cuda_runtime_available(requested_device):
        return (*WINDOWS_CUDA_TORCH_PACKAGES, *WINDOWS_CUDA_PACKAGES, *WINDOWS_CUDA_RECOMMENDED_PACKAGES)
    if bundled_metal_runtime_available() and resolved_generation_backend(requested_device) == "metal":
        return METAL_PACKAGES
    return ()


def generation_modules_for_backend(requested_device: str = "auto") -> tuple[str, ...]:
    backend = resolved_generation_backend(requested_device)
    if backend == "metal":
        return METAL_GENERATION_MODULES
    if backend == "cuda":
        return CUDA_GENERATION_MODULES
    return COMMON_GENERATION_MODULES


def installable_missing_modules(requested_device: str = "auto") -> list[str]:
    backend = resolved_generation_backend(requested_device)
    native_modules = CUDA_ONLY_MODULES if backend == "cuda" else ()
    missing = missing_modules(generation_modules_for_backend(requested_device), native_modules=native_modules)
    if windows_cuda_runtime_available(requested_device):
        return [module_name for module_name in missing if module_name not in WINDOWS_CUDA_OPTIONAL_MISSING_MODULES]
    return missing


def windows_cuda_recommended_missing_modules(requested_device: str = "auto") -> list[str]:
    if not windows_cuda_runtime_available(requested_device):
        return []
    return missing_modules(WINDOWS_CUDA_RECOMMENDED_MODULES)


def _installed_distribution_version(distribution_name: str) -> str:
    ensure_runtime_paths()
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return ""
    except Exception:
        return ""


def windows_cuda_version_mismatches(requested_device: str = "auto") -> list[str]:
    if not windows_cuda_runtime_available(requested_device):
        return []
    mismatches = []
    for distribution_name, expected_version in WINDOWS_CUDA_REQUIRED_DISTRIBUTIONS.items():
        installed_version = _installed_distribution_version(distribution_name)
        if not installed_version:
            mismatches.append(f"{distribution_name}=={expected_version} is not installed")
        elif not installed_version.startswith(expected_version):
            mismatches.append(f"{distribution_name}=={expected_version} is required; installed {installed_version}")
    return mismatches


def dependency_install_needed(requested_device: str = "auto") -> bool:
    if get_install_progress().get("running"):
        return True
    if missing_modules(webview_modules_for_platform()):
        return True
    if webview_version_mismatches():
        return True
    if windows_cuda_runtime_available(requested_device) and windows_cuda_version_mismatches(requested_device):
        return True
    if windows_cuda_runtime_available(requested_device) and windows_cuda_native_import_failures():
        return True
    if windows_cuda_recommended_missing_modules(requested_device):
        return True
    return bool(installable_missing_modules(requested_device))


def unsupported_notes() -> list[str]:
    notes: list[str] = []
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin":
        notes.append(
            "Pixal3D/Trellis.2 upstream inference is documented for Linux with NVIDIA CUDA; "
            "the CUDA extension wheels are not available for this macOS Blender runtime."
        )
    if system not in {"windows"} and sys.version_info[:2] != (3, 10):
        notes.append(
            "The published Hugging Face demo wheels for o_voxel, flex_gemm, cumesh, and flash_attn_3 "
            "target CPython 3.10, while Blender uses CPython "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        notes.append(
            "Apple Metal/MPS can run some torch models, but Pixal3D calls CUDA-only modules and .cuda() paths."
        )
    return notes


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _windows_cuda_native_wheel_urls() -> tuple[str, ...]:
    tag = _python_tag()
    base = WINDOWS_CUDA_NATIVE_BASE_URL
    urls = [
        f"{base}/cumesh-1.0-{tag}-{tag}-win_amd64.whl",
        f"{base}/flex_gemm-0.0.1-{tag}-{tag}-win_amd64.whl",
        f"{base}/nvdiffrast-0.4.0-{tag}-{tag}-win_amd64.whl",
        f"{base}/nvdiffrec_render-0.0.0-{tag}-{tag}-win_amd64.whl",
        f"{base}/o_voxel-0.0.1-{tag}-{tag}-win_amd64.whl",
        WINDOWS_CUDA_FLASH_ATTN_URL,
    ]
    if tag in {"cp310", "cp311", "cp312"}:
        urls.append(f"{WINDOWS_CUDA_NATTEN_BASE_URL}/natten-0.17.5%2Btorch270cu128-{tag}-{tag}-win_amd64.whl")
    return tuple(urls)


def _download_url_to_cache(url: str, cache_dir: Path, log_file, progress_callback=None, progress=0.0, stage="Downloading wheel") -> Path:
    filename = urllib.parse.unquote(url.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1])
    if not filename:
        raise RuntimeError(f"Could not determine filename for {url}")
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        message = f"Cached: {filename}"
        log_file.write(message + "\n")
        log_file.flush()
        _console(message)
        if progress_callback:
            progress_callback(progress, stage, message)
        return target
    log_file.write(f"Downloading {url}\n")
    log_file.flush()
    _console(f"Downloading {filename}")
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        last_update = 0.0
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if progress_callback and now - last_update >= 2.0:
                    last_update = now
                    mb = downloaded / (1024 * 1024)
                    total_text = f" / {total / (1024 * 1024):.1f} MB" if total else " MB"
                    progress_callback(progress, stage, f"{filename}: {mb:.1f}{total_text}")
        tmp_path.replace(target)
    return target


def _run_logged(command: list[str], log_file, progress_callback=None, progress=0.0, stage="Running command") -> subprocess.CompletedProcess[str]:
    command_text = " ".join(command)
    log_file.write("Command: " + command_text + "\n")
    log_file.flush()
    _console(stage)
    _console(command_text)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    output_queue: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    lines: list[str] = []
    last_line = ""
    last_heartbeat = 0.0
    while process.poll() is None or not output_queue.empty():
        try:
            line = output_queue.get(timeout=0.5)
        except queue.Empty:
            line = ""
        if line:
            lines.append(line)
            last_line = line.strip()
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
        now = time.monotonic()
        if progress_callback and now - last_heartbeat >= 5.0:
            last_heartbeat = now
            progress_callback(progress, stage, last_line or "Still working; see Blender console and wheels/install.log.")
    reader_thread.join(timeout=1.0)
    return subprocess.CompletedProcess(command, process.returncode or 0, "".join(lines), "")


def _prune_windows_cuda_cache(log_file) -> None:
    cache_dir = windows_cuda_wheels_cache_dir()
    profile_path = windows_cuda_cache_profile_path()
    cache_profile = profile_path.read_text(encoding="utf-8").strip() if profile_path.is_file() else ""
    if cache_profile != WINDOWS_CUDA_CACHE_PROFILE:
        for path in cache_dir.glob("*.whl"):
            if not path.name.startswith(WINDOWS_CUDA_CONTROLLED_PREFIXES):
                continue
            try:
                path.unlink()
                log_file.write(f"Removed cached wheel from previous Windows CUDA profile: {path.name}\n")
            except Exception as error:
                log_file.write(f"Could not remove previous-profile cached wheel {path.name}: {error}\n")
        profile_path.write_text(WINDOWS_CUDA_CACHE_PROFILE + "\n", encoding="utf-8")
    for path in cache_dir.glob("*.whl"):
        name = path.name
        should_control = name.startswith(WINDOWS_CUDA_CONTROLLED_PREFIXES)
        should_keep = name.startswith(WINDOWS_CUDA_KEEP_PREFIXES)
        too_small = path.stat().st_size < 1024
        if (should_control and not should_keep) or too_small:
            try:
                path.unlink()
                log_file.write(f"Removed stale cached wheel: {name}\n")
            except Exception as error:
                log_file.write(f"Could not remove stale cached wheel {name}: {error}\n")


def _windows_cuda_cached_install_wheels() -> list[str]:
    installed = _installed_vendor_distributions()
    wheels = []
    for path in windows_cuda_wheels_cache_dir().glob("*.whl"):
        name = path.name
        is_controlled = name.startswith(WINDOWS_CUDA_CONTROLLED_PREFIXES)
        if is_controlled and not name.startswith(WINDOWS_CUDA_KEEP_PREFIXES):
            continue
        if path.stat().st_size < 1024:
            continue
        distribution_name = _wheel_distribution_name(path)
        if not is_controlled and distribution_name and distribution_name in installed:
            continue
        wheels.append(str(path))
    return sorted(wheels)


def _normalize_distribution_name(name: str) -> str:
    return name.replace("_", "-").replace(".", "-").lower()


def _installed_vendor_distributions() -> set[str]:
    installed: set[str] = set()
    try:
        distributions = metadata.distributions(path=[str(vendor_dir())])
        for distribution in distributions:
            name = distribution.metadata.get("Name", "")
            if name:
                installed.add(_normalize_distribution_name(name))
    except Exception:
        pass
    return installed


def _wheel_distribution_name(path: Path) -> str:
    try:
        from packaging.utils import parse_wheel_filename

        name, _version, _build, _tags = parse_wheel_filename(path.name)
        return _normalize_distribution_name(str(name))
    except Exception:
        parts = path.name.split("-")
        return _normalize_distribution_name(parts[0]) if parts else ""


def _prune_windows_cuda_vendor(log_file) -> None:
    controlled_names = (
        "torch",
        "torchgen",
        "torchvision",
        "triton",
        "natten",
        "cumesh",
        "flex_gemm",
        "nvdiffrast",
        "nvdiffrec_render",
        "o_voxel",
        "flash_attn",
        "flash_attn_3",
        "flash_attn_interface.py",
    )
    controlled_dist_prefixes = (
        "torch-",
        "torchvision-",
        "triton_windows-",
        "natten-",
        "cumesh-",
        "flex_gemm-",
        "nvdiffrast-",
        "nvdiffrec_render-",
        "o_voxel-",
        "flash_attn-",
        "flash_attn_3-",
    )
    root = vendor_dir().resolve()
    for path in vendor_dir().iterdir():
        name = path.name
        should_remove = name in controlled_names or name.startswith(controlled_dist_prefixes)
        if not should_remove:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            log_file.write(f"Removed stale Windows CUDA vendor package: {name}\n")
            log_file.flush()
        except Exception as error:
            log_file.write(f"Could not remove stale Windows CUDA vendor package {name}: {error}\n")


def _patch_windows_nvdiffrast_torch27_abi(log_file) -> None:
    if platform.system().lower() != "windows":
        return
    binary = vendor_dir() / "_nvdiffrast_c.cp311-win_amd64.pyd"
    if not binary.is_file():
        return
    replacements = (
        (
            b"?toScalarType@TypeMeta@caffe2@@QEBA?AW4ScalarType@c10@@XZ",
            b"?toScalarType@TypeMeta@caffe2@@QEAA?AW4ScalarType@c10@@XZ",
        ),
        (
            b"?SetDevice@cuda@c10@@YA?AW4cudaError@@C_N@Z",
            b"?SetDevice@cuda@c10@@YA?AW4cudaError@@C@Z\x00\x00",
        ),
    )
    try:
        data = binary.read_bytes()
        patched = data
        changed = False
        for old, new in replacements:
            if len(old) != len(new):
                raise ValueError("nvdiffrast ABI replacement sizes do not match")
            if old in patched:
                patched = patched.replace(old, new)
                changed = True
        if not changed:
            return
        backup = binary.with_suffix(binary.suffix + ".bak_abi_patch")
        if not backup.exists():
            backup.write_bytes(data)
        binary.write_bytes(patched)
        log_file.write("Patched nvdiffrast native extension for the Windows Torch 2.7 ABI.\n")
        log_file.flush()
    except Exception as error:
        log_file.write(f"Could not patch nvdiffrast native extension for Torch 2.7 ABI: {error}\n")
        log_file.flush()


def _prune_windows_webview_vendor(log_file) -> None:
    controlled_names = (
        "webview",
        "pythonnet",
        "clr.py",
        "clr_loader",
    )
    controlled_dist_prefixes = (
        "pywebview-",
        "pythonnet-",
        "clr_loader-",
    )
    root = vendor_dir().resolve()
    for path in vendor_dir().iterdir():
        name = path.name
        should_remove = name in controlled_names or name.startswith(controlled_dist_prefixes)
        if not should_remove:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            log_file.write(f"Removed stale Windows webview package: {name}\n")
            log_file.flush()
        except Exception as error:
            log_file.write(f"Could not remove stale Windows webview package {name}: {error}\n")


def _download_windows_cuda_wheels(log_file, progress_callback=None) -> tuple[bool, str]:
    cache_dir = windows_cuda_wheels_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prune_windows_cuda_cache(log_file)
    constraints_path = cache_dir / "constraints_windows_cuda.txt"
    constraints_path.write_text("\n".join(WINDOWS_CUDA_CONSTRAINTS) + "\n", encoding="utf-8")

    commands = (
        (
            [
                blender_python(),
                "-m",
                "pip",
                "download",
                "--dest",
                str(cache_dir),
                "--no-deps",
                "--only-binary=:all:",
                "--index-url",
                WINDOWS_CUDA_TORCH_INDEX,
                "--extra-index-url",
                "https://pypi.org/simple",
                *WINDOWS_CUDA_TORCH_PACKAGES,
            ],
            0.18,
            "Downloading Torch CUDA wheels",
        ),
        (
            [
                blender_python(),
                "-m",
                "pip",
                "download",
                "--dest",
                str(cache_dir),
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "--find-links",
                str(cache_dir),
                "--constraint",
                str(constraints_path),
                *WINDOWS_CUDA_PACKAGES,
                *WINDOWS_CUDA_RECOMMENDED_PACKAGES,
            ],
            0.38,
            "Downloading Pixal3D Python wheels",
        ),
        (
            [
                blender_python(),
                "-m",
                "pip",
                "wheel",
                "--wheel-dir",
                str(cache_dir),
                "--no-deps",
                *WINDOWS_CUDA_WHEEL_SOURCES,
            ],
            0.58,
            "Building/caching MoGe wheels",
        ),
    )
    for command, progress, stage in commands:
        if progress_callback:
            progress_callback(progress, stage, "")
        result = _run_logged(command, log_file, progress_callback=progress_callback, progress=progress, stage=stage)
        if result.returncode != 0:
            return False, result.stdout or "pip failed while downloading Windows CUDA wheels"

    native_urls = _windows_cuda_native_wheel_urls()
    for index, url in enumerate(native_urls, 1):
        progress = 0.58 + (0.18 * index / max(1, len(native_urls)))
        filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        try:
            if progress_callback:
                progress_callback(progress, "Downloading native CUDA wheels", filename)
            _download_url_to_cache(
                url,
                cache_dir,
                log_file,
                progress_callback=progress_callback,
                progress=progress,
                stage="Downloading native CUDA wheels",
            )
        except Exception as error:
            return False, f"Could not download {url}: {error}"

    _prune_windows_cuda_cache(log_file)
    return True, ""


def _install_webview_helpers(log_file, progress_callback=None) -> subprocess.CompletedProcess[str]:
    missing_webview = missing_modules(webview_modules_for_platform())
    result = subprocess.CompletedProcess([], 0, "", "")
    if platform.system().lower() == "windows":
        if "webview" in missing_webview or "clr" in missing_webview or webview_version_mismatches():
            _prune_windows_webview_vendor(log_file)
            command = [
                blender_python(),
                "-m",
                "pip",
                "install",
                "--target",
                str(vendor_dir()),
                "--upgrade",
                *WEBVIEW_WINDOWS_PACKAGES,
            ]
            return _run_logged(
                command,
                log_file,
                progress_callback=progress_callback,
                progress=0.10,
                stage="Installing Windows pywebview backend",
            )
        return result

    if "webview" in missing_webview or platform.system().lower() != "windows":
        packages = list(WEBVIEW_PACKAGES)
        if platform.system().lower() == "darwin":
            packages.extend(WEBVIEW_DARWIN_PACKAGES)
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
            *packages,
        ]
        result = _run_logged(
            command,
            log_file,
            progress_callback=progress_callback,
            progress=0.06,
            stage="Installing bundled helper wheels",
        )
        if result.returncode != 0:
            return result
    return result


def _install_windows_cuda_recommended_helpers(log_file, progress_callback=None) -> subprocess.CompletedProcess[str]:
    missing_recommended = windows_cuda_recommended_missing_modules()
    if not missing_recommended:
        return subprocess.CompletedProcess([], 0, "", "")
    command = [
        blender_python(),
        "-m",
        "pip",
        "install",
        "--target",
        str(vendor_dir()),
        "--upgrade",
        "--no-deps",
        "--only-binary=:all:",
        *WINDOWS_CUDA_RECOMMENDED_PACKAGES,
    ]
    message = "Installing recommended Windows CUDA helper packages: " + ", ".join(WINDOWS_CUDA_RECOMMENDED_PACKAGES)
    log_file.write(message + "\n")
    log_file.flush()
    _console(message)
    return _run_logged(
        command,
        log_file,
        progress_callback=progress_callback,
        progress=0.84,
        stage="Installing recommended Windows CUDA helpers",
    )


def get_runtime_status(requested_device: str = "auto") -> RuntimeStatus:
    ensure_runtime_paths()
    state = _read_state()
    backend = resolved_generation_backend(requested_device)
    required_generation = generation_modules_for_backend(requested_device)
    missing_webview = missing_modules(webview_modules_for_platform())
    webview_mismatches = webview_version_mismatches()
    native_modules = CUDA_ONLY_MODULES if backend == "cuda" else ()
    missing_generation = missing_modules(required_generation, native_modules=native_modules)
    version_mismatches = windows_cuda_version_mismatches(requested_device) if backend == "cuda" else []
    native_import_failures = windows_cuda_native_import_failures() if windows_cuda_runtime_available(requested_device) else {}
    for module_name in native_import_failures:
        if module_name not in missing_generation:
            missing_generation.append(module_name)
    required_missing_generation = missing_generation
    if windows_cuda_runtime_available(requested_device):
        required_missing_generation = [
            module_name for module_name in missing_generation if module_name not in WINDOWS_CUDA_OPTIONAL_MISSING_MODULES
        ]
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
            notes.append("Missing o_voxel. Pixal3D/TRELLIS.2 shape decoding requires o_voxel up front.")
        else:
            notes.append(
                "Metal support uses PyTorch MPS on the Apple GPU, SDPA attention, natten-mps Metal neighborhood "
                "attention for NAF, a Metal sparse-conv compatibility backend, and the bundled pure-Python "
                "o_voxel.convert compatibility layer for shape decoding."
            )
        notes.append(
            "Use Prepare Open Model Assets to cache the open macOS model stack up front: "
            + ", ".join((*OPEN_MODEL_ASSET_REPOS, "valeoai/NAF"))
        )
    generation_ready = not required_missing_generation and not version_mismatches
    if backend == "cuda":
        notes.append(
            "Windows CUDA uses the upstream Pixal3D CUDA runtime where compatible wheels exist for Blender's Python: "
            "natten, o_voxel, cumesh, flex_gemm, triton-windows, flash_attn, nvdiffrast, and nvdiffrec_render."
        )
        if platform.system().lower() == "windows":
            notes.append(
                "Windows CUDA uses the native flash_attn backend from the TRELLIS.2 setup path. If the native wheel "
                "does not import, generation is marked unavailable."
            )
            if native_import_failures:
                notes.append(
                    "Windows CUDA native import failure: "
                    + "; ".join(f"{module_name}: {error}" for module_name, error in native_import_failures.items())
                )
        if version_mismatches:
            notes.append("Windows CUDA runtime version mismatch: " + "; ".join(version_mismatches))
        recommended_missing = windows_cuda_recommended_missing_modules(requested_device)
        if recommended_missing:
            notes.append(
                "Recommended Windows CUDA download helper missing: "
                + ", ".join(recommended_missing)
                + ". Generation can run without it, but Hugging Face downloads may be slower."
            )
        if "natten" in missing_generation:
            notes.append(
                "NATTEN is listed by upstream Pixal3D and is bundled through the Windows Torch 2.7 CUDA profile. "
                "Run Install Windows CUDA Wheels to install natten for the extension-local runtime."
            )
    elif cuda_missing and not notes:
        notes.append("CUDA runtime modules are missing: " + ", ".join(cuda_missing))
    if webview_mismatches:
        notes.append("Windows pywebview backend version mismatch: " + "; ".join(webview_mismatches))
    updated_at = datetime.now(timezone.utc).isoformat()
    status = RuntimeStatus(
        webview_ready=not missing_webview and not webview_mismatches,
        generation_ready=generation_ready,
        missing_webview_modules=missing_webview,
        missing_generation_modules=missing_generation,
        platform_key=platform_key(),
        python_executable=blender_python(),
        vendor_dir=vendor_dir(),
        wheels_dir=wheels_dir(),
        install_log_path=install_log_path(),
        state_path=install_state_path(),
        last_updated=updated_at,
        last_error=str(state.get("last_error", "")),
        unsupported_notes=notes,
    )
    key = requested_device or "auto"
    _RUNTIME_STATUS_CACHE[key] = status
    new_state = dict(state)
    new_state.update(
        {
            "state_version": STATE_VERSION,
            "platform_key": platform_key(),
            "last_updated": updated_at,
            "runtime_status": _status_to_snapshot(status, key),
        }
    )
    _write_state(new_state)
    return status


def install_bundled_wheels(progress_callback=None) -> tuple[bool, str]:
    ensure_runtime_directories()
    if not dependency_install_needed():
        return True, "All installable bundled dependencies are already present."
    if progress_callback:
        progress_callback(0.02, "Preparing wheel install", "")

    helper_needed = bool(missing_modules(webview_modules_for_platform())) or bool(webview_version_mismatches()) or (
        bundled_metal_runtime_available() and bool(installable_missing_modules())
    )
    cuda_core_needed = windows_cuda_runtime_available() and (
        bool(windows_cuda_version_mismatches())
        or bool(installable_missing_modules())
        or bool(windows_cuda_native_import_failures())
    )
    cuda_recommended_needed = bool(windows_cuda_recommended_missing_modules())
    result = subprocess.CompletedProcess([], 0, "", "")
    installed: list[str] = []
    with install_log_path().open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== Install bundled wheels {datetime.now(timezone.utc).isoformat()} ===\n")
        log_file.flush()
        if helper_needed:
            if progress_callback:
                progress_callback(0.06, "Installing bundled helper wheels", "")
            result = _install_webview_helpers(log_file, progress_callback=progress_callback)
            installed.extend(bundled_install_packages())
            if platform.system().lower() == "windows":
                installed.extend(WEBVIEW_WINDOWS_PACKAGES)
        else:
            message = "Bundled helper wheels already installed; skipping helper step."
            log_file.write(message + "\n")
            log_file.flush()
            _console(message)
            if progress_callback:
                progress_callback(0.08, "Skipping bundled helper wheels", message)

        if result.returncode == 0 and cuda_core_needed:
            log_file.write(f"\n=== Cache Windows CUDA wheels {datetime.now(timezone.utc).isoformat()} ===\n")
            log_file.flush()
            ok, error = _download_windows_cuda_wheels(log_file, progress_callback=progress_callback)
            if not ok:
                result = subprocess.CompletedProcess([], 1, error, "")
            else:
                cuda_wheels = _windows_cuda_cached_install_wheels()
                _prune_windows_cuda_vendor(log_file)
                install_command = [
                    blender_python(),
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--find-links",
                    str(windows_cuda_wheels_cache_dir()),
                    "--target",
                    str(vendor_dir()),
                    "--upgrade",
                    *cuda_wheels,
                ]
                log_file.write(f"\n=== Install Windows CUDA wheels {datetime.now(timezone.utc).isoformat()} ===\n")
                log_file.flush()
                if progress_callback:
                    progress_callback(0.82, "Installing Windows CUDA wheels", "")
                result = _run_logged(
                    install_command,
                    log_file,
                    progress_callback=progress_callback,
                    progress=0.82,
                    stage="Installing Windows CUDA wheels",
                )
                if result.returncode == 0:
                    _patch_windows_nvdiffrast_torch27_abi(log_file)
                installed.extend(path.name for path in map(Path, cuda_wheels))
        elif result.returncode == 0 and cuda_recommended_needed:
            log_file.write(f"\n=== Install recommended Windows CUDA helpers {datetime.now(timezone.utc).isoformat()} ===\n")
            log_file.flush()
            result = _install_windows_cuda_recommended_helpers(log_file, progress_callback=progress_callback)
            installed.extend(WINDOWS_CUDA_RECOMMENDED_PACKAGES)
        elif result.returncode == 0 and windows_cuda_runtime_available():
            message = "Windows CUDA wheels already installed; skipping CUDA step."
            log_file.write(message + "\n")
            log_file.flush()
            _console(message)
            if progress_callback:
                progress_callback(0.82, "Skipping Windows CUDA wheels", message)

    state = {
        "state_version": STATE_VERSION,
        "platform_key": platform_key(),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "last_error": "" if result.returncode == 0 else (result.stdout or result.stderr or "pip failed"),
    }
    _write_state(state)
    if result.returncode != 0:
        _write_install_progress(running=False, progress=1.0, stage="Install failed", message="", last_error=state["last_error"])
        if progress_callback:
            progress_callback(1.0, "Install failed", state["last_error"])
        return False, state["last_error"]
    message = f"Installed bundled runtime packages into {vendor_dir()}: {', '.join(installed)}"
    try:
        get_runtime_status()
    except Exception as error:
        _console(f"Installed packages, but runtime status refresh failed: {error}")
    _write_install_progress(running=False, progress=1.0, stage="Install complete", message=message, last_error="")
    if progress_callback:
        progress_callback(1.0, "Install complete", message)
    return True, message


def install_bundled_wheels_async() -> tuple[bool, str]:
    global _INSTALL_THREAD
    current_progress = get_install_progress()
    if current_progress.get("running"):
        return False, "Wheel install is already running."
    if _INSTALL_THREAD is not None and _INSTALL_THREAD.is_alive():
        return False, "Wheel install is already running."
    if not dependency_install_needed():
        _write_install_progress(
            running=False,
            progress=1.0,
            stage="Install skipped",
            message="All installable bundled dependencies are already present.",
            last_error="",
        )
        return True, "All installable bundled dependencies are already present."

    def progress(progress_value: float, stage: str, message: str = "") -> None:
        _write_install_progress(running=True, progress=progress_value, stage=stage, message=message, last_error="")

    def worker() -> None:
        _write_install_progress(running=True, progress=0.0, stage="Starting wheel install", message="", last_error="")
        _console("Wheel install worker started.")
        ok, message = install_bundled_wheels(progress_callback=progress)
        _write_install_progress(
            running=False,
            progress=1.0,
            stage="Install complete" if ok else "Install failed",
            message=message if ok else "",
            last_error="" if ok else message,
        )
        _console(("Wheel install complete. " if ok else "Wheel install failed. ") + message[:500])

    _INSTALL_THREAD = threading.Thread(target=worker, name="BeyondPixal3DWheelInstall", daemon=True)
    _INSTALL_THREAD.start()
    return True, "Wheel install started in the background."


def prepare_open_model_assets(requested_device: str = "auto") -> tuple[bool, str]:
    if not open_model_asset_prep_available(requested_device):
        state = {
            "state_version": STATE_VERSION,
            "platform_key": platform_key(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "last_error": "",
        }
        _write_state(state)
        return True, "Skipped open model asset prep; it is only needed for the bundled macOS Metal runtime."

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


def run_worker(
    image_path: str,
    output_path: str,
    seed: int,
    model_path: str,
    device: str = "auto",
    decimation_target: int = 1000000,
    target_resolution: int = 1536,
    texture_size: int = 4096,
    enable_mps_fallback: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    vendor_path = str(vendor_dir())
    root_path = str(extension_root())
    env["PYTHONPATH"] = os.pathsep.join([vendor_path, root_path, env.get("PYTHONPATH", "")])
    env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    configure_windows_triton_environment(env)
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
        "--decimation_target",
        str(decimation_target),
        "--target_resolution",
        str(target_resolution),
        "--texture_size",
        str(texture_size),
    ]
    if not enable_mps_fallback:
        command.append("--disable_mps_fallback")
    return subprocess.run(command, capture_output=True, text=True, check=False, env=env)
