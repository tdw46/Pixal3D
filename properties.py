from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup, Scene


def _default_output_dir() -> str:
    return str(Path.home() / "Pixal3D_Outputs")


class BEYONDPIXAL3D_PG_settings(PropertyGroup):
    image_path: StringProperty(  # type: ignore[valid-type]
        name="Input Image",
        subtype="FILE_PATH",
        description="Image to convert into a 3D asset",
    )
    output_dir: StringProperty(  # type: ignore[valid-type]
        name="Output Folder",
        subtype="DIR_PATH",
        default=_default_output_dir(),
        description="Folder for generated GLB files",
    )
    output_name: StringProperty(  # type: ignore[valid-type]
        name="Output Name",
        default="pixal3d_asset.glb",
        description="Generated GLB file name",
    )
    model_path: StringProperty(  # type: ignore[valid-type]
        name="Model",
        default="TencentARC/Pixal3D",
        description="Hugging Face model id or local Pixal3D model folder",
    )
    seed: IntProperty(  # type: ignore[valid-type]
        name="Seed",
        default=42,
        min=0,
        description="Generation seed",
    )
    import_after_generate: BoolProperty(  # type: ignore[valid-type]
        name="Import Result",
        default=True,
        description="Import the generated GLB into the current Blender scene",
    )
    device: EnumProperty(  # type: ignore[valid-type]
        name="Device",
        items=(
            ("auto", "Auto", "Prefer CUDA when available, then Apple Metal, then CPU"),
            ("cuda", "CUDA", "Use the default NVIDIA CUDA device"),
            ("mps", "Metal", "Use Apple Metal through PyTorch MPS"),
            ("cpu", "CPU", "Use CPU fallback"),
        ),
        default="auto",
        description="Generation backend device",
    )
    enable_mps_fallback: BoolProperty(  # type: ignore[valid-type]
        name="MPS Fallback",
        default=True,
        description="Allow unsupported Metal operations to fall back to CPU",
    )
    last_output_path: StringProperty(  # type: ignore[valid-type]
        name="Last Output",
        subtype="FILE_PATH",
        description="Most recent generated GLB path",
    )

    def resolved_output_path(self) -> str:
        output_dir = Path(bpy.path.abspath(self.output_dir or _default_output_dir())).expanduser()
        output_name = (self.output_name or "pixal3d_asset.glb").strip()
        if not output_name.lower().endswith(".glb"):
            output_name += ".glb"
        return str(output_dir / output_name)


def register_properties() -> None:
    Scene.beyond_pixal3d = PointerProperty(type=BEYONDPIXAL3D_PG_settings)


def unregister_properties() -> None:
    if hasattr(Scene, "beyond_pixal3d"):
        del Scene.beyond_pixal3d
