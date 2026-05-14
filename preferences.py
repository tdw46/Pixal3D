from __future__ import annotations

import bpy
from bpy.types import AddonPreferences, Context

from .dependency_manager import get_runtime_status
from .utils import wrap_text_to_panel


def get_preferences(context: Context | None = None) -> "BeyondPixal3DPreferences | None":
    ctx = context or bpy.context
    addon = ctx.preferences.addons.get(__package__)
    if not addon:
        return None
    prefs = addon.preferences
    if isinstance(prefs, BeyondPixal3DPreferences):
        return prefs
    return None


class BeyondPixal3DPreferences(AddonPreferences):
    bl_idname = __package__

    def draw(self, context: Context) -> None:
        layout = self.layout
        status = get_runtime_status()

        web_box = layout.box()
        web_box.label(
            text="Pywebview Ready" if status.webview_ready else "Pywebview Not Installed",
            icon="CHECKMARK" if status.webview_ready else "ERROR",
        )
        web_box.label(text=f"Python: {status.python_executable}")
        web_box.label(text=f"Platform: {status.platform_key}")
        web_box.label(text=f"Vendor: {status.vendor_dir}")
        web_box.operator("beyond_pixal3d.install_bundled_wheels", icon="IMPORT")
        web_box.operator("beyond_pixal3d.prepare_open_model_assets", icon="FILE_REFRESH")

        gen_box = layout.box()
        gen_box.label(
            text="Generation Runtime Ready" if status.generation_ready else "Generation Runtime Needs Attention",
            icon="CHECKMARK" if status.generation_ready else "ERROR",
        )
        if status.missing_generation_modules:
            wrapped = wrap_text_to_panel(
                "Missing modules: " + ", ".join(status.missing_generation_modules),
                context,
                full_width=True,
            )
            for line in wrapped.splitlines() or [""]:
                gen_box.label(text=line)
        for note in status.unsupported_notes:
            for line in wrap_text_to_panel(note, context, full_width=True).splitlines() or [""]:
                row = gen_box.row()
                row.alert = True
                row.label(text=line, icon="ERROR")
        gen_box.operator("beyond_pixal3d.refresh_runtime_status", icon="FILE_REFRESH")
