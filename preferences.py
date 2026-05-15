from __future__ import annotations

import bpy
from bpy.props import BoolProperty
from bpy.types import AddonPreferences, Context

from .dependency_manager import (
    bundled_install_label,
    get_cached_runtime_status,
    get_install_progress,
    open_model_asset_prep_available,
)
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

    show_install_details: BoolProperty(  # type: ignore[valid-type]
        name="Install Details",
        default=False,
    )
    show_dependency_details: BoolProperty(  # type: ignore[valid-type]
        name="Dependency Details",
        default=False,
    )

    def draw(self, context: Context) -> None:
        layout = self.layout
        props = getattr(getattr(context, "scene", None), "beyond_pixal3d", None)
        requested_device = getattr(props, "device", "auto")
        status = get_cached_runtime_status(requested_device)
        install = get_install_progress()
        can_install = install["running"] or (not status.webview_ready) or (not status.generation_ready)

        web_box = layout.box()
        web_box.label(
            text="Pywebview Ready" if status.webview_ready else "Pywebview Not Installed",
            icon="CHECKMARK" if status.webview_ready else "ERROR",
        )
        web_box.label(text=f"Python: {status.python_executable}")
        web_box.label(text=f"Platform: {status.platform_key}")
        web_box.label(text=f"Vendor: {status.vendor_dir}")
        show_install_action = can_install or install["running"] or bool(status.missing_generation_modules)
        if show_install_action:
            install_row = web_box.row()
            install_row.enabled = not install["running"] and can_install
            install_row.operator("beyond_pixal3d.install_bundled_wheels", text=bundled_install_label(), icon="IMPORT")
            if status.missing_generation_modules and not can_install and not install["running"]:
                web_box.label(text="No bundled installer is available for the remaining missing dependency.", icon="INFO")
        else:
            web_box.label(text="Bundled installable dependencies are present.", icon="CHECKMARK")
        if install["running"] or install["last_error"]:
            text = install["stage"] or ("Install failed" if install["last_error"] else "Install")
            if hasattr(web_box, "progress"):
                web_box.progress(factor=install["progress"], text=text, type="BAR")
            else:
                web_box.label(text=f"{int(install['progress'] * 100)}% - {text}")
            detail = install["message"] or install["last_error"]
            if detail:
                header = web_box.row()
                header.prop(
                    self,
                    "show_install_details",
                    text="Install Details",
                    icon="TRIA_DOWN" if self.show_install_details else "TRIA_RIGHT",
                    emboss=False,
                )
                if self.show_install_details or install["last_error"]:
                    for line in wrap_text_to_panel(detail, context, full_width=True).splitlines() or [""]:
                        row = web_box.row()
                        row.alert = bool(install["last_error"])
                        row.label(text=line, icon="ERROR" if install["last_error"] else "INFO")
        if open_model_asset_prep_available():
            web_box.operator("beyond_pixal3d.prepare_open_model_assets", icon="FILE_REFRESH")

        gen_box = layout.box()
        gen_box.label(
            text="Generation Runtime Ready" if status.generation_ready else "Generation Runtime Needs Attention",
            icon="CHECKMARK" if status.generation_ready else "ERROR",
        )
        if status.missing_generation_modules or status.unsupported_notes:
            gen_box.label(text=f"Missing dependencies: {len(status.missing_generation_modules)}")
            details_row = gen_box.row()
            details_row.prop(
                self,
                "show_dependency_details",
                text="Dependency Details",
                icon="TRIA_DOWN" if self.show_dependency_details else "TRIA_RIGHT",
                emboss=False,
            )
            if self.show_dependency_details:
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
        layout.operator("beyond_pixal3d.refresh_runtime_status", text="Refresh All Dependencies", icon="FILE_REFRESH")
