from __future__ import annotations

import bpy
from bpy.types import Operator

from .dependency_manager import (
    get_install_progress,
    get_runtime_status,
    install_bundled_wheels_async,
    prepare_open_model_assets,
)

_INSTALL_PROGRESS_TIMER_RUNNING = False


def _tag_redraw() -> None:
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in getattr(wm, "windows", []):
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in getattr(screen, "areas", []):
            if getattr(area, "type", "") in {"VIEW_3D", "PREFERENCES"}:
                area.tag_redraw()


def _install_progress_timer():
    if not _INSTALL_PROGRESS_TIMER_RUNNING:
        return None
    progress = get_install_progress()
    if progress["running"]:
        _tag_redraw()
        return 0.5
    return 2.0


def register_install_progress_timer() -> None:
    global _INSTALL_PROGRESS_TIMER_RUNNING
    _INSTALL_PROGRESS_TIMER_RUNNING = True
    if not bpy.app.timers.is_registered(_install_progress_timer):
        bpy.app.timers.register(_install_progress_timer, first_interval=0.5, persistent=True)


def unregister_install_progress_timer() -> None:
    global _INSTALL_PROGRESS_TIMER_RUNNING
    _INSTALL_PROGRESS_TIMER_RUNNING = False
    if bpy.app.timers.is_registered(_install_progress_timer):
        try:
            bpy.app.timers.unregister(_install_progress_timer)
        except Exception:
            pass


class BEYONDPIXAL3D_OT_install_bundled_wheels(Operator):
    bl_idname = "beyond_pixal3d.install_bundled_wheels"
    bl_label = "Install Bundled Wheels"
    bl_description = "Download/cache platform wheels and install them into this extension's local _vendor folder"
    bl_options = {"REGISTER"}

    def execute(self, context):
        ok, message = install_bundled_wheels_async()
        register_install_progress_timer()
        self.report({"INFO"} if ok else {"ERROR"}, message[:900])
        return {"FINISHED" if ok else "CANCELLED"}


class BEYONDPIXAL3D_OT_refresh_runtime_status(Operator):
    bl_idname = "beyond_pixal3d.refresh_runtime_status"
    bl_label = "Refresh Pixal3D Runtime"
    bl_description = "Refresh Pixal3D dependency status"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = getattr(getattr(context, "scene", None), "beyond_pixal3d", None)
        requested_device = getattr(props, "device", "auto")
        status = get_runtime_status(requested_device)
        if status.generation_ready:
            self.report({"INFO"}, "All Pixal3D dependencies are ready.")
        elif status.webview_ready:
            missing = ", ".join(status.missing_generation_modules[:8])
            suffix = "..." if len(status.missing_generation_modules) > 8 else ""
            self.report({"WARNING"}, f"Pywebview is ready; missing generation dependencies: {missing}{suffix}")
        else:
            missing = ", ".join((status.missing_webview_modules + status.missing_generation_modules)[:8])
            suffix = "..." if len(status.missing_webview_modules) + len(status.missing_generation_modules) > 8 else ""
            self.report({"WARNING"}, f"Pixal3D dependencies are incomplete: {missing}{suffix}")
        return {"FINISHED"}


class BEYONDPIXAL3D_OT_prepare_open_model_assets(Operator):
    bl_idname = "beyond_pixal3d.prepare_open_model_assets"
    bl_label = "Prepare Open Model Assets"
    bl_description = "Download/cache the open Hugging Face model assets used by the macOS Metal Pixal3D runtime"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = getattr(getattr(context, "scene", None), "beyond_pixal3d", None)
        requested_device = getattr(props, "device", "auto")
        ok, message = prepare_open_model_assets(requested_device)
        self.report({"INFO"} if ok else {"ERROR"}, message[:900])
        return {"FINISHED" if ok else "CANCELLED"}
