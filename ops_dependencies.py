from __future__ import annotations

import bpy
from bpy.types import Operator

from .dependency_manager import get_runtime_status, install_bundled_wheels, prepare_open_model_assets


class BEYONDPIXAL3D_OT_install_bundled_wheels(Operator):
    bl_idname = "beyond_pixal3d.install_bundled_wheels"
    bl_label = "Install Bundled Wheels"
    bl_description = "Install wheels from this extension's wheels folder into its local _vendor folder"
    bl_options = {"REGISTER"}

    def execute(self, context):
        ok, message = install_bundled_wheels()
        self.report({"INFO"} if ok else {"ERROR"}, message[:900])
        return {"FINISHED" if ok else "CANCELLED"}


class BEYONDPIXAL3D_OT_refresh_runtime_status(Operator):
    bl_idname = "beyond_pixal3d.refresh_runtime_status"
    bl_label = "Refresh Pixal3D Runtime"
    bl_description = "Refresh Pixal3D dependency status"
    bl_options = {"REGISTER"}

    def execute(self, context):
        status = get_runtime_status()
        if status.generation_ready:
            self.report({"INFO"}, "Pixal3D generation runtime is ready.")
        elif status.webview_ready:
            self.report({"WARNING"}, "Pywebview is ready, but the Pixal3D generation stack is incomplete.")
        else:
            self.report({"WARNING"}, "Pixal3D dependencies are incomplete.")
        return {"FINISHED"}


class BEYONDPIXAL3D_OT_prepare_open_model_assets(Operator):
    bl_idname = "beyond_pixal3d.prepare_open_model_assets"
    bl_label = "Prepare Open Model Assets"
    bl_description = "Download/cache the open Hugging Face model assets used by the macOS Metal Pixal3D runtime"
    bl_options = {"REGISTER"}

    def execute(self, context):
        ok, message = prepare_open_model_assets()
        self.report({"INFO"} if ok else {"ERROR"}, message[:900])
        return {"FINISHED" if ok else "CANCELLED"}
