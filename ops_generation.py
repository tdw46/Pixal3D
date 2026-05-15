from __future__ import annotations

import json
import math
import os
import subprocess
import uuid
from pathlib import Path

import bpy
from bpy.props import StringProperty
from bpy.types import Context, Operator
from mathutils import Vector

from .dependency_manager import (
    blender_python,
    bundled_install_label,
    extension_root,
    get_cached_runtime_status,
    get_install_progress,
    get_runtime_status,
    vendor_dir,
)
from .utils import wrap_text_to_panel

_AUTO_IMPORT_RUNNING = False
_AUTO_IMPORT_TOKEN = ""
_IMPORT_UPRIGHT_X_ROTATION = math.radians(90.0)
_WEBVIEW_SESSION_ID = uuid.uuid4().hex


def _settings(context: Context):
    return context.scene.beyond_pixal3d


def _correct_imported_meshes(objects) -> None:
    mesh_objects = [obj for obj in objects if obj.type == "MESH" and obj.data is not None]
    if not mesh_objects:
        return

    for obj in mesh_objects:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler.rotate_axis("X", _IMPORT_UPRIGHT_X_ROTATION)

    bpy.context.view_layer.update()
    lowest_z = min(
        (obj.matrix_world @ vertex.co).z
        for obj in mesh_objects
        for vertex in obj.data.vertices
    )
    if abs(lowest_z) <= 1.0e-6:
        return

    world_delta = Vector((0.0, 0.0, -lowest_z))
    for obj in mesh_objects:
        local_delta = obj.matrix_world.inverted().to_3x3() @ world_delta
        for vertex in obj.data.vertices:
            vertex.co += local_delta
        obj.data.update()

    bpy.context.view_layer.update()


def _import_glb(filepath: str) -> None:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    _correct_imported_meshes(imported)


def _webview_state_path() -> Path:
    return extension_root() / "wheels" / "webview_state.json"


def _last_webview_output_path() -> str:
    data = _read_webview_state()
    if data.get("session_id") != _WEBVIEW_SESSION_ID:
        return ""
    return str(data.get("last_output_path") or "")


def _read_webview_state() -> dict:
    path = _webview_state_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _auto_import_timer():
    global _AUTO_IMPORT_TOKEN
    if not _AUTO_IMPORT_RUNNING:
        return None

    state = _read_webview_state()
    filepath = str(state.get("last_output_path") or "")
    token = f"{state.get('last_updated') or ''}:{filepath}"
    if (
        state.get("session_id") == _WEBVIEW_SESSION_ID
        and state.get("import_requested")
        and filepath
        and token != _AUTO_IMPORT_TOKEN
        and Path(filepath).is_file()
    ):
        try:
            _import_glb(filepath)
            _AUTO_IMPORT_TOKEN = token
            scene = getattr(bpy.context, "scene", None)
            if scene is not None and hasattr(scene, "beyond_pixal3d"):
                scene.beyond_pixal3d.last_output_path = filepath
        except Exception as error:
            print(f"Beyond Pixal3D: auto-import failed for {filepath}: {error}")

    return 2.0


def register_webview_import_timer() -> None:
    global _AUTO_IMPORT_RUNNING, _AUTO_IMPORT_TOKEN
    state = _read_webview_state()
    existing_path = str(state.get("last_output_path") or "") if state.get("session_id") == _WEBVIEW_SESSION_ID else ""
    _AUTO_IMPORT_TOKEN = f"{state.get('last_updated') or ''}:{existing_path}" if existing_path else ""
    _AUTO_IMPORT_RUNNING = True
    if not bpy.app.timers.is_registered(_auto_import_timer):
        bpy.app.timers.register(_auto_import_timer, first_interval=1.0, persistent=True)


def unregister_webview_import_timer() -> None:
    global _AUTO_IMPORT_RUNNING
    _AUTO_IMPORT_RUNNING = False
    if bpy.app.timers.is_registered(_auto_import_timer):
        try:
            bpy.app.timers.unregister(_auto_import_timer)
        except Exception:
            pass


class BEYONDPIXAL3D_OT_open_studio(Operator):
    bl_idname = "beyond_pixal3d.open_studio"
    bl_label = "Open Pixal3D Studio"
    bl_description = "Open the Pixal3D pywebview helper window"
    bl_options = {"REGISTER"}

    def execute(self, context: Context):
        props = _settings(context)
        status = get_cached_runtime_status(props.device)
        if not status.webview_ready:
            detail = " Press Refresh Pixal3D Runtime if this looks stale." if not status.last_updated else ""
            self.report({"ERROR"}, "Install bundled pywebview wheels first." + detail)
            return {"CANCELLED"}

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(vendor_dir()), str(extension_root()), env.get("PYTHONPATH", "")]
        )
        env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            from .dependency_manager import configure_windows_triton_environment

            configure_windows_triton_environment(env)
        except Exception:
            pass
        command = [
            blender_python(),
            str(extension_root() / "webview_app.py"),
            "--extension-root",
            str(extension_root()),
            "--session-id",
            _WEBVIEW_SESSION_ID,
        ]
        try:
            subprocess.Popen(command, cwd=str(extension_root()), env=env)
        except Exception as error:
            self.report({"ERROR"}, f"Could not open Pixal3D Studio: {error}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Pixal3D Studio opened.")
        return {"FINISHED"}


class BEYONDPIXAL3D_OT_import_last_output(Operator):
    bl_idname = "beyond_pixal3d.import_last_output"
    bl_label = "Import Pixal3D GLB"
    bl_description = "Import the most recent Pixal3D GLB into the scene"
    bl_options = {"REGISTER", "UNDO"}

    filepath: StringProperty(  # type: ignore[valid-type]
        name="GLB Path",
        subtype="FILE_PATH",
    )

    def execute(self, context: Context):
        props = _settings(context)
        filepath = bpy.path.abspath(self.filepath or props.last_output_path or _last_webview_output_path() or "").strip()
        if not filepath or not Path(filepath).is_file():
            self.report({"ERROR"}, "No generated GLB exists to import.")
            return {"CANCELLED"}
        _import_glb(filepath)
        props.last_output_path = filepath
        self.report({"INFO"}, f"Imported {filepath}")
        return {"FINISHED"}


def draw_generation_controls(layout, context: Context) -> None:
    props = _settings(context)
    status = get_cached_runtime_status(props.device)
    install = get_install_progress()
    missing_count = len(status.missing_webview_modules) + len(status.missing_generation_modules)
    status_unchecked = (
        not status.webview_ready
        and not status.generation_ready
        and missing_count == 0
        and any("has not been refreshed" in note for note in status.unsupported_notes)
    )
    can_install = (not status_unchecked) and ((not status.webview_ready) or (not status.generation_ready))

    action_row = layout.row(align=True)
    action_row.operator("beyond_pixal3d.open_studio", icon="WINDOW")
    action_row.operator("beyond_pixal3d.refresh_runtime_status", text="", icon="FILE_REFRESH")

    status_box = layout.box()
    webview_text = "Webview: unknown" if status_unchecked else ("Webview: ready" if status.webview_ready else "Webview: unavailable")
    generation_text = "Generation: unknown" if status_unchecked else ("Generation: ready" if status.generation_ready else "Generation: unavailable")
    status_box.label(text=webview_text, icon="WINDOW")
    status_box.label(text=generation_text, icon="MODIFIER")
    status_box.label(text=f"Backend: {status.platform_key}", icon="SYSTEM")

    if install["running"]:
        progress_box = layout.box()
        progress_box.label(text="Dependency install running", icon="IMPORT")
        if hasattr(progress_box, "progress"):
            progress_box.progress(factor=install["progress"], text=install["stage"], type="BAR")
        else:
            progress_box.label(text=f"{int(install['progress'] * 100)}% - {install['stage']}")

    if status_unchecked:
        refresh_box = layout.box()
        refresh_box.alert = True
        refresh_box.label(text="Runtime status needs refresh", icon="FILE_REFRESH")
        for note in status.unsupported_notes:
            for line in wrap_text_to_panel(note, context, full_width=True).splitlines() or [""]:
                refresh_box.label(text=line)
        refresh_box.operator("beyond_pixal3d.refresh_runtime_status", text="Refresh Pixal3D Runtime", icon="FILE_REFRESH")
        return

    last_output_path = props.last_output_path or _last_webview_output_path()
    if last_output_path:
        last_box = layout.box()
        for line in wrap_text_to_panel(f"Last: {last_output_path}", context, full_width=True).splitlines() or [""]:
            last_box.label(text=line, icon="FILE_3D")
        import_op = last_box.operator("beyond_pixal3d.import_last_output", icon="IMPORT")
        import_op.filepath = last_output_path

    if not status.webview_ready:
        warning = layout.box()
        warning.alert = True
        warning.label(text="Pywebview wheels are not installed.", icon="ERROR")
        if can_install or install["running"]:
            install_row = warning.row()
            install_row.enabled = not install["running"]
            install_row.operator("beyond_pixal3d.install_bundled_wheels", text=bundled_install_label(), icon="IMPORT")

    if not status.generation_ready:
        runtime_box = layout.box()
        runtime_box.alert = True
        runtime_box.label(text="Generation runtime unavailable", icon="ERROR")
        show_install_action = can_install or install["running"] or bool(status.missing_generation_modules)
        if show_install_action:
            install_row = runtime_box.row()
            install_row.enabled = not install["running"] and can_install
            install_row.operator("beyond_pixal3d.install_bundled_wheels", text=bundled_install_label(), icon="IMPORT")
            if status.missing_generation_modules and not can_install and not install["running"]:
                runtime_box.label(text="No bundled installer is available for the remaining missing dependency.", icon="INFO")
        else:
            runtime_box.label(text="Bundled installable dependencies are present.", icon="CHECKMARK")
        runtime_box.label(text=f"Missing dependencies: {len(status.missing_generation_modules)}")
        details_row = runtime_box.row()
        details_row.prop(
            props,
            "show_dependency_details",
            text="Dependency Details",
            icon="TRIA_DOWN" if props.show_dependency_details else "TRIA_RIGHT",
            emboss=False,
        )
        if props.show_dependency_details:
            messages = ["Missing modules: " + ", ".join(status.missing_generation_modules)]
            messages.extend(status.unsupported_notes)
            for message in messages:
                for line in wrap_text_to_panel(message, context, full_width=True).splitlines() or [""]:
                    runtime_box.label(text=line)
