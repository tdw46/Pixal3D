from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import bpy
from bpy.props import StringProperty
from bpy.types import Context, Operator
from mathutils import Vector

from .dependency_manager import blender_python, extension_root, get_runtime_status, vendor_dir
from .utils import wrap_text_to_panel

_AUTO_IMPORT_RUNNING = False
_AUTO_IMPORT_TOKEN = ""


def _settings(context: Context):
    return context.scene.beyond_pixal3d


def _correct_imported_meshes(objects) -> None:
    mesh_objects = [obj for obj in objects if obj.type == "MESH" and obj.data is not None]
    if not mesh_objects:
        return

    for obj in mesh_objects:
        obj.rotation_mode = "XYZ"
        obj.rotation_euler.rotate_axis("X", math.pi)

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
    if state.get("import_requested") and filepath and token != _AUTO_IMPORT_TOKEN and Path(filepath).is_file():
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
    existing_path = str(state.get("last_output_path") or "")
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
        status = get_runtime_status(props.device)
        if not status.webview_ready:
            self.report({"ERROR"}, "Install bundled pywebview wheels first.")
            return {"CANCELLED"}

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(vendor_dir()), str(extension_root()), env.get("PYTHONPATH", "")]
        )
        command = [
            blender_python(),
            str(extension_root() / "webview_app.py"),
            "--extension-root",
            str(extension_root()),
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
    status = get_runtime_status("auto")

    action_row = layout.row(align=True)
    action_row.operator("beyond_pixal3d.open_studio", icon="WINDOW")
    action_row.operator("beyond_pixal3d.refresh_runtime_status", text="", icon="FILE_REFRESH")

    status_box = layout.box()
    status_box.label(text="Webview: ready" if status.webview_ready else "Webview: unavailable", icon="WINDOW")
    status_box.label(text="Generation: ready" if status.generation_ready else "Generation: unavailable", icon="MODIFIER")
    status_box.label(text=f"Backend: {status.platform_key}", icon="SYSTEM")

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
        warning.operator("beyond_pixal3d.install_bundled_wheels", icon="IMPORT")

    if not status.generation_ready:
        runtime_box = layout.box()
        runtime_box.alert = True
        runtime_box.label(text="Generation runtime unavailable", icon="ERROR")
        messages = status.unsupported_notes or [
            "Missing modules: " + ", ".join(status.missing_generation_modules)
        ]
        for message in messages:
            for line in wrap_text_to_panel(message, context, full_width=True).splitlines() or [""]:
                runtime_box.label(text=line)
