from __future__ import annotations

import importlib
import sys

from . import auto_load
from . import dependency_manager


dependency_manager.ensure_runtime_paths()
auto_load.set_excludes(("__init__", "auto_load", "__pycache__", "pixal3d", "tools", "worker"))
auto_load.set_modules(
    [
        "dependency_manager",
        "utils",
        "properties",
        "preferences",
        "ops_dependencies",
        "ops_generation",
        "ui_panels",
    ]
)


def register() -> None:
    auto_load.register()

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "register_properties"):
        props_mod.register_properties()

    ui_mod = auto_load.get_module("ui_panels")
    if ui_mod and hasattr(ui_mod, "register_menus"):
        ui_mod.register_menus()

    ops_mod = auto_load.get_module("ops_generation")
    if ops_mod and hasattr(ops_mod, "register_webview_import_timer"):
        ops_mod.register_webview_import_timer()


def unregister() -> None:
    ui_mod = auto_load.get_module("ui_panels")
    if ui_mod and hasattr(ui_mod, "unregister_menus"):
        try:
            ui_mod.unregister_menus()
        except Exception as error:
            print(f"Beyond Pixal3D: failed to unregister menus: {error}")

    ops_mod = auto_load.get_module("ops_generation")
    if ops_mod and hasattr(ops_mod, "unregister_webview_import_timer"):
        try:
            ops_mod.unregister_webview_import_timer()
        except Exception as error:
            print(f"Beyond Pixal3D: failed to unregister webview import timer: {error}")

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "unregister_properties"):
        try:
            props_mod.unregister_properties()
        except Exception as error:
            print(f"Beyond Pixal3D: failed to unregister properties: {error}")

    auto_load.unregister()

    try:
        pkg = __package__
        to_delete = [name for name in list(sys.modules.keys()) if name == pkg or name.startswith(pkg + ".")]
        for name in to_delete:
            del sys.modules[name]
        importlib.invalidate_caches()
    except Exception as error:
        print(f"Beyond Pixal3D: sys.modules purge warning: {error}")
