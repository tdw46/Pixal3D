from __future__ import annotations

import bpy
from bpy.types import Context, Menu, Panel

from .dependency_manager import bundled_install_label, open_model_asset_prep_available
from .ops_generation import draw_generation_controls


class BEYONDPIXAL3D_MT_menu(Menu):
    bl_idname = "BEYONDPIXAL3D_MT_menu"
    bl_label = "Beyond Pixal3D"

    def draw(self, context: Context) -> None:
        layout = self.layout
        layout.operator("beyond_pixal3d.open_studio", icon="WINDOW")
        layout.operator("beyond_pixal3d.import_last_output", icon="IMPORT")
        layout.separator()
        layout.operator("beyond_pixal3d.install_bundled_wheels", text=bundled_install_label(), icon="IMPORT")
        if open_model_asset_prep_available():
            layout.operator("beyond_pixal3d.prepare_open_model_assets", icon="FILE_REFRESH")
        layout.operator("beyond_pixal3d.refresh_runtime_status", icon="FILE_REFRESH")


class BEYONDPIXAL3D_PT_main(Panel):
    bl_idname = "BEYONDPIXAL3D_PT_main"
    bl_label = "Pixal3D"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pixal3D"

    def draw(self, context: Context) -> None:
        draw_generation_controls(self.layout, context)


def _draw_add_menu(self, context: Context) -> None:
    self.layout.separator()
    self.layout.menu(BEYONDPIXAL3D_MT_menu.bl_idname, icon="MESH_MONKEY")


def _draw_object_menu(self, context: Context) -> None:
    self.layout.separator()
    self.layout.menu(BEYONDPIXAL3D_MT_menu.bl_idname, icon="MESH_MONKEY")


def _draw_import_menu(self, context: Context) -> None:
    self.layout.operator("beyond_pixal3d.import_last_output", text="Pixal3D Generated GLB", icon="IMPORT")


def _draw_view3d_header(self, context: Context) -> None:
    row = self.layout.row(align=True)
    row.popover(panel=BEYONDPIXAL3D_PT_main.bl_idname, text="Pixal3D", icon="MESH_MONKEY")


def register_menus() -> None:
    bpy.types.VIEW3D_MT_add.append(_draw_add_menu)
    bpy.types.VIEW3D_MT_object.append(_draw_object_menu)
    bpy.types.TOPBAR_MT_file_import.append(_draw_import_menu)
    bpy.types.VIEW3D_HT_header.append(_draw_view3d_header)


def unregister_menus() -> None:
    for menu, draw_func in (
        (bpy.types.VIEW3D_MT_add, _draw_add_menu),
        (bpy.types.VIEW3D_MT_object, _draw_object_menu),
        (bpy.types.TOPBAR_MT_file_import, _draw_import_menu),
        (bpy.types.VIEW3D_HT_header, _draw_view3d_header),
    ):
        try:
            menu.remove(draw_func)
        except Exception:
            pass
