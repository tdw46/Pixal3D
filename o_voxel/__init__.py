"""Portable o_voxel shim for Pixal3D.

This package provides the subset required by TRELLIS.2/Pixal3D shape decoding
inside Blender runtimes where the native CUDA package is unavailable or only
partially importable. The CUDA package is still preferred on CUDA systems when
installed ahead of this extension on sys.path.
"""

__all__ = ("convert", "postprocess")
