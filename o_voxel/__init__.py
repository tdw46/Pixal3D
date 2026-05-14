"""macOS-compatible o_voxel shim for Pixal3D.

This package provides the subset required by TRELLIS.2/Pixal3D shape decoding
inside Blender's macOS Python runtime. The CUDA package is still preferred on
CUDA systems when installed ahead of this extension on sys.path.
"""

from . import postprocess  # noqa: F401
