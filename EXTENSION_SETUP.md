# Beyond Pixal3D Blender Extension

This repo is installed into Blender 5.0 as a user extension through:

`~/Library/Application Support/Blender/5.0/extensions/user_default/beyond_pixal3d`

That path is a symlink to this working tree, so edits in this repo are immediately visible to
Blender after disabling/enabling the add-on.

## Entry Points

- 3D Viewport `N` panel: `Pixal3D`
- 3D Viewport header popover: `Pixal3D`
- `Shift+A`: `Beyond Pixal3D`
- 3D Viewport Object menu: `Beyond Pixal3D`
- File > Import: `Pixal3D Generated GLB`

## Runtime Layout

- `wheels/`: bundled wheels, including pywebview and macOS pyobjc support
- `_vendor/`: extension-local wheel install target
- `webview_app.py`: pywebview helper process
- `worker/pixal3d_worker.py`: subprocess entry point for Pixal3D inference
- `tools/vendor_wheels.py`: wheel download/install helper

## CUDA And Metal Runtime Reality

Pixal3D `main` follows Trellis.2. The upstream Trellis.2 README currently documents Linux,
NVIDIA GPU, CUDA Toolkit 12.4, Python 3.10, PyTorch 2.6.0 CUDA, and compiled CUDA extensions.
That CUDA path is still preserved. On CUDA, the extension expects the upstream CUDA wheels for
`natten`, `flash_attn`, `flex_gemm`, `cumesh`, `o_voxel`, `nvdiffrast`, and
`nvdiffrec_render`.

The macOS path is separate. On Apple Silicon, the extension now uses:

- PyTorch MPS / Metal (`device=mps`)
- `PYTORCH_ENABLE_MPS_FALLBACK=1`
- PyTorch SDPA attention instead of FlashAttention
- A dense PyTorch fallback for sparse 3D convolutions
- Plain GLB export if the CUDA-only `o_voxel` PBR exporter is unavailable

The Metal path keeps Blender alive and lets the model stack load on macOS, but it is experimental
and will be much slower than the CUDA path. PBR extraction through `o_voxel` remains CUDA-only
unless a compatible Metal implementation becomes available.
