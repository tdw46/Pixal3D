# Pixal3D Dependency Notes

Pixal3D `main` follows the Trellis.2 runtime stack. The upstream install guides require Linux,
an NVIDIA GPU, CUDA Toolkit 12.4, Python 3.10, PyTorch 2.6.0 CUDA wheels, and compiled CUDA
extensions:

- `flash-attn` / `flash_attn_3`
- `cumesh`
- `flex_gemm`
- `o_voxel`
- `nvdiffrast`
- `nvdiffrec_render`
- `natten`
- `utils3d`
- `MoGe`

The Blender 5.0 runtime on this machine is macOS arm64 with CPython 3.11. The published Hugging
Face demo wheels referenced by Pixal3D are Linux x86_64 CPython 3.10 CUDA wheels, so they cannot
be installed into this Blender runtime. The add-on keeps those modules isolated behind runtime
checks so Blender can load safely and the UI remains available.

Use `tools/vendor_wheels.py --download --install` to populate pywebview and other compatible
wheels locally. Use `--include-pixal3d` only on a platform where the requested wheels exist.
# Native TRELLIS.2 Dependencies

`o_voxel` is required for Pixal3D/TRELLIS.2 shape decoding. It is not available on PyPI for Blender 5.0's macOS arm64 CPython 3.11 runtime, and the public TRELLIS.2 demo wheel is currently Linux CPython 3.10 only. For macOS Metal, this extension bundles an `o_voxel` compatibility package with the pure-Python/PyTorch `convert.flexible_dual_grid_to_mesh` implementation adapted from `shivampkumar/trellis-mac` (MIT). CUDA installs should still use the native upstream `o_voxel` package when available.

# macOS Open Model Assets

The upstream Pixal3D pipeline config references `briaai/RMBG-2.0` for background removal. That repo can be gated on Hugging Face, so the macOS Metal runtime routes background removal to the open PyTorch/MPS-compatible `ZhengPeng7/BiRefNet` model by default. Users can override this with `PIXAL3D_RMBG_MODEL` if they intentionally want a different model and have access.

Use the extension's "Prepare Open Model Assets" action to cache the open macOS model stack before generation:

- `TencentARC/Pixal3D`
- `ZhengPeng7/BiRefNet`
- `Ruicheng/moge-2-vitl`
- `camenduru/dinov3-vitl16-pretrain-lvd1689m`
- `valeoai/NAF`

`valeoai/NAF` normally imports `natten`. On macOS Metal, this extension bundles `natten-mps`, aliases it to NAF's expected `natten` import surface during NAF loading, and caches the NAF checkpoint up front. If the community Metal backend fails for a future NAF change, the runtime falls back to a PyTorch/MPS interpolation upsampler instead of crashing generation.
