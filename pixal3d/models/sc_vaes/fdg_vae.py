from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules import sparse as sp
from .sparse_unet_vae import (
    SparseResBlock3d,
    SparseConvNeXtBlock3d,
    
    SparseResBlockDownsample3d,
    SparseResBlockUpsample3d,
    SparseResBlockS2C3d,
    SparseResBlockC2S3d,
)
from .sparse_unet_vae import (
    SparseUnetVaeEncoder,
    SparseUnetVaeDecoder,
)
from ...representations import Mesh

try:
    from o_voxel.convert import flexible_dual_grid_to_mesh
except Exception:
    flexible_dual_grid_to_mesh = None


def _cube_surface_fallback(
    coords: torch.Tensor,
    vertices: torch.Tensor,
    intersected: torch.Tensor,
    quad_lerp: torch.Tensor,
    aabb: list,
    grid_size: int,
    train: bool = False,
):
    mask = intersected.reshape(intersected.shape[0], -1).any(dim=1)
    occupied_coords = coords[mask].detach().cpu().int()
    device = vertices.device
    if occupied_coords.numel() == 0:
        empty_vertices = torch.empty((0, 3), dtype=torch.float32, device=device)
        empty_faces = torch.empty((0, 3), dtype=torch.int32, device=device)
        return empty_vertices, empty_faces

    occupied = {tuple(int(v) for v in coord.tolist()) for coord in occupied_coords}
    vertex_index: dict[tuple[int, int, int], int] = {}
    out_vertices: list[tuple[int, int, int]] = []
    out_faces: list[tuple[int, int, int]] = []

    def vid(corner: tuple[int, int, int]) -> int:
        index = vertex_index.get(corner)
        if index is None:
            index = len(out_vertices)
            vertex_index[corner] = index
            out_vertices.append(corner)
        return index

    faces = (
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    )

    for x, y, z in occupied:
        for normal, corners in faces:
            neighbor = (x + normal[0], y + normal[1], z + normal[2])
            if neighbor in occupied:
                continue
            ids = [vid((x + cx, y + cy, z + cz)) for cx, cy, cz in corners]
            out_faces.append((ids[0], ids[1], ids[2]))
            out_faces.append((ids[0], ids[2], ids[3]))

    min_corner = torch.tensor(aabb[0], dtype=torch.float32, device=device)
    scale = (torch.tensor(aabb[1], dtype=torch.float32, device=device) - min_corner) / float(grid_size)
    out_vertices_tensor = torch.tensor(out_vertices, dtype=torch.float32, device=device) * scale + min_corner
    out_faces_tensor = torch.tensor(out_faces, dtype=torch.int32, device=device)
    return out_vertices_tensor, out_faces_tensor


def _flexible_dual_grid_to_mesh(*args, **kwargs):
    if flexible_dual_grid_to_mesh is not None:
        return flexible_dual_grid_to_mesh(*args, **kwargs)
    print("[Metal] o_voxel flexible dual grid converter unavailable; using coarse voxel surface fallback.", flush=True)
    return _cube_surface_fallback(*args, **kwargs)


class FlexiDualGridVaeEncoder(SparseUnetVaeEncoder):
    def __init__(
        self,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        down_block_type: List[str],
        block_args: List[Dict[str, Any]],
        use_fp16: bool = False,
    ):
        super().__init__(
            6,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            down_block_type,
            block_args,
            use_fp16,
        )
        
    def forward(self, vertices: sp.SparseTensor, intersected: sp.SparseTensor, sample_posterior=False, return_raw=False):
        x = vertices.replace(torch.cat([
            vertices.feats - 0.5,
            intersected.feats.float() - 0.5,
        ], dim=1))
        return super().forward(x, sample_posterior, return_raw)
    
    
class FlexiDualGridVaeDecoder(SparseUnetVaeDecoder):
    def __init__(
        self,
        resolution: int,
        model_channels: List[int],
        latent_channels: int,
        num_blocks: List[int],
        block_type: List[str],
        up_block_type: List[str],
        block_args: List[Dict[str, Any]],
        voxel_margin: float = 0.5,
        use_fp16: bool = False,
    ):
        self.resolution = resolution
        self.voxel_margin = voxel_margin
        
        super().__init__(
            7,
            model_channels,
            latent_channels,
            num_blocks,
            block_type,
            up_block_type,
            block_args,
            use_fp16,
        )

    def set_resolution(self, resolution: int) -> None:
        self.resolution = resolution
        
    def forward(self, x: sp.SparseTensor, gt_intersected: sp.SparseTensor = None, **kwargs):
        decoded = super().forward(x, **kwargs)
        if self.training:
            h, subs_gt, subs = decoded
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected_logits = h.replace(h.feats[..., 3:6])
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*_flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=True
            )) for v, i, q in zip(vertices, gt_intersected, quad_lerp)]
            return mesh, vertices, intersected_logits, subs_gt, subs
        else:
            out_list = list(decoded) if isinstance(decoded, tuple) else [decoded]
            h = out_list[0]
            vertices = h.replace((1 + 2 * self.voxel_margin) * F.sigmoid(h.feats[..., 0:3]) - self.voxel_margin)
            intersected = h.replace(h.feats[..., 3:6] > 0)
            quad_lerp = h.replace(F.softplus(h.feats[..., 6:7]))
            mesh = [Mesh(*_flexible_dual_grid_to_mesh(
                v.coords[:, 1:], v.feats, i.feats, q.feats,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                grid_size=self.resolution,
                train=False
            )) for v, i, q in zip(vertices, intersected, quad_lerp)]
            out_list[0] = mesh
            return out_list[0] if len(out_list) == 1 else tuple(out_list)
