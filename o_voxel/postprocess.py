from __future__ import annotations

from typing import Dict, Union

import cv2
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image


def _normalize_grid(
    coords: torch.Tensor,
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
):
    device = coords.device
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    if voxel_size is not None:
        if isinstance(voxel_size, (int, float)):
            voxel_size = [float(voxel_size)] * 3
        if isinstance(voxel_size, (list, tuple, np.ndarray)):
            voxel_size = torch.tensor(np.array(voxel_size), dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple, np.ndarray)):
            grid_size = torch.tensor(np.array(grid_size), dtype=torch.int32, device=device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size.float()
    return aabb, voxel_size, grid_size


def _sparse_sample_trilinear(
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    points: torch.Tensor,
    aabb: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: torch.Tensor,
    chunk_size: int = 262144,
) -> torch.Tensor:
    device = attr_volume.device
    coords_cpu = coords.detach().cpu().long()
    attrs = attr_volume
    grid_size_cpu = grid_size.detach().cpu().long()
    stride_y = int(grid_size_cpu[2].item())
    stride_z = int(grid_size_cpu[1].item() * stride_y)
    coord_keys = coords_cpu[:, 0] * stride_z + coords_cpu[:, 1] * stride_y + coords_cpu[:, 2]
    sorted_keys, sorted_order = torch.sort(coord_keys)
    out_chunks = []

    for start in range(0, points.shape[0], chunk_size):
        point_chunk = points[start:start + chunk_size]
        grid = ((point_chunk.to(device) - aabb[0]) / voxel_size).detach().cpu()
        lower = torch.floor(grid).long()
        frac = (grid - lower.float()).clamp(0, 1)
        accum = torch.zeros((lower.shape[0], attrs.shape[1]), dtype=attrs.dtype, device=device)
        weights = torch.zeros((lower.shape[0],), dtype=attrs.dtype, device=device)

        for dz in (0, 1):
            wz = frac[:, 0] if dz else 1.0 - frac[:, 0]
            for dy in (0, 1):
                wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
                for dx in (0, 1):
                    wx = frac[:, 2] if dx else 1.0 - frac[:, 2]
                    query = lower + torch.tensor([dz, dy, dx], dtype=torch.long)
                    valid_bounds = (
                        (query[:, 0] >= 0) & (query[:, 0] < grid_size_cpu[0]) &
                        (query[:, 1] >= 0) & (query[:, 1] < grid_size_cpu[1]) &
                        (query[:, 2] >= 0) & (query[:, 2] < grid_size_cpu[2])
                    )
                    if not bool(valid_bounds.any()):
                        continue
                    active = torch.where(valid_bounds)[0]
                    query_keys = query[active, 0] * stride_z + query[active, 1] * stride_y + query[active, 2]
                    positions = torch.searchsorted(sorted_keys, query_keys)
                    in_range = positions < sorted_keys.numel()
                    if not bool(in_range.any()):
                        continue
                    active = active[in_range]
                    query_keys = query_keys[in_range]
                    positions = positions[in_range]
                    hits = sorted_keys[positions] == query_keys
                    if not bool(hits.any()):
                        continue
                    active = active[hits]
                    attr_indices = sorted_order[positions[hits]].to(device=device)
                    weight = (wz[active] * wy[active] * wx[active]).to(device=device, dtype=attrs.dtype)
                    accum[active.to(device=device)] += attrs[attr_indices] * weight[:, None]
                    weights[active.to(device=device)] += weight

        weights = weights.clamp_min(torch.finfo(weights.dtype).eps)
        out_chunks.append((accum / weights[:, None]).detach().cpu())

    return torch.cat(out_chunks, dim=0)


def _unwrap_with_xatlas(mesh: trimesh.Trimesh):
    import xatlas

    vmapping, faces, uvs = xatlas.parametrize(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float32)[vmapping]
    return vertices, faces.astype(np.int64), uvs.astype(np.float32)


def _remove_small_components(mesh: trimesh.Trimesh, min_faces: int = 500, verbose: bool = False) -> trimesh.Trimesh:
    if len(mesh.faces) < min_faces * 2:
        return mesh
    try:
        components = trimesh.graph.connected_components(
            mesh.face_adjacency,
            nodes=np.arange(len(mesh.faces)),
            min_len=min_faces,
        )
        if not components:
            return mesh
        keep = np.zeros(len(mesh.faces), dtype=bool)
        for component in components:
            keep[component] = True
        if keep.all():
            return mesh
        mesh = mesh.copy()
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
        if verbose:
            print(f"[o_voxel.portable] removed small components; kept {len(mesh.faces)} faces", flush=True)
    except Exception as error:
        if verbose:
            print(f"[o_voxel.portable] small-component cleanup skipped: {error}", flush=True)
    return mesh


def _simplify_mesh(mesh: trimesh.Trimesh, target_faces: int, verbose: bool = False) -> trimesh.Trimesh:
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import fast_simplification

        vertices, faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
            target_count=int(target_faces),
            agg=10.0,
        )
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if verbose:
            print(f"[o_voxel.portable] simplified mesh to {len(mesh.faces)} faces", flush=True)
    except Exception as error:
        if verbose:
            print(f"[o_voxel.portable] simplification skipped: {error}", flush=True)
    return mesh


def _rasterize_uv_faces(uvs: np.ndarray, faces: np.ndarray, texture_size: int) -> np.ndarray:
    face_ids = np.full((texture_size, texture_size), -1, dtype=np.int32)
    uv_pixels = uvs.copy()
    uv_pixels[:, 0] *= texture_size - 1
    uv_pixels[:, 1] = (1.0 - uv_pixels[:, 1]) * (texture_size - 1)
    for face_index, face in enumerate(faces):
        tri = np.rint(uv_pixels[face]).astype(np.int32)
        if (
            tri[:, 0].max() < 0 or tri[:, 1].max() < 0 or
            tri[:, 0].min() >= texture_size or tri[:, 1].min() >= texture_size
        ):
            continue
        tri[:, 0] = np.clip(tri[:, 0], 0, texture_size - 1)
        tri[:, 1] = np.clip(tri[:, 1], 0, texture_size - 1)
        cv2.fillConvexPoly(face_ids, tri, int(face_index))
    return face_ids


def _barycentric_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    face_ids: np.ndarray,
    chunk_size: int = 262144,
) -> tuple[torch.Tensor, np.ndarray]:
    ys, xs = np.nonzero(face_ids >= 0)
    ids = face_ids[ys, xs].astype(np.int64)
    texture_size = face_ids.shape[0]
    points = []
    uv_pixels = uvs.copy()
    uv_pixels[:, 0] *= texture_size - 1
    uv_pixels[:, 1] = (1.0 - uv_pixels[:, 1]) * (texture_size - 1)

    for start in range(0, ids.shape[0], chunk_size):
        face_chunk = ids[start:start + chunk_size]
        px = xs[start:start + chunk_size].astype(np.float32) + 0.5
        py = ys[start:start + chunk_size].astype(np.float32) + 0.5
        tri = uv_pixels[faces[face_chunk]]
        v0 = tri[:, 0]
        v1 = tri[:, 1]
        v2 = tri[:, 2]
        denom = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        denom = np.where(np.abs(denom) < 1.0e-8, 1.0, denom)
        w0 = ((v1[:, 1] - v2[:, 1]) * (px - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (py - v2[:, 1])) / denom
        w1 = ((v2[:, 1] - v0[:, 1]) * (px - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (py - v2[:, 1])) / denom
        w2 = 1.0 - w0 - w1
        face_vertices = vertices[faces[face_chunk]]
        point_chunk = (
            face_vertices[:, 0] * w0[:, None] +
            face_vertices[:, 1] * w1[:, None] +
            face_vertices[:, 2] * w2[:, None]
        )
        points.append(torch.from_numpy(point_chunk.astype(np.float32)))

    return torch.cat(points, dim=0), np.stack([ys, xs], axis=1)


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    del remesh, remesh_band, remesh_project
    del mesh_cluster_threshold_cone_half_angle_rad, mesh_cluster_refine_iterations
    del mesh_cluster_global_iterations, mesh_cluster_smooth_strength, use_tqdm

    aabb, voxel_size, grid_size = _normalize_grid(coords, aabb, voxel_size, grid_size)
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    print(f"[o_voxel.portable] Starting portable postprocess: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces", flush=True)
    try:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    except Exception as error:
        if verbose:
            print(f"[o_voxel.portable] mesh repair skipped: {error}", flush=True)

    mesh = _remove_small_components(mesh, min_faces=min(500, max(1, len(mesh.faces) // 10000)), verbose=True)
    bake_face_target = int(decimation_target or min(500000, len(mesh.faces)))
    if decimation_target <= 0 and len(mesh.faces) > bake_face_target:
        print(
            f"[o_voxel.portable] No decimation target supplied; capping portable UV bake at {bake_face_target} faces.",
            flush=True,
        )
    mesh = _simplify_mesh(mesh, bake_face_target, verbose=True)

    print("[o_voxel.portable] UV unwrapping with xatlas...", flush=True)
    out_vertices, out_faces, out_uvs = _unwrap_with_xatlas(mesh)
    print(f"[o_voxel.portable] UV unwrap complete: {len(out_vertices)} vertices, {len(out_faces)} faces", flush=True)
    print(f"[o_voxel.portable] Rasterizing UV atlas at {texture_size}x{texture_size}...", flush=True)
    face_ids = _rasterize_uv_faces(out_uvs, out_faces, int(texture_size))
    print("[o_voxel.portable] Baking sparse PBR attributes into texture maps...", flush=True)
    sample_points, pixel_indices = _barycentric_points(out_vertices, out_faces, out_uvs, face_ids)
    attrs = torch.zeros(int(texture_size), int(texture_size), attr_volume.shape[1], dtype=torch.float32)
    if sample_points.numel() > 0:
        sampled = _sparse_sample_trilinear(attr_volume, coords, sample_points, aabb, voxel_size, grid_size)
        attrs[pixel_indices[:, 0], pixel_indices[:, 1]] = sampled.to(dtype=torch.float32)

    mask = face_ids >= 0
    mask_inv = (~mask).astype(np.uint8)
    base_color = np.clip(attrs[..., attr_layout["base_color"]].numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout["metallic"]].numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout["roughness"]].numpy() * 255, 0, 255).astype(np.uint8)
    alpha_slice = attr_layout.get("alpha")
    if alpha_slice is None:
        alpha = np.full((int(texture_size), int(texture_size), 1), 255, dtype=np.uint8)
    else:
        alpha = np.clip(attrs[..., alpha_slice].numpy() * 255, 0, 255).astype(np.uint8)

    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)
    if metallic.ndim == 2:
        metallic = metallic[..., None]
    if roughness.ndim == 2:
        roughness = roughness[..., None]
    if alpha.ndim == 2:
        alpha = alpha[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )
    print("[o_voxel.portable] PBR texture bake complete.", flush=True)
    return trimesh.Trimesh(
        vertices=out_vertices,
        faces=out_faces,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=out_uvs, material=material),
    )
