import os
import argparse
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image
from pixal3d.utils.device_utils import configure_backend_for_device, describe_device, resolve_device

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'
_BOOT_DEVICE = resolve_device(os.environ.get("PIXAL3D_DEVICE", "auto"))
configure_backend_for_device(_BOOT_DEVICE)

from pixal3d.pipelines import Pixal3DImageTo3DPipeline

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="auto", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    device = resolve_device(device)
    moge_model = MoGeModel.from_pretrained(model_name).to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="auto"):
    device = resolve_device(device)
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    pipeline.low_vram = device.type != "cuda"
    pipeline.to(device)

    pipeline.image_cond_model_ss.to(device)
    pipeline.image_cond_model_shape_512.to(device)
    pipeline.image_cond_model_shape_1024.to(device)
    pipeline.image_cond_model_tex_1024.to(device)

    print("[NAF] Pre-loading NAF upsampler model...")
    for attr in ['image_cond_model_ss', 'image_cond_model_shape_512', 'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
        model = getattr(pipeline, attr, None)
        if model is not None and getattr(model, 'use_naf_upsample', False):
            model._load_naf()

    return pipeline


def _sparse_attr_lookup(coords, attrs, query_coords, stride_z, stride_y):
    coord_keys = coords[:, 0] * stride_z + coords[:, 1] * stride_y + coords[:, 2]
    sorted_keys, sorted_order = torch.sort(coord_keys)
    query_keys = query_coords[:, 0] * stride_z + query_coords[:, 1] * stride_y + query_coords[:, 2]
    positions = torch.searchsorted(sorted_keys, query_keys)
    valid = positions < sorted_keys.numel()
    result = torch.zeros((query_coords.shape[0], attrs.shape[1]), dtype=attrs.dtype)
    if not bool(valid.any()):
        return result, valid

    valid_positions = positions[valid]
    hits = sorted_keys[valid_positions] == query_keys[valid]
    hit_mask = torch.zeros(query_coords.shape[0], dtype=torch.bool)
    if not bool(hits.any()):
        return result, hit_mask

    vertex_indices = torch.where(valid)[0][hits]
    attr_indices = sorted_order[valid_positions[hits]]
    result[vertex_indices] = attrs[attr_indices]
    hit_mask[vertex_indices] = True
    return result, hit_mask


def _mesh_vertex_colors_from_voxels(mesh, attr_layout):
    if not hasattr(mesh, "attrs") or not hasattr(mesh, "coords") or "base_color" not in attr_layout:
        return None

    vertices = mesh.vertices.detach().cpu()
    coords = mesh.coords.detach().cpu().long()
    attrs = mesh.attrs.detach().cpu()[:, attr_layout["base_color"]].clamp(0, 1)
    if vertices.numel() == 0 or coords.numel() == 0 or attrs.numel() == 0:
        return None

    origin = mesh.origin.detach().cpu() if hasattr(mesh.origin, "detach") else torch.tensor(mesh.origin)
    voxel_size = float(mesh.voxel_size)
    grid = (vertices - origin) / voxel_size - 0.5
    lower = torch.floor(grid).long()
    frac = (grid - lower.float()).clamp(0, 1)
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values

    stride_y = int(maxs[2].item() + 1)
    stride_z = int(maxs[1].item() + 1) * stride_y
    accum = torch.zeros((vertices.shape[0], attrs.shape[1]), dtype=torch.float32)
    weights = torch.zeros(vertices.shape[0], dtype=torch.float32)

    for dz in (0, 1):
        wz = frac[:, 0] if dz else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dx in (0, 1):
                wx = frac[:, 2] if dx else 1.0 - frac[:, 2]
                weight = (wz * wy * wx).to(torch.float32)
                query = lower + torch.tensor([dz, dy, dx], dtype=torch.long)
                valid_bounds = ((query >= mins) & (query <= maxs)).all(dim=1)
                if not bool(valid_bounds.any()):
                    continue
                sample, hit_mask = _sparse_attr_lookup(coords, attrs, query, stride_z, stride_y)
                hit_mask &= valid_bounds
                if not bool(hit_mask.any()):
                    continue
                accum[hit_mask] += sample[hit_mask].to(torch.float32) * weight[hit_mask, None]
                weights[hit_mask] += weight[hit_mask]

    nearest = torch.round(grid).long()
    nearest[:, 0].clamp_(int(coords[:, 0].min()), int(coords[:, 0].max()))
    nearest[:, 1].clamp_(int(coords[:, 1].min()), int(coords[:, 1].max()))
    nearest[:, 2].clamp_(int(coords[:, 2].min()), int(coords[:, 2].max()))
    nearest_sample, nearest_hits = _sparse_attr_lookup(coords, attrs, nearest, stride_z, stride_y)
    missing = weights <= 1.0e-6
    if bool((missing & nearest_hits).any()):
        accum[missing & nearest_hits] = nearest_sample[missing & nearest_hits]
        weights[missing & nearest_hits] = 1.0

    colors = torch.full((vertices.shape[0], 4), 255, dtype=torch.uint8)
    colored = weights > 1.0e-6
    if not bool(colored.any()):
        return colors.numpy()
    base_color = (accum[colored] / weights[colored, None]).clamp(0, 1)
    colors[colored, :3] = (base_color * 255).round().to(torch.uint8)
    return colors.numpy()

# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="auto", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    device = resolve_device(device)
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: str,
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    decimation_target: int = 1000000,
    target_resolution: int = 1536,
    texture_size: int = 4096,
    model_path: str = MODEL_PATH,
    device: str = "auto",
    enable_mps_fallback: bool = True,
):
    device_obj = resolve_device(device)
    configure_backend_for_device(device_obj, enable_mps_fallback=enable_mps_fallback)
    print(f"[Device] Using {describe_device(device_obj)}")

    # Load models
    pipeline = init_pipeline(model_path, device=device_obj)

    print("[MoGe-2] Loading model for camera estimation...")
    moge_model = load_moge_model(device=device_obj)

    # Preprocess image
    print(f"[Inference] Processing image: {image_path}")
    img = Image.open(image_path)
    image_preprocessed = pipeline.preprocess_image(img)

    # Save preprocessed image for MoGe
    tmp_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), f"_tmp_preprocessed_{int(time.time()*1000)}.png")
    image_preprocessed.save(tmp_path)

    # Camera estimation
    print("[Inference] Estimating camera parameters...")
    camera_params = get_camera_params_wild_moge(
        tmp_path, moge_model, device=device_obj,
        mesh_scale=mesh_scale, extend_pixel=extend_pixel,
        image_resolution=image_resolution,
    )
    os.remove(tmp_path)
    print(f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, distance={camera_params['distance']:.4f}")

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    if target_resolution not in {1024, 1536}:
        raise ValueError(f"Unsupported Pixal3D target resolution: {target_resolution}")
    pipeline_type = f"{target_resolution}_cascade"
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    try:
        import o_voxel.postprocess as o_voxel_postprocess
        if device_obj.type == "cuda":
            module_file = os.path.normcase(os.path.abspath(getattr(o_voxel_postprocess, "__file__", "")))
            vendor_root = os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(__file__), "_vendor")))
            if not module_file.startswith(vendor_root + os.sep):
                raise RuntimeError(f"CUDA requires native vendor o_voxel.postprocess, got {module_file}")
        glb = o_voxel_postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
            grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target, texture_size=texture_size,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )
        if device_obj.type != "cuda":
            print("[Portable] o_voxel postprocess exported PBR texture maps through the xatlas bake path.", flush=True)
    except Exception as error:
        if device_obj.type == "cuda":
            raise
        print(f"[Metal] o_voxel PBR export unavailable, exporting plain GLB mesh: {error}")
        import trimesh
        vertex_colors = _mesh_vertex_colors_from_voxels(mesh, pipeline.pbr_attr_layout)
        if vertex_colors is not None:
            print("[Metal] Exporting GLB with trilinear sparse-voxel base-color vertex colors.")
        glb = trimesh.Trimesh(
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.faces.detach().cpu().numpy(),
            vertex_colors=vertex_colors,
            process=False,
        )
        try:
            filled = trimesh.repair.fill_holes(glb)
            trimesh.repair.fix_normals(glb)
            print(f"[Metal] CPU mesh repair fill_holes applied: {filled}", flush=True)
        except Exception as repair_error:
            print(f"[Metal] CPU mesh repair unavailable, keeping raw mesh: {repair_error}", flush=True)
        if decimation_target and decimation_target > 0 and len(glb.faces) > decimation_target:
            try:
                print(f"[Metal] Simplifying mesh to about {decimation_target} faces...", flush=True)
                glb = glb.simplify_quadric_decimation(face_count=int(decimation_target), aggression=5)
                print(f"[Metal] Simplified mesh: {len(glb.faces)} faces.", flush=True)
            except Exception as simplify_error:
                print(f"[Metal] Mesh simplification unavailable, keeping full mesh: {simplify_error}", flush=True)

    # Match Blender's expected upright orientation. Earlier exports needed a
    # manual +90 X / +180 Z world correction after import; bake it into the GLB.
    blender_fix = np.array([
        [-1,  0,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(blender_fix @ rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cuda:0", "mps", "metal", "cpu"], help="Runtime device")
    parser.add_argument("--decimation_target", type=int, default=1000000, help="Optional low-poly target face count; 0 keeps full detail")
    parser.add_argument("--target_resolution", type=int, default=1536, choices=[1024, 1536], help="Pixal3D cascade target resolution")
    parser.add_argument("--max_num_tokens", type=int, default=49152, help="Maximum high-resolution sparse tokens before lowering the cascade resolution")
    parser.add_argument("--texture_size", type=int, default=4096, help="CUDA o_voxel PBR texture size")
    parser.add_argument("--disable_mps_fallback", action="store_true", help="Disable PyTorch MPS CPU fallback")

    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        model_path=args.model_path,
        device=args.device,
        decimation_target=args.decimation_target,
        target_resolution=args.target_resolution,
        max_num_tokens=args.max_num_tokens,
        texture_size=args.texture_size,
        enable_mps_fallback=not args.disable_mps_fallback,
    )
