from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_3tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value, value)


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    if stride != 1:
        raise NotImplementedError("Metal sparse conv fallback currently supports stride=1 only.")
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = _as_3tuple(kernel_size)
    self.stride = _as_3tuple(stride)
    self.dilation = _as_3tuple(dilation)
    if padding is None:
        padding = tuple((k // 2) * d for k, d in zip(self.kernel_size, self.dilation))
    self.padding = _as_3tuple(padding)
    weight_shape = (out_channels, *self.kernel_size, in_channels)
    self.weight = torch.nn.Parameter(torch.empty(weight_shape))
    self.bias = torch.nn.Parameter(torch.empty(out_channels)) if bias else None
    torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
    if self.bias is not None:
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = 1 / fan_in**0.5
        torch.nn.init.uniform_(self.bias, -bound, bound)


def sparse_conv3d_forward(self, x):
    from .. import SparseTensor

    if x.shape[0] != 1:
        raise NotImplementedError("Metal sparse conv fallback currently supports batch size 1.")
    spatial_shape = tuple(int(v) for v in x.spatial_shape)
    coords = x.coords[:, 1:].long()
    if x.feats.device.type == "mps":
        out_feats = _mps_sparse_conv3d_linear_lookup(
            x.feats,
            coords,
            spatial_shape,
            self.weight,
            self.bias,
            padding=self.padding,
            dilation=self.dilation,
        )
    else:
        dense = torch.zeros(
            (1, self.in_channels, *spatial_shape),
            dtype=x.feats.dtype,
            device=x.feats.device,
        )
        dense[(0, slice(None), coords[:, 0], coords[:, 1], coords[:, 2])] = x.feats.T
        out_dense = F.conv3d(
            dense,
            self.weight.permute(0, 4, 1, 2, 3).contiguous(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        out_feats = out_dense[(0, slice(None), coords[:, 0], coords[:, 1], coords[:, 2])].T.contiguous()
    return x.replace(out_feats)


def _mps_sparse_conv3d_linear_lookup(feats, coords, spatial_shape, weight, bias, padding, dilation):
    depth, height, width = spatial_shape
    out_channels, kernel_depth, kernel_height, kernel_width, _weight_in = weight.shape
    pad_depth, pad_height, pad_width = padding
    dil_depth, dil_height, dil_width = dilation
    device = feats.device

    out_feats = torch.zeros((coords.shape[0], out_channels), dtype=feats.dtype, device=device)
    coords_cpu = coords.detach().cpu()
    stride_z = height * width
    stride_y = width
    input_keys = coords_cpu[:, 0] * stride_z + coords_cpu[:, 1] * stride_y + coords_cpu[:, 2]
    sorted_keys, sorted_order = torch.sort(input_keys)
    active_all = torch.arange(coords_cpu.shape[0], dtype=torch.long)

    for kd in range(kernel_depth):
        zs_cpu = coords_cpu[:, 0] + kd * dil_depth - pad_depth
        z_mask = (zs_cpu >= 0) & (zs_cpu < depth)
        if not bool(z_mask.any()):
            continue
        for kh in range(kernel_height):
            ys_cpu = coords_cpu[:, 1] + kh * dil_height - pad_height
            zy_mask = z_mask & (ys_cpu >= 0) & (ys_cpu < height)
            if not bool(zy_mask.any()):
                continue
            for kw in range(kernel_width):
                xs_cpu = coords_cpu[:, 2] + kw * dil_width - pad_width
                mask = zy_mask & (xs_cpu >= 0) & (xs_cpu < width)
                if not bool(mask.any()):
                    continue

                active_cpu = active_all[mask]
                query_keys = zs_cpu[mask] * stride_z + ys_cpu[mask] * stride_y + xs_cpu[mask]
                positions = torch.searchsorted(sorted_keys, query_keys)
                in_range = positions < sorted_keys.numel()
                if not bool(in_range.any()):
                    continue
                active_cpu = active_cpu[in_range]
                query_keys = query_keys[in_range]
                positions = positions[in_range]
                hits = sorted_keys[positions] == query_keys
                if not bool(hits.any()):
                    continue

                active_cpu = active_cpu[hits]
                neighbor_cpu = sorted_order[positions[hits]]
                active_mps = active_cpu.to(device=device)
                neighbor_mps = neighbor_cpu.to(device=device)
                kernel = weight[:, kd, kh, kw, :]
                out_feats[active_mps] = out_feats[active_mps] + feats[neighbor_mps] @ kernel.T

    if bias is not None:
        out_feats = out_feats + bias.reshape(1, out_channels)
    return out_feats


def _mps_sparse_conv3d_gather_matmul(dense, coords, weight, bias, padding, dilation):
    _batch, _in_channels, depth, height, width = dense.shape
    out_channels, kernel_depth, kernel_height, kernel_width, _weight_in = weight.shape
    pad_depth, pad_height, pad_width = padding
    dil_depth, dil_height, dil_width = dilation
    device = dense.device

    out_feats = torch.zeros((coords.shape[0], out_channels), dtype=dense.dtype, device=device)
    coords_cpu = coords.detach().cpu()

    for kd in range(kernel_depth):
        zs_cpu = coords_cpu[:, 0] + kd * dil_depth - pad_depth
        z_mask = (zs_cpu >= 0) & (zs_cpu < depth)
        if not bool(z_mask.any()):
            continue
        for kh in range(kernel_height):
            ys_cpu = coords_cpu[:, 1] + kh * dil_height - pad_height
            zy_mask = z_mask & (ys_cpu >= 0) & (ys_cpu < height)
            if not bool(zy_mask.any()):
                continue
            for kw in range(kernel_width):
                xs_cpu = coords_cpu[:, 2] + kw * dil_width - pad_width
                mask = zy_mask & (xs_cpu >= 0) & (xs_cpu < width)
                if not bool(mask.any()):
                    continue

                active = torch.where(mask)[0]
                active_mps = active.to(device=device)
                zs = zs_cpu[active].to(device=device)
                ys = ys_cpu[active].to(device=device)
                xs = xs_cpu[active].to(device=device)
                feats = dense[0, :, zs, ys, xs].T
                kernel = weight[:, kd, kh, kw, :]
                out_feats[active_mps] = out_feats[active_mps] + feats @ kernel.T

    if bias is not None:
        out_feats = out_feats + bias.reshape(1, out_channels)
    return out_feats


def _mps_sparse_conv3d_via_conv2d(dense, coords, weight, bias, padding, dilation):
    batch, in_channels, depth, height, width = dense.shape
    out_channels, kernel_depth, _kernel_height, _kernel_width, _weight_in = weight.shape
    pad_depth, pad_height, pad_width = padding
    dil_depth, dil_height, dil_width = dilation
    device = dense.device

    depth_padded = torch.zeros(
        (batch, in_channels, depth + pad_depth * 2, height, width),
        dtype=dense.dtype,
        device=device,
    )
    depth_padded[:, :, pad_depth:pad_depth + depth, :, :] = dense
    out_feats = torch.zeros((coords.shape[0], out_channels), dtype=dense.dtype, device=device)
    coords_cpu = coords.detach().cpu()
    coords_by_depth = []
    for depth_index in range(depth):
        active_cpu = torch.where(coords_cpu[:, 0] == depth_index)[0]
        if active_cpu.numel() == 0:
            coords_by_depth.append(None)
            continue
        coords_by_depth.append((
            active_cpu.to(device=device),
            coords_cpu[active_cpu, 1],
            coords_cpu[active_cpu, 2],
        ))

    for kernel_index in range(kernel_depth):
        depth_slice = depth_padded[:, :, kernel_index * dil_depth:kernel_index * dil_depth + depth, :, :]
        weight_2d = weight[:, kernel_index, :, :, :].permute(0, 3, 1, 2).contiguous()
        for depth_start in range(depth):
            active = coords_by_depth[depth_start]
            if active is None:
                continue
            depth_stop = depth_start + 1
            plane_chunk = depth_slice[:, :, depth_start:depth_stop, :, :].squeeze(2)
            active_idx, ys_cpu, xs_cpu = active
            point_feats = _mps_conv2d_tiled_points(
                plane_chunk,
                weight_2d,
                ys_cpu,
                xs_cpu,
                padding=(pad_height, pad_width),
                dilation=(dil_height, dil_width),
            )
            out_feats[active_idx] = out_feats[active_idx] + point_feats

    if bias is not None:
        out_feats = out_feats + bias.reshape(1, out_channels)
    return out_feats


def _mps_conv2d_tiled_points(planes, weight_2d, ys_cpu, xs_cpu, padding, dilation, tile_size=64):
    batch_depth, _in_channels, height, width = planes.shape
    out_channels = weight_2d.shape[0]
    pad_height, pad_width = padding
    dil_height, dil_width = dilation
    kernel_height, kernel_width = weight_2d.shape[2:]
    effective_height = (kernel_height - 1) * dil_height + 1
    effective_width = (kernel_width - 1) * dil_width + 1
    device = planes.device

    padded = torch.zeros(
        (batch_depth, planes.shape[1], height + pad_height * 2, width + pad_width * 2),
        dtype=planes.dtype,
        device=device,
    )
    padded[:, :, pad_height:pad_height + height, pad_width:pad_width + width] = planes
    point_output = torch.empty((ys_cpu.numel(), out_channels), dtype=planes.dtype, device=device)

    for h_start in range(0, height, tile_size):
        h_stop = min(h_start + tile_size, height)
        y_mask = (ys_cpu >= h_start) & (ys_cpu < h_stop)
        if not bool(y_mask.any()):
            continue
        for w_start in range(0, width, tile_size):
            w_stop = min(w_start + tile_size, width)
            mask = y_mask & (xs_cpu >= w_start) & (xs_cpu < w_stop)
            if not bool(mask.any()):
                continue
            tile = padded[:, :, h_start:h_stop + effective_height - 1, w_start:w_stop + effective_width - 1]
            try:
                conv_tile = F.conv2d(
                    tile,
                    weight_2d,
                    None,
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(dil_height, dil_width),
                )
            except RuntimeError:
                print(
                    "[Metal Conv3D] tiled point conv2d failed: "
                    f"tile={tuple(tile.shape)}, weight={tuple(weight_2d.shape)}, "
                    f"output_tile={(h_stop - h_start, w_stop - w_start)}, dilation={(dil_height, dil_width)}, "
                    f"points={int(mask.sum().item())}",
                    flush=True,
                )
                raise
            local_rows = (ys_cpu[mask] - h_start).to(device=device)
            local_cols = (xs_cpu[mask] - w_start).to(device=device)
            point_indices = torch.where(mask)[0].to(device=device)
            point_output[point_indices] = conv_tile[0, :, local_rows, local_cols].T

    return point_output


def _mps_conv2d_tiled(planes, weight_2d, padding, dilation, tile_size=96):
    batch_depth, _in_channels, height, width = planes.shape
    out_channels = weight_2d.shape[0]
    pad_height, pad_width = padding
    dil_height, dil_width = dilation
    kernel_height, kernel_width = weight_2d.shape[2:]
    effective_height = (kernel_height - 1) * dil_height + 1
    effective_width = (kernel_width - 1) * dil_width + 1

    padded = torch.zeros(
        (batch_depth, planes.shape[1], height + pad_height * 2, width + pad_width * 2),
        dtype=planes.dtype,
        device=planes.device,
    )
    padded[:, :, pad_height:pad_height + height, pad_width:pad_width + width] = planes
    output = torch.zeros(
        (batch_depth, out_channels, height, width),
        dtype=planes.dtype,
        device=planes.device,
    )

    for h_start in range(0, height, tile_size):
        h_stop = min(h_start + tile_size, height)
        for w_start in range(0, width, tile_size):
            w_stop = min(w_start + tile_size, width)
            tile = padded[:, :, h_start:h_stop + effective_height - 1, w_start:w_stop + effective_width - 1]
            try:
                output[:, :, h_start:h_stop, w_start:w_stop] = F.conv2d(
                    tile,
                    weight_2d,
                    None,
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(dil_height, dil_width),
                )
            except RuntimeError:
                print(
                    "[Metal Conv3D] tiled conv2d failed: "
                    f"tile={tuple(tile.shape)}, weight={tuple(weight_2d.shape)}, "
                    f"output_tile={(h_stop - h_start, w_stop - w_start)}, dilation={(dil_height, dil_width)}",
                    flush=True,
                )
                raise
    return output


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    raise NotImplementedError("Metal sparse inverse conv fallback is not implemented.")


def sparse_inverse_conv3d_forward(self, x):
    raise NotImplementedError("Metal sparse inverse conv fallback is not implemented.")
