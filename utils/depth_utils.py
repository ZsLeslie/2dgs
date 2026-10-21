"""3DGS-compatible inverse-depth calibration and on-demand loading."""

import json
import os
import warnings

import cv2
import numpy as np
import torch


def read_depth_params(path):
    with open(path) as file:
        params = json.load(file)
    if not isinstance(params, dict):
        raise ValueError("{} must map image names to scale/offset objects".format(path))
    for name, entry in params.items():
        try:
            scale, offset = float(entry["scale"]), float(entry["offset"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid depth calibration for {!r} in {}".format(name, path)) from error
        if not np.isfinite(scale) or not np.isfinite(offset):
            raise ValueError("Non-finite depth calibration for {!r} in {}".format(name, path))
        entry["scale"], entry["offset"] = scale, offset
    positive_scales = [p["scale"] for p in params.values() if p["scale"] > 0]
    median = float(np.median(positive_scales)) if positive_scales else 0.0
    for entry in params.values():
        entry["med_scale"] = median
    return params


def depth_metadata(image_name, depths_folder, params):
    """Preserve subdirectories and dots in COLMAP names, as in 3DGS."""
    if not depths_folder:
        return "", None
    key = os.path.splitext(image_name)[0]
    calibration = params.get(key)
    if calibration is None:
        warnings.warn("No depth calibration for {!r}; skipping its depth supervision".format(key))
        return "", None
    return os.path.join(depths_folder, key + ".png"), calibration


def depth_is_usable(info):
    if not info.depth_path:
        return False
    if info.is_nerf_synthetic:
        return True
    params = info.depth_params
    if params is None:
        return False
    scale, median = params["scale"], params["med_scale"]
    return scale > 0 and 0.2 * median <= scale <= 5 * median


def load_inverse_depth(info, resolution, alpha_mask=None):
    """Return CPU float32 depth/mask tensors; decoding happens only here."""
    if not depth_is_usable(info):
        return None, None
    if not os.path.isfile(info.depth_path):
        raise FileNotFoundError("Depth image not found: {}".format(info.depth_path))
    pixels = cv2.imread(info.depth_path, cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise ValueError("Cannot decode depth image: {}".format(info.depth_path))
    if pixels.ndim != 2 or pixels.dtype != np.uint16:
        raise ValueError("Expected a single-channel uint16 inverse-depth PNG: {}".format(info.depth_path))
    divisor = 512.0 if info.is_nerf_synthetic else 65536.0
    inverse_depth = cv2.resize(pixels.astype(np.float32) / divisor, resolution, interpolation=cv2.INTER_LINEAR)
    if info.depth_params is not None:
        inverse_depth = inverse_depth * info.depth_params["scale"] + info.depth_params["offset"]
    valid = np.isfinite(inverse_depth) & (inverse_depth > 0)
    inverse_depth = np.where(valid, inverse_depth, 0).astype(np.float32)
    mask = torch.from_numpy(valid[None].astype(np.float32))
    if alpha_mask is not None:
        mask *= alpha_mask.cpu()
    return torch.from_numpy(inverse_depth[None]), mask
