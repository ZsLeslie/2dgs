"""Label original COLMAP points whose checked observations all lie in black masks.

Masks only select fixed voxel seeds; neither images nor training points are
modified. One mask is decoded at a time. No all-point/all-camera projections.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def label_mask_observations(point_ids, images, cameras, image_dir, mask_dir, threshold=0):
    """All checked observations must be black; at least one is required."""
    if not 0 <= threshold <= 254:
        raise ValueError("Mask threshold must be between 0 and 254")
    order = np.argsort(point_ids)
    sorted_ids = np.asarray(point_ids)[order]
    if len(sorted_ids) == 0 or np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("COLMAP point IDs must be nonempty and unique")
    seen = np.zeros(len(point_ids), dtype=bool)
    rejected = np.zeros(len(point_ids), dtype=bool)
    checked = skipped = 0
    for image in images.values():
        name = Path(image.name)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f"Unsafe COLMAP image name: {image.name}")
        image_path = Path(image_dir) / name
        if not image_path.is_file():
            skipped += 1
            continue
        mask_path = Path(mask_dir) / name.with_suffix('.png')
        if not mask_path.is_file():
            raise ValueError(f"Missing voxel mask: {mask_path}")
        camera = cameras[image.camera_id]
        with Image.open(image_path) as rgb, Image.open(mask_path) as mask_image:
            if mask_image.size != rgb.size or mask_image.size != (camera.width, camera.height):
                raise ValueError(f"Image/mask/COLMAP camera size mismatch: {image.name}")
            mask = np.asarray(mask_image.convert('L'))
        tracked = image.point3D_ids != -1
        ids = image.point3D_ids[tracked]
        positions = np.searchsorted(sorted_ids, ids)
        if np.any(positions >= len(sorted_ids)) or np.any(sorted_ids[positions] != ids):
            raise ValueError(f"Observation references unknown COLMAP point: {image.name}")
        indices = order[positions]
        xy = image.xys[tracked]
        x, y = xy[:, 0], xy[:, 1]
        valid = (np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
                 & (x < camera.width) & (y < camera.height))
        black = np.zeros(len(x), dtype=bool)
        black[valid] = mask[np.floor(y[valid]).astype(np.int64),
                            np.floor(x[valid]).astype(np.int64)] <= threshold
        seen[indices] = True
        rejected[indices[~black]] = True
        checked += 1
        if checked % 100 == 0:
            print(f"[Voxel masks] checked {checked}/{len(images)} registered images", flush=True)
    if checked == 0:
        raise ValueError("No registered images found for voxel masks")
    selected = seen & ~rejected
    print(f"[Voxel masks] images={checked} skipped={skipped} selected={selected.sum()}/"
          f"{len(selected)} rejected={rejected.sum()} unobserved={(~seen).sum()}")
    return selected


def align_point_labels(input_xyz, colmap_xyz, labels):
    """Match PLY float32 coordinates, allowing reordered/subset input points.

    A PLY has no COLMAP IDs. Ambiguous duplicate coordinates are selected only
    if all matching COLMAP points pass. Unmatched points are an input error,
    not silently assigned background labels.
    """
    def keys(xyz):
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("Point coordinates must be finite Nx3 arrays")
        return xyz.view(np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])).reshape(-1)

    source = keys(colmap_xyz)
    query = keys(input_xyz)
    unique, inverse = np.unique(source, return_inverse=True)
    if len(unique) == 0:
        raise ValueError("COLMAP point cloud is empty")
    selected = np.ones(len(unique), dtype=bool)
    np.logical_and.at(selected, inverse, labels)
    positions = np.searchsorted(unique, query)
    if np.any(positions >= len(unique)) or np.any(unique[positions] != query):
        raise ValueError("Input PLY coordinates do not match sparse/0/points3D; "
                         "voxel masks require the corresponding original COLMAP point cloud")
    return selected[positions]


def load_voxel_seed_mask(input_xyz, source_path, images_dir, masks_dir, threshold=0):
    # Lazy import keeps the pure labelling routines usable without CUDA/PLY I/O.
    from scene.colmap_loader import (read_extrinsics_binary, read_intrinsics_binary,
                                    read_points3D_binary, read_extrinsics_text,
                                    read_intrinsics_text, read_points3D_text)
    root = Path(source_path)
    sparse = root / 'sparse' / '0'
    if (sparse / 'images.bin').is_file() and (sparse / 'cameras.bin').is_file():
        images = read_extrinsics_binary(sparse / 'images.bin')
        cameras = read_intrinsics_binary(sparse / 'cameras.bin')
    else:
        images = read_extrinsics_text(sparse / 'images.txt')
        cameras = read_intrinsics_text(sparse / 'cameras.txt')
    if (sparse / 'points3D.bin').is_file():
        xyz, _, _, ids = read_points3D_binary(sparse / 'points3D.bin', return_ids=True)
    else:
        xyz, _, _, ids = read_points3D_text(sparse / 'points3D.txt', return_ids=True)
    labels = label_mask_observations(ids, images, cameras, root / images_dir,
                                     root / masks_dir, threshold)
    labels = align_point_labels(input_xyz, xyz, labels)
    if not labels.any():
        raise ValueError("No input points selected by black voxel masks; check masks and threshold")
    return labels
