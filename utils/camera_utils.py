#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, getWorld2View2
from utils.image_loading import ImageSource
from utils.depth_utils import depth_is_usable, load_inverse_depth
from collections.abc import Sequence
import torch

WARNED = False

def camera_resolution(args, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return resolution


def loadCam(args, id, cam_info, resolution_scale):
    resolution = camera_resolution(args, cam_info, resolution_scale)
    lazy = isinstance(cam_info.image, ImageSource)
    image = cam_info.image.open() if lazy else cam_info.image
    try:
        if len(image.getbands()) > 3:
            channels = image.split()
            gt_image = torch.cat([PILtoTorch(im, resolution) for im in channels[:3]], dim=0)
            loaded_mask = PILtoTorch(channels[3], resolution)
        else:
            gt_image = PILtoTorch(image, resolution)
            loaded_mask = None
    finally:
        if lazy:
            image.close()

    invdepthmap, depth_mask = load_inverse_depth(cam_info, resolution, loaded_mask)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  invdepthmap=invdepthmap, depth_mask=depth_mask)

class CameraDataset(Sequence):
    """List-like camera metadata; indexing loads only the requested image."""

    def __init__(self, cam_infos, resolution_scale, args):
        self.cam_infos = cam_infos
        self.resolution_scale = resolution_scale
        self.args = args
        # Include alpha and account for PILtoTorch's floating-point output.
        pixel_bytes = torch.tensor([], dtype=torch.get_default_dtype()).element_size()
        self.image_sizes = []
        for info in cam_infos:
            width, height = camera_resolution(args, info, resolution_scale)
            bytes_per_pixel = min(info.image.channels, 4) * pixel_bytes
            if depth_is_usable(info):
                bytes_per_pixel += 8  # float32 inverse depth + float32 validity mask
            self.image_sizes.append(width * height * bytes_per_pixel)

    def __len__(self):
        return len(self.cam_infos)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return loadCam(self.args, index, self.cam_infos[index], self.resolution_scale)

    def image_nbytes(self, index):
        return self.image_sizes[index]

    def camera_to_worlds(self):
        """Match Camera's default transforms without decoding image pixels."""
        return np.array([np.linalg.inv(getWorld2View2(info.R, info.T))
                         for info in self.cam_infos])


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    if getattr(args, "data_cache_mb", 0) > 0:
        return CameraDataset(cam_infos, resolution_scale, args)
    return [loadCam(args, id, c, resolution_scale) for id, c in enumerate(cam_infos)]

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
