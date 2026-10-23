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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.depth_utils import depth_is_usable

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        lazy_images = getattr(args, "data_cache_mb", 0) > 0
        depths = getattr(args, "depths", "")
        masks = getattr(args, "masks", "")

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            image_root = os.path.join(args.source_path, args.images or "images")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, lazy_images=lazy_images, depths=depths)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            image_root = args.source_path
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, lazy_images=lazy_images, depths=depths)
        else:
            assert False, "Could not recognize scene type!"

        if masks:
            mask_root = os.path.join(args.source_path, masks)
            for infos in (scene_info.train_cameras, scene_info.test_cameras):
                for i, info in enumerate(infos):
                    relative_name = os.path.relpath(info.image_path, image_root)
                    mask_path = os.path.join(mask_root, os.path.splitext(relative_name)[0] + ".png")
                    if not os.path.isfile(mask_path):
                        raise FileNotFoundError("Mask missing for {}: {}".format(info.image_path, mask_path))
                    infos[i] = info._replace(mask_path=mask_path)

        self.train_depth_count = sum(depth_is_usable(c) for c in scene_info.train_cameras)
        if depths:
            print("Depth calibration: {}/{} training views eligible".format(
                self.train_depth_count, len(scene_info.train_cameras)))

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, skip_empty_masks=True)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            if masks and resolution_scale == 1.0:
                cameras = self.train_cameras[resolution_scale]
                self.train_depth_count = (sum(depth_is_usable(c) for c in cameras.cam_infos)
                                          if hasattr(cameras, "cam_infos") else sum(c.invdepthmap is not None for c in cameras))
        
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
