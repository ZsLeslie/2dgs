import gc
import json
import random
import tempfile
import unittest
import warnings
import weakref
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams
from scene.dataset_readers import CameraInfo, readColmapCameras, readCamerasFromTransforms
from utils.camera_utils import CameraDataset, cameraList_from_camInfos, loadCam
from utils.data_cache import CameraBatchSampler
from utils.image_loading import ImageSource


class FakeCamera:
    def __init__(self, index, size):
        self.index = index
        self.pixels = bytearray(size)


class TrackingDataset:
    def __init__(self, sizes):
        self.sizes = sizes
        self.live = weakref.WeakValueDictionary()
        self.peak_bytes = 0
        self.loads = 0

    def __len__(self):
        return len(self.sizes)

    def image_nbytes(self, index):
        return self.sizes[index]

    def __getitem__(self, index):
        camera = FakeCamera(index, self.sizes[index])
        self.live[id(camera)] = camera
        self.peak_bytes = max(self.peak_bytes, sum(len(c.pixels) for c in self.live.values()))
        self.loads += 1
        return camera


class BatchSamplerTests(unittest.TestCase):
    def setUp(self):
        random.seed(17)

    def test_bounded_batches_cover_every_view_each_epoch_and_release_pixels(self):
        dataset = TrackingDataset([120, 180, 100, 200, 140, 160, 120])
        sampler = CameraBatchSampler(dataset, 400 / 1024**2)
        epochs = []
        for _ in range(3):
            seen = []
            for _ in range(len(dataset)):
                camera = next(sampler)
                seen.append(camera.index)
                self.assertLessEqual(sampler.cached_bytes, 400)
                del camera
            self.assertEqual(sorted(seen), list(range(len(dataset))))
            epochs.append(seen)
        self.assertNotEqual(epochs[0], epochs[1])
        self.assertLessEqual(dataset.peak_bytes, 400)
        self.assertEqual(dataset.loads, 3 * len(dataset))

    def test_full_dataset_is_reused_when_it_fits(self):
        dataset = TrackingDataset([100] * 5)
        sampler = CameraBatchSampler(dataset, 500 / 1024**2)
        for _ in range(3):
            seen = [next(sampler).index for _ in range(len(dataset))]
            self.assertEqual(sorted(seen), list(range(len(dataset))))
        self.assertEqual(dataset.loads, len(dataset))
        self.assertEqual(dataset.peak_bytes, 500)

    def test_oversize_images_load_alone_and_warn_once(self):
        dataset = TrackingDataset([50, 400, 60, 500])
        sampler = CameraBatchSampler(dataset, 100 / 1024**2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2 * len(dataset)):
                camera = next(sampler)
                if sampler.cached_bytes > 100:
                    self.assertEqual(len(sampler.batch), 1)
                del camera
        self.assertEqual(len(caught), 1)
        self.assertLessEqual(dataset.peak_bytes, 500)

    def test_invalid_budgets_and_empty_data(self):
        for budget in (0, -1, float('nan'), float('inf')):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                CameraBatchSampler(TrackingDataset([100]), budget)
        with self.assertRaisesRegex(ValueError, "empty"):
            CameraBatchSampler(TrackingDataset([]), 1)


class ImageLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(resolution=1, data_device="cpu", data_cache_mb=1)
        # Camera transforms use CUDA independently of pixel loading. Keep these
        # tests CPU-only while exercising the real Camera and tensor conversion.
        self.cuda = patch.object(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor)
        self.cuda.start()

    def tearDown(self):
        self.cuda.stop()
        self.temp.cleanup()

    def make_image(self, name="frame.png", size=(12, 8), rgba=True):
        path = self.root / name
        pixels = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 4 if rgba else 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path)
        return path, pixels

    def info(self, image):
        return CameraInfo(3, np.eye(3), np.zeros(3), 0.8, 1.0, image,
                          str(self.root / "frame.png"), "frame", 12, 8)

    def test_metadata_does_not_decode_or_retain_file_handles(self):
        path, _ = self.make_image()
        with patch.object(Image.Image, "load", side_effect=AssertionError("unexpected decode")):
            source = ImageSource(path)
            dataset = CameraDataset([self.info(source)], 1.0, self.args)
        self.assertEqual(source.size, (12, 8))
        self.assertEqual(source.channels, 4)
        self.assertEqual(dataset.image_nbytes(0), 12 * 8 * 4 * 4)
        self.assertFalse(any(isinstance(v, Image.Image) for v in vars(source).values()))

    def test_lazy_and_eager_pixels_masks_and_byte_estimates_match(self):
        for rgba in (False, True):
            path, _ = self.make_image(rgba=rgba)
            for resolution, scale in ((1, 1.0), (2, 1.0), (1, 2.0), (6, 1.0), (-1, 1.0)):
                with self.subTest(rgba=rgba, resolution=resolution, scale=scale):
                    self.args.resolution = resolution
                    with Image.open(path) as image:
                        eager = loadCam(self.args, 0, self.info(image), scale)
                    dataset = CameraDataset([self.info(ImageSource(path))], scale, self.args)
                    lazy = dataset[0]
                    torch.testing.assert_close(lazy.original_image, eager.original_image, rtol=0, atol=0)
                    size = lazy.original_image.numel() * lazy.original_image.element_size()
                    if rgba:
                        torch.testing.assert_close(lazy.gt_alpha_mask, eager.gt_alpha_mask, rtol=0, atol=0)
                        size += lazy.gt_alpha_mask.numel() * lazy.gt_alpha_mask.element_size()
                    else:
                        self.assertIsNone(lazy.gt_alpha_mask)
                    self.assertEqual(dataset.image_nbytes(0), size)

    def test_blender_background_matches_original_compositing(self):
        path, pixels = self.make_image()
        transforms = {"camera_angle_x": 0.9, "frames": [
            {"file_path": "./frame", "transform_matrix": np.eye(4).tolist()}]}
        (self.root / "transforms_train.json").write_text(json.dumps(transforms))
        for white in (True, False):
            eager_info = readCamerasFromTransforms(str(self.root), "transforms_train.json", white)[0]
            lazy_info = readCamerasFromTransforms(str(self.root), "transforms_train.json", white, lazy_images=True)[0]
            self.assertIsInstance(lazy_info.image, ImageSource)
            self.assertEqual(lazy_info.image.channels, 3)
            normalized = pixels / 255.0
            expected = (normalized[:, :, :3] * normalized[:, :, 3:4]
                        + int(white) * (1 - normalized[:, :, 3:4])) * 255.0
            np.testing.assert_array_equal(np.asarray(eager_info.image), expected.astype(np.uint8))
            eager = loadCam(self.args, 0, eager_info, 1)
            lazy = loadCam(self.args, 0, lazy_info, 1)
            torch.testing.assert_close(lazy.original_image, eager.original_image, rtol=0, atol=0)
            self.assertIsNone(lazy.gt_alpha_mask)
            eager_info.image.close()

    def test_colmap_reader_stores_only_image_sources(self):
        path, _ = self.make_image()
        extrinsics = {1: SimpleNamespace(camera_id=1, qvec=np.array([1, 0, 0, 0]),
                                         tvec=np.zeros(3), name=path.name)}
        intrinsics = {1: SimpleNamespace(id=1, height=8, width=12,
                                         model="PINHOLE", params=[10, 10, 6, 4])}
        infos = readColmapCameras(extrinsics, intrinsics, str(self.root), lazy_images=True)
        self.assertIsInstance(infos[0].image, ImageSource)
        dataset = cameraList_from_camInfos(infos, 1, self.args)
        self.assertIsInstance(dataset, CameraDataset)
        self.assertEqual(dataset[-1].image_name, "frame")
        with self.assertRaises(IndexError):
            dataset[1]

    def test_lazy_dataset_does_not_retain_loaded_camera(self):
        path, _ = self.make_image()
        dataset = CameraDataset([self.info(ImageSource(path))], 1, self.args)
        camera = dataset[0]
        reference = weakref.ref(camera)
        del camera
        gc.collect()
        self.assertIsNone(reference())

    def test_zero_budget_returns_original_eager_list(self):
        path, _ = self.make_image()
        self.args.data_cache_mb = 0
        with Image.open(path) as image:
            cameras = cameraList_from_camInfos([self.info(image)], 1, self.args)
        self.assertIsInstance(cameras, list)
        self.assertEqual(len(cameras), 1)


class ArgumentTests(unittest.TestCase):
    def test_default_fractional_budget_and_invalid_values(self):
        parser = ArgumentParser()
        params = ModelParams(parser)
        self.assertEqual(params.extract(parser.parse_args([])).data_cache_mb, 0)
        self.assertEqual(params.extract(parser.parse_args(["--data_cache_mb", "1.5"])).data_cache_mb, 1.5)
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                params.extract(parser.parse_args(["--data_cache_mb", value]))

    def test_legacy_config_without_cache_setting(self):
        parser = ArgumentParser()
        params = ModelParams(parser, sentinel=True)
        self.assertEqual(params.extract(SimpleNamespace(source_path=".")).data_cache_mb, 0)


if __name__ == "__main__":
    unittest.main()
