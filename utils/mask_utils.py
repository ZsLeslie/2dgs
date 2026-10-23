"""Binary supervision masks, independent of image opacity."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_valid_mask(path, image_size, resolution):
    """Read one mask, validate alignment and resize with nearest-neighbor.

    Black is invalid; any nonzero grayscale/RGB value is valid. Alpha in a
    mask file is deliberately ignored. Return a CPU bool tensor [1, H, W].
    """
    with Image.open(path) as image:
        if image.size != image_size:
            raise ValueError("Mask {} has size {}; expected RGB size {}".format(path, image.size, image_size))
        if image.mode in ("P", "RGB", "RGBA"):
            pixels = np.asarray(image.convert("RGB"))
            valid = np.any(pixels != 0, axis=-1)
        else:
            pixels = np.asarray(image)
            if pixels.ndim == 3:  # LA: ignore the alpha channel
                pixels = pixels[..., 0]
            valid = pixels != 0
        resized = Image.fromarray(valid.astype(np.uint8)).resize(resolution, Image.Resampling.NEAREST)
        return torch.from_numpy(np.array(resized, dtype=bool)[None])


def interior_mask(mask, window_size):
    """Keep only centers whose entire window is inside the valid image area."""
    if window_size < 1 or window_size % 2 != 1:
        raise ValueError("Mask window size must be a positive odd number")
    radius = window_size // 2
    invalid = (~mask.bool()).float()
    invalid = F.pad(invalid, (radius, radius, radius, radius), value=1)
    return F.max_pool2d(invalid, window_size, stride=1) == 0
