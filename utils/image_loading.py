"""Image metadata that can be kept without decoded pixels or open files."""

import numpy as np
from PIL import Image


class ImageSource:
    def __init__(self, path, white_background=None):
        self.path = path
        self.white_background = white_background
        with Image.open(path) as image:
            self.size = image.size
            self.channels = len(image.getbands()) if white_background is None else 3

    def open(self):
        """Return an image owned by the caller, with Blender alpha composited."""
        if self.white_background is None:
            return Image.open(self.path)
        with Image.open(self.path) as image:
            pixels = np.array(image.convert("RGBA")) / 255.0
        background = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])
        rgb = pixels[:, :, :3] * pixels[:, :, 3:4] + background * (1 - pixels[:, :, 3:4])
        return Image.fromarray(np.array(rgb * 255.0, dtype=np.uint8))
