"""Memory-budgeted camera batches for training and sequential reconstruction."""

import math
import random
import warnings
from collections import deque


class CameraBatchIterator:
    """One ordered pass over cameras, loading pixel-budgeted batches.

    Uses the same image_nbytes/MiB budget as CameraBatchSampler. The caller
    must release the returned camera before advancing to the next batch.
    Model, render outputs, TSDF storage and decoding temporaries are excluded.
    A camera larger than the budget is loaded alone.
    """

    def __init__(self, cameras, cache_mb):
        if not math.isfinite(cache_mb) or cache_mb <= 0:
            raise ValueError("Camera batch cache must be finite and positive")
        self.cameras = cameras
        self.budget_bytes = max(1, int(cache_mb * 1024 * 1024))
        self.next_index = 0
        self.batch = deque()
        self.warned_oversize = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch:
            if self.next_index == len(self.cameras):
                raise StopIteration
            cached_bytes = 0
            while self.next_index < len(self.cameras):
                size = self.cameras.image_nbytes(self.next_index)
                if self.batch and cached_bytes + size > self.budget_bytes:
                    break
                if size > self.budget_bytes and not self.warned_oversize:
                    warnings.warn(
                        "An image exceeds --data_cache_mb; loading it alone "
                        "({:.2f} MiB). Increase the budget or reduce the image resolution.".format(size / 1024**2)
                    )
                    self.warned_oversize = True
                self.batch.append(self.cameras[self.next_index])
                self.next_index += 1
                cached_bytes += size
        return self.batch.popleft()


class CameraBatchSampler:
    """Shuffle each epoch and load successive batches within a pixel budget.

    The dataset must support len(), indexing and image_nbytes(index). The
    caller must release the previous camera before requesting the next one.
    Decoding temporaries, camera matrices and model memory are not budgeted.
    """

    def __init__(self, cameras, cache_mb):
        if not math.isfinite(cache_mb) or cache_mb <= 0:
            raise ValueError("Camera batch cache must be finite and positive")
        if not len(cameras):
            raise ValueError("Cannot train with an empty camera dataset")
        self.cameras = cameras
        self.budget_bytes = max(1, int(cache_mb * 1024 * 1024))
        self.sizes = [cameras.image_nbytes(i) for i in range(len(cameras))]
        self.all_fit = sum(self.sizes) <= self.budget_bytes
        self.pending = []
        self.batch = {}
        self.batch_order = []
        self.cached_bytes = 0
        self.warned_oversize = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.batch_order:
            if not self.pending:
                self.pending = list(range(len(self.cameras)))
                random.shuffle(self.pending)
                if self.all_fit and self.batch:
                    self.batch_order, self.pending = self.pending, []
                    return self.batch[self.batch_order.pop()]

            # Drop all previous pixels before allocating the next batch.
            self.batch.clear()
            self.cached_bytes = 0
            while self.pending:
                index = self.pending[-1]
                size = self.sizes[index]
                if self.batch and self.cached_bytes + size > self.budget_bytes:
                    break
                if size > self.budget_bytes and not self.warned_oversize:
                    warnings.warn(
                        "A training image exceeds --data_cache_mb; loading it alone "
                        "({:.2f} MiB). Increase the budget or reduce the image resolution.".format(size / 1024**2)
                    )
                    self.warned_oversize = True
                self.batch[index] = self.cameras[index]
                self.cached_bytes += size
                self.batch_order.append(self.pending.pop())

        return self.batch[self.batch_order.pop()]
