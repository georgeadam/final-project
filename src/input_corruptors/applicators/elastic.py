import cv2
import numpy as np
import torch
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian

from .applicator import Applicator
from .creation import applicators


class Elastic(Applicator):
    def corrupt_single_sample(self, x):
        x = torch.tensor(x)
        x = self.transform(x)

        image = np.array(x, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]
        height = shape[0]

        c = [(height * 2, height * 0.7, height * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
             (height * 2, height * 0.08, height * 0.2), (height * 0.05, height * 0.01, height * 0.02),
             (height * 0.07, height * 0.01, height * 0.02), (height * 0.12, height * 0.01, height * 0.02)][
            self.severity - 1]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(
            np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1], mode='reflect', truncate=3) * c[0]).astype(
            np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


applicators.register_builder("elastic", Elastic)
