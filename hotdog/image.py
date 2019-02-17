"""image processing"""

import typing as T

import cv2
from skimage import exposure
import numpy as np


def is_image(p: str) -> bool:
    tc = bytearray(
        {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f}
    )
    with open(p, 'rb') as f:
        if not bool(f.read(1024).translate(None, tc)):
            return False

    return bool(cv2.imread(p) is not None)


def rotate_image(image, angle: int):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))


def normalize_image(image, image_size: T.Tuple[int, int]):
    angle = np.random.randint(0, 360)
    image = rotate_image(image, angle)
    image = cv2.blur(image, (5, 5))
    image = cv2.resize(image, image_size)
    image = exposure.equalize_hist(image)
    image = image.reshape(image.shape + (1, ))
    return image
