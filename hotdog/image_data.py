"""image data loader"""

import dataclasses
import glob
import typing as T

import cv2
import numpy as np

from .config import Config
from .image import normalize_image, rotate_image


@dataclasses.dataclass
class Image:
    # category of the image (i.e. pets/food...)
    labels: T.List[str]
    # path of the image
    path: str
    # loaded image data
    image_data: T.Optional[T.Any] = None

    def load_image_data(self, refresh: bool = False) -> T.Any:
        """load image data with opencv

        Args:
            refresh: force invalidate cache
        Returns:
            loaded image data
        """
        if self.image_data is not None and not refresh:
            return self.image_data
        # NOTE: convert to grayscale on reading
        self.image_data = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        return self.image_data

    def show_image_data(self):
        """show image data"""
        cv2.imshow('image', self.load_image_data())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __str__(self) -> str:
        return f'Image(labels={self.labels}, path={self.path})'

    def __repr__(self) -> str:
        return self.__str__()


MakeLabels = T.Callable[[str], T.List[str]]


def prefix_label(x: str) -> T.List[str]:
    filename = x.split('/')[-1]
    category = filename.split('-')[0].strip()
    return [
        f'category={category}'
    ]


def load(
    config: Config,
    make_labels: MakeLabels,
) -> T.List[Image]:
    """load list of tagged images

    Args:
        config: configuration object
        make_labels: callback for generating image labels from image name
    Returns:
        list of images
    """
    image_path = config.get_image_path('**')

    return [
        Image(
            labels=make_labels(p),
            path=p,
            image_data=None,
        )
        for p in glob.glob(image_path)
        # TODO: maybe check with `is_image`
        if not p.endswith('txt')
    ]


def augment(
    from_images: T.List[Image],
    count: int,
    image_size: T.Tuple[int, int]
) -> T.List[Image]:
    """Augment images with given count & size.

    Args:
        from_images: images source
        count: count required
        image_size: image x, y size tuple, all images will be resized
    Returns:
        list of augmented images
    """
    def get_images():
        i = 0
        for image in from_images:
            yield image
            i = i + 1
            if i >= count:
                return
        # augmented images
        while i < count:
            image = from_images[np.random.randint(0, len(from_images))]
            yield Image(
                labels=image.labels,
                path=image.path,
                image_data=rotate_image(image.load_image_data()),
            )
            i = i + 1

    return [
        Image(
            labels=image.labels,
            path=image.path,
            image_data=normalize_image(image.load_image_data(), image_size)
        )
        for image in get_images()
    ]
