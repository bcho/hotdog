import pathlib
from os import path

from .logging import make_logger


logger = make_logger(__name__)


class Config:
    """configuration"""

    ROOT = path.dirname(path.dirname(__file__))

    @classmethod
    def get_image_path(cls, *args) -> str:
        return path.join(cls.ROOT, 'data', 'image', *args)

    @classmethod
    def ensure_data_paths(cls):
        """Create required data paths."""
        for p in (
                cls.ROOT,
                cls.get_image_path(),
        ):
            logger.debug(f'creating data path: {p}')
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)
