from hashlib import md5
import os

from hotdog.logging import make_logger
from hotdog.config import Config


logger = make_logger('clean_data', 'info')


def is_flicker_invalid_image(p: str) -> bool:
    with open(p, 'rb') as f:
        content = f.read(4096)
    checksum = md5(content).hexdigest()
    # This photo is no longer avaiable
    return checksum == '880a7a58e05d3e83797f27573bb6d35c'


def is_image(p: str) -> bool:
    tc = bytearray(
        {7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f}
    )
    with open(p, 'rb') as f:
        return bool(f.read(1024).translate(None, tc))


def delete_image(p: str):
    logger.debug(f'removing {p}')
    os.remove(p)


def main():
    config = Config

    for _, _, files in os.walk(config.get_image_path()):
        for f in files:
            p = config.get_image_path(f)
            if p.endswith('txt'):
                # index file
                continue
            if not is_image(p):
                logger.info(f'not an image: {f}, should delete')
                delete_image(p)
            elif is_flicker_invalid_image(p):
                logger.info(f'is flicker invalid image: {f}, should delete')
                delete_image(p)
            else:
                logger.debug(f'{f} is a valid image')


if __name__ == '__main__':
    main()
