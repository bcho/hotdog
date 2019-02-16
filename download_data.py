import dataclasses
from hashlib import sha1
import typing as T

import requests

from hotdog.logging import make_logger
from hotdog.config import Config


logger = make_logger('download_data', 'debug')


@dataclasses.dataclass
class Image:
    # category of the image (i.e. pets/food...)
    category: str
    # url of the image
    url: str

    def get_image_name(self) -> str:
        h = sha1(self.url.encode('u8')).hexdigest()[:8]
        return f'{self.category}-{h}'


def fetch_imagenet_index(
    url: str,
    category: str,
) -> T.List[Image]:
    logger.debug(f'fetching imagenet index {url}')

    resp = requests.get(url)
    resp.raise_for_status()
    d = resp.text

    return [
        Image(category=category, url=url.strip())
        for url in d.split('\n')
        if url.strip()
    ]


def fetch_image(
    config: Config,
    image: Image,
):
    logger.debug(f'fetching image: {image.url}')

    image_path = config.get_image_path(image.get_image_name())
    try:
        resp = requests.get(image.url, timeout=10)
        resp.raise_for_status()
        if not resp.headers['content-type'].startswith('image'):
            return
        with open(image_path, 'wb') as f:
                f.write(resp.content)
                logger.debug(f'saved image: {image.url} {image_path}')
    except Exception as e:
        logger.error(f'download failed: {image.url}')
        logger.error(e)


IMAGENET_INDICES = [
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894',
     'pets'),
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725',
     'furniture'),
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152',
     'people'),
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265',
     'food'),
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019',
     'frankfurter'),
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105',
     'chili-dog'),
    ('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537',
     'hotdog'),
]


def download_index(
    config: Config,
    url: str,
    category: str,
):
    images = fetch_imagenet_index(url, category)

    image_index_path = config.get_image_path(f'{category}.txt')
    with open(image_index_path, 'w') as f:
        content = '\n'.join(' '.join([i.url, i.get_image_name()])
                            for i in images)
        f.write(content)

    for image in images:
        fetch_image(config, image)


def main():
    config = Config
    config.ensure_data_paths()

    for url, category in IMAGENET_INDICES:
        download_index(config, url, category)


if __name__ == '__main__':
    main()
