import asyncio
import dataclasses
from hashlib import sha1
import typing as T

import aiohttp

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


async def fetch_imagenet_index(
    session: aiohttp.ClientSession,
    url: str,
    category: str,
) -> T.List[Image]:
    logger.debug(f'fetching imagenet index {url}')

    async with session.get(url) as resp:
        resp.raise_for_status()

        d = await resp.text()

        return [
            Image(category=category, url=url.strip())
            for url in d.split('\n')
            if url.strip()
        ]


async def fetch_image(
    session: aiohttp.ClientSession,
    config: Config,
    image: Image,
):
    logger.debug(f'fetching image: {image.url}')

    image_path = config.get_image_path(image.get_image_name())
    async with session.get(image.url) as resp:
        with open(image_path, 'wb') as f:
            try:
                f.write(await resp.read())
                logger.debug(f'saved image: {image_path}')
            except Exception as e:
                logger.error(f'download failed: {image.url}', e)


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


async def download_index(
    session: aiohttp.ClientSession,
    config: Config,
    url: str,
    category: str,
):
    images = await fetch_imagenet_index(session, url, category)

    image_index_path = config.get_image_path(f'{category}.txt')
    with open(image_index_path, 'w') as f:
        content = '\n'.join(' '.join(i.url, i.get_image_name())
                            for i in images)
        f.write(content)

    await asyncio.wait([
        fetch_image(session, config, image)
        for image in images
    ])


async def main():
    config = Config
    config.ensure_data_paths()

    async with aiohttp.ClientSession() as session:
        await asyncio.wait([
            download_index(session, config, url, category)
            for url, category in IMAGENET_INDICES
        ])


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
