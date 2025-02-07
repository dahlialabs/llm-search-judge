import requests
import PIL.Image
from PIL import UnidentifiedImageError
import io
import os
import logging

logger = logging.getLogger(__name__)


def img_proxy_url(image_path):
    if image_path.startswith('products/'):
        # Then change url to img proxy
        image_path = image_path.replace('products/', 'products://')
    if image_path.startswith('/products/'):
        # Then change url to img proxy
        image_path = image_path.replace('/products/', 'products://')
    base_url = "https://cdn.stag.dahlialabs.dev/fotomancer/_/rs:fit:512:1024/plain/"
    image_path = f"{base_url}{image_path}"
    logger.debug(f"Using img proxy url: {image_path}")
    return image_path


def fetch_and_resize(url, option_id, width=512, height=1024, dest='~/.local-llm-judge/img/'):
    if url is None:
        return None
    if url.startswith('products/'):
        url = img_proxy_url(url)
        logger.debug(f"Using img proxy url: {url}")
    dest = os.path.expanduser(dest)
    if not os.path.exists(dest):
        os.makedirs(dest)

    if os.path.exists(f"{dest}/{option_id}.png"):
        logger.debug(f"Image {dest}/{option_id}.png already exists, returning path")
        return f"{dest}/{option_id}.png"

    logger.debug(f"Fetching image from {url} for option {option_id} to {dest}")
    response = requests.get(url)
    try:
        image = PIL.Image.open(io.BytesIO(response.content))
        image = image.resize((width, height))
        image.save(f"{dest}/{option_id}.png")
        return f"{dest}/{option_id}.png"
    except UnidentifiedImageError:
        logger.error(f"Failed to fetch image from {url} for option {option_id} due to UnidentifiedImageError")
        return None
