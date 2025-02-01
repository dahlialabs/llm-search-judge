import requests
import PIL.Image
import io
import os
import logging

logger = logging.getLogger(__name__)


def fetch_and_resize(url, option_id, width=512, height=1024, dest='~/.local-llm-judge/img/'):
    if url is None:
        return None
    dest = os.path.expanduser(dest)
    if not os.path.exists(dest):
        os.makedirs(dest)

    if os.path.exists(f"{dest}/{option_id}.png"):
        logger.debug(f"Image {dest}/{option_id}.png already exists, returning path")
        return f"{dest}/{option_id}.png"

    logger.debug(f"Fetching image from {url} for option {option_id} to {dest}")
    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    image = image.resize((width, height))
    image.save(f"{dest}/{option_id}.png")
    return f"{dest}/{option_id}.png"
