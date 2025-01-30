import requests
import PIL.Image
import io
import os


def fetch_and_resize(url, option_id, width=512, height=1024, dest='~/.local-llm-judge/img/'):
    if url is None:
        return None
    dest = os.path.expanduser(dest)
    if not os.path.exists(dest):
        os.makedirs(dest)

    if os.path.exists(f"{dest}/{option_id}.png"):
        return f"{dest}/{option_id}.png"

    response = requests.get(url)
    image = PIL.Image.open(io.BytesIO(response.content))
    image = image.resize((width, height))
    image.save(f"{dest}/{option_id}.png")
    return f"{dest}/{option_id}.png"
