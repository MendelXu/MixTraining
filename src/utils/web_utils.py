import os
import requests

shortname = {"mmdet": "https://raw.githubusercontent.com/open-mmlab/mmdetection/master"}


def load_text_from_web(url):
    header, path = url.split(":")
    if header in shortname:
        header = shortname[header]
        url = os.path.join(header, path)
    r = requests.get(url, stream=True)
    return r.content.decode(r.encoding)


def check_url_exist(url):
    header, path = url.split(":")
    if header in shortname:
        header = shortname[header]
        url = os.path.join(header, path)

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise FileNotFoundError(f"{url} does not exist")
