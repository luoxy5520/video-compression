"""Publish a server-home copy of the report while preserving local anchors."""
from html.parser import HTMLParser
from pathlib import Path
import re


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs["id"])
        key = {"a":"href","video":"src","source":"src"}.get(tag)
        if key and key in attrs:
            self.paths.append(attrs[key])


def publish(home):
    repo = home/"dit-paired-experiment"
    page = (repo/"实验地图.html").read_text()
    assets = {"初版实验.pdf","senior-oracle-dataset-kit-20261106/experiment_report.json"}

    def prefix(match):
        attr, path = match.groups()
        if path.startswith(("#","https://","http://")):
            return match.group(0)
        folder = "experiment_map_assets" if path in assets else "dit-paired-experiment"
        return f'{attr}="{folder}/{path}"'

    page = re.sub(r'(href|src)="([^"]+)"',prefix,page)
    resources = Resources()
    resources.feed(page)
    for path in resources.paths:
        if path.startswith("#"):
            if path[1:] not in resources.ids:
                raise ValueError(f"Missing anchor: {path}")
        elif not path.startswith(("https://","http://")) and not (home/path).is_file():
            raise FileNotFoundError(home/path)
    tmp = home/".实验地图.html.tmp"
    tmp.write_text(page)
    tmp.replace(home/"实验地图.html")


if __name__ == "__main__":
    publish(Path("/home/luoxiangyu"))
