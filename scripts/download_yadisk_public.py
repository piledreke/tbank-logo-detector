import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from urllib.parse import quote
import urllib.request


YA_BASE = "https://cloud-api.yandex.net/v1/disk/public/resources"


def api_get_json(url: str) -> Dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))


def get_download_href(public_link: str, path: str) -> str:
    public_key = quote(public_link, safe="")
    remote_path = quote(path, safe="") if path else ""
    url = f"{YA_BASE}/download?public_key={public_key}"
    if remote_path:
        url += f"&path={remote_path}"
    meta = api_get_json(url)
    href = meta.get("href")
    if not href:
        raise RuntimeError(f"No download href for path={path}; response={meta}")
    return href


def download_file(href: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(href) as r, open(dest_path, "wb") as f:
        f.write(r.read())


def list_dir(public_link: str, path: str) -> List[Dict]:
    public_key = quote(public_link, safe="")
    remote_path = quote(path, safe="") if path else ""
    url = f"{YA_BASE}?public_key={public_key}"
    if remote_path:
        url += f"&path={remote_path}"
    url += "&limit=1000"
    meta = api_get_json(url)
    emb = meta.get("_embedded", {})
    return emb.get("items", [])


def download_folder(public_link: str, remote_path: str, dest_dir: Path) -> None:
    items = list_dir(public_link, remote_path)
    for item in items:
        item_type = item.get("type")
        item_name = item.get("name")
        item_path = item.get("path")  # full path on YaDisk
        if not item_name or not item_path:
            continue
        if item_type == "dir":
            # recurse
            sub_remote = item_path
            sub_dest = dest_dir / item_name
            download_folder(public_link, sub_remote, sub_dest)
        else:
            href = get_download_href(public_link, item_path)
            dest_path = dest_dir / item_name
            print(f"Downloading: {item_path} -> {dest_path}")
            download_file(href, dest_path)


def main() -> None:
    p = argparse.ArgumentParser(description="Download public folder/files from Yandex Disk")
    p.add_argument("--public-link", required=True, help="Public link to YaDisk folder or file")
    p.add_argument("--remote-path", default="", help="Remote path inside public resource (e.g. 'test/images')")
    p.add_argument("--dest", required=True, help="Local destination directory")
    args = p.parse_args()

    dest_dir = Path(args.dest)

    # Try to list dir; if it fails, assume it's a single file
    try:
        items = list_dir(args.public_link, args.remote_path)
        if items:
            download_folder(args.public_link, args.remote_path, dest_dir)
            return
    except Exception:
        pass

    # Single file fallback
    href = get_download_href(args.public_link, args.remote_path)
    filename = Path(args.remote_path).name or "download.bin"
    dest_path = dest_dir / filename
    print(f"Downloading file: {args.remote_path or args.public_link} -> {dest_path}")
    download_file(href, dest_path)


if __name__ == "__main__":
    main()


