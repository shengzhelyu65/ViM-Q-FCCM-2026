#!/usr/bin/env python3
"""Download Vision Mamba checkpoints into local subfolders.

This script downloads the three Vision Mamba Hugging Face repos directly and
stores the resulting files under:

  Checkpoints/tiny/
  Checkpoints/small/
  Checkpoints/base/

Run it with plain Python.
"""

from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator, Optional


HF_HOSTS = ("huggingface.co", "www.huggingface.co")
TARGETS = {
    "tiny": "hustvl/Vim-tiny-midclstok",
    "small": "hustvl/Vim-small-midclstok",
    "base": "hustvl/Vim-base-midclstok",
}


def parse_hf_repo_id(value: str) -> Optional[str]:
    """Return a Hugging Face repo ID if the value looks like one."""
    value = value.strip().rstrip("/")
    if "/" in value and not value.startswith(("http://", "https://")):
        return value

    parsed = urllib.parse.urlparse(value)
    if parsed.netloc not in HF_HOSTS:
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 2:
        return f"{path_parts[0]}/{path_parts[1]}"
    return None


def is_hf_file_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.netloc in HF_HOSTS and "/resolve/" in parsed.path


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def encode_hf_path(path: str) -> str:
    return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))


def iter_hf_repo_files(repo_id: str, revision: str = "main") -> Iterator[str]:
    api_url = (
        f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}"
        f"?revision={urllib.parse.quote(revision, safe='')}"
    )
    payload = request_json(api_url)
    for sibling in payload.get("siblings", []):
        filename = sibling.get("rfilename")
        if filename:
            yield filename


def download_hf_file(source: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(source)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5:
        raise ValueError(f"Unexpected Hugging Face file URL: {source}")

    repo_id = f"{parts[0]}/{parts[1]}"
    revision = parts[3]
    filename = "/".join(parts[4:])
    file_url = (
        f"https://huggingface.co/{repo_id}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/{encode_hf_path(filename)}"
    )
    download_url(file_url, destination / filename)


def download_hf_repo(source: str, destination: Path) -> None:
    repo_id = parse_hf_repo_id(source)
    if repo_id is None:
        raise ValueError(f"Could not parse Hugging Face repo ID from: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    for filename in iter_hf_repo_files(repo_id):
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{encode_hf_path(filename)}"
        download_url(file_url, destination / filename)


def folder_has_files(path: Path) -> bool:
    return path.exists() and any(child.is_file() for child in path.rglob("*"))


def download_checkpoint(label: str, source: str, output_root: Path) -> None:
    destination = output_root / label
    if folder_has_files(destination):
        print(f"Skipping {label}; files already exist in {destination}")
        return

    if is_hf_file_url(source):
        download_hf_file(source, destination)
        return

    repo_id = parse_hf_repo_id(source)
    if repo_id is not None:
        download_hf_repo(source, destination)
        return

    parsed = urllib.parse.urlparse(source)
    filename = Path(parsed.path).name or f"{label}.bin"
    download_url(source, destination / filename)


def main() -> int:
    output_root = Path(__file__).resolve().parent
    output_root.mkdir(parents=True, exist_ok=True)

    for label, source in TARGETS.items():
        print(f"Downloading {label} from {source} -> {output_root / label}")
        download_checkpoint(label, source, output_root)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())