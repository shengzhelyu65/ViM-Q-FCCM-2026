import os
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = DATA_ROOT / ".cache"
CACHE_DATASETS = CACHE_ROOT / "datasets"
SPLITS = ["train", "validation"]

os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "huggingface" / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))

from datasets import DownloadConfig, load_dataset


def load_hf_token() -> str | None:
    token_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for token_path in token_paths:
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
    return None


def cache_has_splits(cache_dir: Path, splits: list[str]) -> bool:
    if not cache_dir.is_dir():
        return False

    for split in splits:
        pattern = f"ILSVRC___imagenet-1k/default/0.0.0/*/imagenet-1k-{split}-*.arrow"
        if not any(cache_dir.glob(pattern)):
            return False
    return True


def cleanup_source_blobs(cache_dir: Path) -> int:
    removed = 0
    for file_path in cache_dir.iterdir():
        if file_path.is_file():
            file_path.unlink()
            removed += 1
    return removed


def main() -> None:
    CACHE_DATASETS.mkdir(parents=True, exist_ok=True)

    if cache_has_splits(CACHE_DATASETS, SPLITS):
        removed = cleanup_source_blobs(CACHE_DATASETS)
        print(
            f"Cache already has prepared splits {SPLITS}. "
            f"Cleaned {removed} source-blob files from cache root."
        )
        return

    hf_token = load_hf_token()
    if hf_token is None:
        raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login` first.")

    print(f"Preparing splits: {SPLITS}")
    load_dataset(
        "ILSVRC/imagenet-1k",
        split=SPLITS,
        token=hf_token,
        cache_dir=str(CACHE_DATASETS),
        download_config=DownloadConfig(cache_dir=str(CACHE_DATASETS)),
    )

    removed = cleanup_source_blobs(CACHE_DATASETS)
    print(f"Prepared Arrow cache under: {CACHE_DATASETS.resolve()}")
    print(f"Cleaned {removed} source-blob files from cache root.")
    print("Next step: python Data/convert_raw_data.py")


if __name__ == "__main__":
    main()
