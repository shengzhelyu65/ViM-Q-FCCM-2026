import os
import re
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = DATA_ROOT / ".cache"
CACHE_DATASETS = CACHE_ROOT / "datasets"
OUTPUT_ROOT = DATA_ROOT
SPLITS = ["train", "val"]

os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "huggingface" / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))

from datasets import Dataset, Image, concatenate_datasets


def target_name_train(src_path: str) -> str:
    src_name = Path(src_path).name
    m = re.match(r"^(n\d+)_(\d+)", src_name, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Unexpected train image path format: {src_path}")
    synset = m.group(1).lower()
    image_id = m.group(2)
    return f"{synset}_{image_id}.jpeg"


def target_name_val(src_path: str) -> str:
    src_name = Path(src_path).name
    m = re.match(r"^(ILSVRC2012_val_\d+)", src_name, flags=re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Unexpected validation image path format: {src_path}")
    return f"{m.group(1).lower()}.jpeg"


def load_split_from_local_prepared_cache(cache_dir: Path, split: str):
    prefix = "imagenet-1k-train-" if split == "train" else "imagenet-1k-validation-"
    pattern = f"ILSVRC___imagenet-1k/default/0.0.0/*/{prefix}*.arrow"
    arrow_files = sorted(cache_dir.glob(pattern))
    if not arrow_files:
        return None

    shards = [Dataset.from_file(str(path)) for path in arrow_files]
    if len(shards) == 1:
        return shards[0]
    return concatenate_datasets(shards)


def convert_split(ds, split: str, output_root: Path) -> int:
    out_split = output_root / ("train" if split == "train" else "val")
    out_split.mkdir(parents=True, exist_ok=True)
    ds = ds.cast_column("image", Image(decode=False))

    count = 0
    for item in ds:
        label = int(item["label"])
        image_rec = item["image"]
        src_path = image_rec["path"]
        image_bytes = image_rec["bytes"]

        filename = target_name_train(src_path) if split == "train" else target_name_val(src_path)

        class_dir = out_split / f"{label:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        target = class_dir / filename

        if not target.exists():
            with target.open("wb") as f:
                f.write(image_bytes)

        count += 1
        if count % 20000 == 0:
            print(f"[{split}] processed {count}")

    print(f"[{split}] done. total={count}")
    return count


def main() -> None:
    train_ds = load_split_from_local_prepared_cache(CACHE_DATASETS, "train")
    val_ds = load_split_from_local_prepared_cache(CACHE_DATASETS, "val")

    if train_ds is None or val_ds is None:
        raise RuntimeError(
            "Prepared Arrow cache missing. Run `python Data/download_raw_data.py` first."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    convert_split(train_ds, "train", OUTPUT_ROOT)
    convert_split(val_ds, "validation", OUTPUT_ROOT)

    print("Conversion completed.")
    print(f"Output train folder: {(OUTPUT_ROOT / 'train').resolve()}")
    print(f"Output val folder: {(OUTPUT_ROOT / 'val').resolve()}")


if __name__ == "__main__":
    main()
