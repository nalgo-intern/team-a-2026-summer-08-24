"""Kaggleの鮮度画像約200枚に学習済みYOLOのBBoxを描画して保存する。"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from ultralytics import YOLO

KAGGLE_DATASET = "ulnnproject/food-freshness-dataset"
DEFAULT_WEIGHTS = Path("./models/yolo26m-30epochs-best.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0", help="GPU番号。CPUの場合はcpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, default=Path("data/food_freshness_evaluation"))
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def download_dataset(dataset_dir: Path) -> None:
    if image_files(dataset_dir):
        print(f"ダウンロード済みデータを使用: {dataset_dir}")
        return
    dataset_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(override=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Kaggleからダウンロード: {KAGGLE_DATASET}")
    api.dataset_download_files(KAGGLE_DATASET, path=str(dataset_dir), unzip=True, quiet=False)


def prepare_sample(dataset_dir: Path, sample_dir: Path, count: int, seed: int) -> list[Path]:
    candidates = image_files(dataset_dir)
    if not candidates:
        raise FileNotFoundError(f"画像が見つかりません: {dataset_dir}")
    selected = random.Random(seed).sample(candidates, min(count, len(candidates)))
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)
    copied: list[Path] = []
    for index, source in enumerate(selected, start=1):
        destination = sample_dir / f"{index:04d}_{source.stem}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {weights}")
    if args.num_images < 1:
        raise ValueError("--num-imagesには1以上を指定してください。")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが利用できません。CPUの場合は --device cpu を指定してください。")

    work_dir = args.work_dir.resolve()
    dataset_dir = work_dir / "dataset"
    sample_dir = work_dir / "sample_images"
    output_dir = work_dir / "bbox_results"
    download_dataset(dataset_dir)
    samples = prepare_sample(dataset_dir, sample_dir, args.num_images, args.seed)
    print(f"推論対象: {len(samples)}枚")
    print(f"使用モデル: {weights}")

    model = YOLO(str(weights))
    model.predict(source=str(sample_dir), imgsz=args.imgsz, conf=args.conf, device=args.device, save=True, project=str(work_dir), name=output_dir.name, exist_ok=True)
    print(f"BBox付き画像の保存先: {output_dir}")


if __name__ == "__main__":
    main()
