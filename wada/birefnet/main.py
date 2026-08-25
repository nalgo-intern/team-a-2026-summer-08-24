"""Kaggle画像約200枚からBiRefNetで前景を抽出し、BBox・マスク・透過PNGを保存する。"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

KAGGLE_DATASET = "ulnnproject/food-freshness-dataset"
DEFAULT_MODEL = "ZhengPeng7/BiRefNet"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5, help="前景マスクの二値化閾値")
    parser.add_argument("--min-area-ratio", type=float, default=0.005, help="画像面積に対する最小前景面積")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
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
    copied = []
    for index, source in enumerate(selected, start=1):
        destination = sample_dir / f"{index:04d}_{source.stem}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def largest_component(mask: np.ndarray, min_pixels: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    if int(area) < min_pixels:
        return np.zeros_like(mask), None
    result = np.where(labels == component, 255, 0).astype(np.uint8)
    return result, (int(x), int(y), int(x + width - 1), int(y + height - 1))


def prediction_tensor(output: object) -> torch.Tensor:
    while isinstance(output, (list, tuple)):
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"未対応のBiRefNet出力形式です: {type(output)}")
    return output


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-imagesには1以上を指定してください。")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--thresholdは0より大きく1未満にしてください。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが利用できません。CPUの場合は --device cpu を指定してください。")

    work_dir = args.work_dir.resolve()
    dataset_dir = work_dir / "dataset"
    sample_dir = work_dir / "sample_images"
    bbox_dir = work_dir / "sod_bbox_results"
    mask_dir = work_dir / "sod_masks"
    foreground_dir = work_dir / "sod_foregrounds"
    for directory in (bbox_dir, mask_dir, foreground_dir):
        directory.mkdir(parents=True, exist_ok=True)

    download_dataset(dataset_dir)
    samples = prepare_sample(dataset_dir, sample_dir, args.num_images, args.seed)
    print(f"BiRefNetを読み込み: {args.model}")
    model = AutoModelForImageSegmentation.from_pretrained(args.model, trust_remote_code=True)
    model.to(args.device).eval()
    if args.device == "cpu":
        model.float()
    model_dtype = next(model.parameters()).dtype
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    rows: list[dict[str, object]] = []
    for index, image_path in enumerate(samples, start=1):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device=args.device, dtype=model_dtype)
        with torch.inference_mode():
            logits = prediction_tensor(model(tensor))
            probability = logits.sigmoid()[0, 0].float().cpu().numpy()
        probability = cv2.resize(probability, image.size, interpolation=cv2.INTER_LINEAR)
        binary = np.where(probability >= args.threshold, 255, 0).astype(np.uint8)
        mask, box = largest_component(binary, max(1, int(image.width * image.height * args.min_area_ratio)))

        stem = image_path.stem
        Image.fromarray(mask).save(mask_dir / f"{stem}.png")
        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(mask))
        rgba.save(foreground_dir / f"{stem}.png")
        boxed = image.copy()
        if box is not None:
            ImageDraw.Draw(boxed).rectangle(box, outline=(255, 0, 0), width=max(2, min(image.size) // 100))
        boxed.save(bbox_dir / f"{stem}.jpg", quality=95)
        rows.append({"image": image_path.name, "detected": box is not None, "bbox": "" if box is None else ",".join(map(str, box))})
        print(f"[{index}/{len(samples)}] {image_path.name}: {box}")

    with (work_dir / "sod_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("image", "detected", "bbox"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"BBox画像: {bbox_dir}")
    print(f"マスク: {mask_dir}")
    print(f"透過PNG: {foreground_dir}")


if __name__ == "__main__":
    main()
