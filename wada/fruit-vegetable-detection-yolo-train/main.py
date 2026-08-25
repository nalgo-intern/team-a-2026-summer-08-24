"""果物・野菜9品目をYOLO26で学習し、W&Bで比較する実行スクリプト。"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
import yaml
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image, ImageDraw
from ultralytics import YOLO

KAGGLE_DATASET = "henningheyen/lvis-fruits-and-vegetables-dataset"
WANDB_PROJECT = "fruit-vegetable-detection"
TARGET_CLASS_NAMES = ["strawberry", "potato", "tomato", "cucumber", "bell pepper", "apple", "banana", "orange", "carrot"]
SOURCE_ALIASES = {
    "strawberry": {"strawberry"}, "potato": {"potato"}, "tomato": {"tomato"},
    "cucumber": {"cucumber/cuke"}, "bell pepper": {"bell pepper/capsicum"},
    "apple": {"apple"}, "banana": {"banana"}, "orange": {"orange/orange fruit"}, "carrot": {"carrot"},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--models", default="yolo26n,yolo26s,yolo26m,yolo26l,yolo26x", help="カンマ区切りのモデル名")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--work-dir", type=Path, default=Path("data"))
    return parser.parse_args()


def require_environment() -> None:
    load_dotenv(override=True)
    required = ("KAGGLE_USERNAME", "KAGGLE_KEY", "WANDB_API_KEY")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise EnvironmentError(f".env に必要な値がありません: {', '.join(missing)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA利用可否: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPUが必要です。CUDA対応PyTorchとGPU環境を確認してください。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def labels_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_index = parts.index("images")
    return Path(*parts[:image_index], "labels", *parts[image_index + 1:]).with_suffix(".txt")


def download_dataset(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not list(raw_dir.rglob("data.yaml")):
        api = KaggleApi()
        api.authenticate()
        print(f"Kaggleからダウンロード: {KAGGLE_DATASET}")
        api.dataset_download_files(KAGGLE_DATASET, path=str(raw_dir), unzip=True, quiet=False)
    for archive in list(raw_dir.rglob("*.zip")):
        output_dir = archive.with_suffix("")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            with ZipFile(archive) as zf:
                zf.extractall(output_dir)
    candidates = list(raw_dir.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError("展開後の data.yaml が見つかりません。")
    return candidates[0]


def resolve_split(source_yaml: Path, source_data: dict, raw_dir: Path, split: str) -> Path | None:
    if source_data.get(split):
        relative = Path(str(source_data[split]))
        if relative.is_absolute() and relative.exists():
            return relative
        root = Path(str(source_data.get("path", ".")))
        for base in (source_yaml.parent / root, source_yaml.parent):
            candidate = (base / relative).resolve()
            if candidate.exists():
                return candidate
    candidates = [path for path in raw_dir.rglob("images") if path.is_dir() and split in path.as_posix().lower()]
    return candidates[0] if candidates else None


def prepare_dataset(source_yaml: Path, raw_dir: Path, prepared_dir: Path) -> tuple[Path, dict[str, Path]]:
    with source_yaml.open(encoding="utf-8") as file:
        source_data = yaml.safe_load(file)
    source_names = source_data["names"]
    if isinstance(source_names, dict):
        source_names = [source_names[key] for key in sorted(source_names, key=int)]
    source_splits = {split: resolve_split(source_yaml, source_data, raw_dir, split) for split in ("train", "val", "test")}
    if not source_splits["train"] or not source_splits["val"]:
        raise FileNotFoundError(f"train/val画像ディレクトリを解決できません: {source_splits}")

    source_to_target: dict[int, int] = {}
    for source_id, source_name in enumerate(source_names):
        for target_id, target_name in enumerate(TARGET_CLASS_NAMES):
            if source_name.strip().lower() in SOURCE_ALIASES[target_name]:
                source_to_target[source_id] = target_id
    missing = set(range(len(TARGET_CLASS_NAMES))) - set(source_to_target.values())
    if missing:
        raise ValueError(f"元データに対象クラスがありません: {missing}")

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_splits: dict[str, Path] = {}
    for split, source_image_dir in source_splits.items():
        if source_image_dir is None:
            continue
        output_images, output_labels = prepared_dir / split / "images", prepared_dir / split / "labels"
        kept_images = kept_boxes = 0
        for source_image in image_files(source_image_dir):
            rows: list[str] = []
            source_label = labels_path(source_image)
            if source_label.exists():
                for line in source_label.read_text().splitlines():
                    values = line.split()
                    if values and int(values[0]) in source_to_target:
                        rows.append(" ".join([str(source_to_target[int(values[0])]), *values[1:]]))
            if not rows:
                continue
            relative_path = source_image.relative_to(source_image_dir)
            target_image = output_images / relative_path
            target_label = (output_labels / relative_path).with_suffix(".txt")
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_image.symlink_to(source_image.resolve())
            target_label.write_text("\n".join(rows) + "\n")
            kept_images += 1
            kept_boxes += len(rows)
        prepared_splits[split] = output_images
        print(f"{split}: 画像 {kept_images}枚 / 対象BBox {kept_boxes}個")

    data_yaml = prepared_dir / "data.yaml"
    config: dict[str, object] = {"train": str(prepared_splits["train"]), "val": str(prepared_splits["val"]), "names": TARGET_CLASS_NAMES}
    if "test" in prepared_splits:
        config["test"] = str(prepared_splits["test"])
    with data_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return data_yaml, prepared_splits


def read_labels(image_path: Path) -> list[tuple[int, float, float, float, float]]:
    file = labels_path(image_path)
    if not file.exists():
        return []
    return [(int(row[0]), *map(float, row[1:5])) for row in (line.split() for line in file.read_text().splitlines())]


def save_dataset_visualizations(splits: dict[str, Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, image_dir in splits.items():
        samples = random.sample(image_files(image_dir), min(6, len(image_files(image_dir))))
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis in axes.flat:
            axis.axis("off")
        for axis, image_path in zip(axes.flat, samples):
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for class_id, x, y, w, h in read_labels(image_path):
                x1, y1, x2, y2 = (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height
                draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
                draw.text((x1, max(0, y1 - 14)), TARGET_CLASS_NAMES[class_id], fill="lime")
            axis.imshow(image)
            axis.set_title(f"{split}: {image_path.name}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{split}_ground_truth_samples.png", dpi=160)
        plt.close(figure)
    counts = Counter(class_id for image_path in image_files(splits["train"]) for class_id, *_ in read_labels(image_path))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(TARGET_CLASS_NAMES, [counts[index] for index in range(len(TARGET_CLASS_NAMES))])
    axis.tick_params(axis="x", rotation=35)
    axis.set_ylabel("Bounding Box数")
    figure.tight_layout()
    figure.savefig(output_dir / "train_class_distribution.png", dpi=160)
    plt.close(figure)


def enable_wandb() -> None:
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    subprocess.run(["yolo", "settings", "wandb=True"], check=True)


def train_one(model_name: str, args: argparse.Namespace, data_yaml: Path, output_project: Path) -> dict[str, object]:
    run_name = f"{model_name}-{args.epochs}epochs"
    last_error: RuntimeError | None = None
    for batch in dict.fromkeys((args.batch, 8, 4)):
        try:
            torch.cuda.empty_cache()
            model = YOLO(f"{model_name}.pt")
            print(f"{model_name}: 保存先 {output_project / run_name}")
            model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=batch, device=0, project=str(output_project), name=run_name, plots=True, save=True, exist_ok=True)
            best_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
            if not best_path.exists():
                raise FileNotFoundError(f"best.pt が見つかりません: {best_path}")
            return {"model": model_name, "run_name": run_name, "batch": batch, "best_path": best_path}
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            torch.cuda.empty_cache()
            print(f"{model_name}: batch={batch} でGPUメモリ不足。次のbatch sizeで再試行します。")
    raise RuntimeError(f"{model_name} はbatch=4でも学習できませんでした。") from last_error


def ground_truth_boxes(image_path: Path) -> list[dict]:
    return [{"position": {"minX": x-w/2, "minY": y-h/2, "maxX": x+w/2, "maxY": y+h/2}, "class_id": class_id, "box_caption": TARGET_CLASS_NAMES[class_id]} for class_id, x, y, w, h in read_labels(image_path)]


def prediction_boxes(result) -> list[dict]:
    height, width = result.orig_shape
    boxes = []
    for (x1, y1, x2, y2), confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
        class_id = int(class_id)
        boxes.append({"position": {"minX": x1/width, "minY": y1/height, "maxX": x2/width, "maxY": y2/height}, "class_id": class_id, "box_caption": TARGET_CLASS_NAMES[class_id], "scores": {"confidence": float(confidence)}})
    return boxes


def evaluate_and_log(trained: dict[str, object], args: argparse.Namespace, data_yaml: Path, splits: dict[str, Path]) -> list[dict]:
    model_name = str(trained["model"])
    model = YOLO(str(trained["best_path"]))
    metrics_by_split: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    for split in ("val", "test"):
        if split not in splits:
            print(f"{model_name}: {split} splitがないためスキップします。")
            continue
        metrics = model.val(data=str(data_yaml), split=split, imgsz=args.imgsz, batch=int(trained["batch"]), device=0, plots=True)
        values = {"precision": metrics.box.mp, "recall": metrics.box.mr, "mAP50": metrics.box.map50, "mAP50-95": metrics.box.map}
        metrics_by_split[split] = values
        rows.append({"model": model_name, "batch": trained["batch"], "split": split, **values})

    wandb.finish()
    with wandb.init(project=WANDB_PROJECT, name=f"{trained['run_name']}-evaluation", group=str(trained["run_name"]), job_type="evaluation") as run:
        run.log({f"{split}/{metric}": value for split, values in metrics_by_split.items() for metric, value in values.items()})
        labels = dict(enumerate(TARGET_CLASS_NAMES))
        for split in ("val", "test"):
            if split not in splits:
                continue
            samples = random.sample(image_files(splits[split]), min(12, len(image_files(splits[split]))))
            table = wandb.Table(columns=["split", "image_path", "prediction"])
            for image_path in samples:
                result = model.predict(str(image_path), imgsz=args.imgsz, conf=0.25, device=0, verbose=False)[0]
                image = wandb.Image(str(image_path), boxes={"ground_truth": {"box_data": ground_truth_boxes(image_path), "class_labels": labels}, "predictions": {"box_data": prediction_boxes(result), "class_labels": labels}})
                table.add_data(split, str(image_path), image)
            run.log({f"{split}/predictions": table})
    return rows


def main() -> None:
    args = parse_args()
    require_environment()
    enable_wandb()
    work_dir = args.work_dir.resolve()
    output_project = work_dir / WANDB_PROJECT
    data_yaml, splits = prepare_dataset(download_dataset(work_dir / "raw"), work_dir / "raw", work_dir / "prepared_9_classes")
    save_dataset_visualizations(splits, work_dir / "visualizations")
    os.chdir(work_dir)  # Ultralyticsのローカル出力もdata配下にまとめる

    rows: list[dict] = []
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("少なくとも1つのモデルを指定してください。")
    for model_name in model_names:
        trained = train_one(model_name, args, data_yaml, output_project)
        print(f"学習完了: {model_name} / {trained['best_path']}")
        rows.extend(evaluate_and_log(trained, args, data_yaml, splits))

    comparison = pd.DataFrame(rows).sort_values(["split", "mAP50-95"], ascending=[True, False])
    comparison_path = work_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\n=== モデル比較 ===")
    print(comparison.to_string(index=False))
    with wandb.init(project=WANDB_PROJECT, name="model-comparison", job_type="comparison") as run:
        run.log({"model_comparison": wandb.Table(dataframe=comparison)})
        run.save(str(comparison_path))


if __name__ == "__main__":
    main()
