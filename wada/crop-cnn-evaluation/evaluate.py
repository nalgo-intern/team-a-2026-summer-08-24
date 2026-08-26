"""Kaggleデータをダウンロードし、BiRefNetでクロップ後にCNNで鮮度評価する。"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


KAGGLE_DATASET = "ulnnproject/food-freshness-dataset"
DEFAULT_CNN_MODEL = Path(__file__).resolve().parents[2] / "odaira_keiji/cnn_test/model.keras"
DEFAULT_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
QUALITY_CLASSES = ("fresh", "rotten")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/food_freshness_evaluation"))
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--birefnet-model", default=DEFAULT_BIREFNET_MODEL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-images", type=int, default=200, help="テストデータからの評価枚数。既定値は200枚、0なら全テストデータ")
    parser.add_argument("--seed", type=int, default=42, help="学習時と同じデータ分割シード")
    parser.add_argument("--max-per-group", type=int, default=200, help="正解/不正解フォルダーへ保存する最大枚数")
    parser.add_argument("--clean-output", action="store_true", help="出力ディレクトリを削除してから実行")
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def training_code_image_paths(folder: Path) -> list[Path]:
    """cnntrain_code.pyのget_image_pathsと同じ順序で画像を取得する。"""
    paths: list[Path] = []
    for root, _dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_SUFFIXES:
                paths.append(Path(root) / file)
    return paths


def find_class_directory(dataset_dir: Path, class_name: str) -> Path:
    """Dataset/Freshのような、展開後のクラスディレクトリを探す。"""
    expected = class_name.casefold()
    candidates = [
        path for path in dataset_dir.rglob("*")
        if path.is_dir() and path.name.casefold() == expected
    ]
    if not candidates:
        raise FileNotFoundError(f"{class_name}フォルダーが見つかりません: {dataset_dir}")
    return candidates[0]


def make_test_split(dataset_dir: Path, seed: int) -> list[tuple[Path, str]]:
    """学習コードのseedと70/15/15分割を再現してテストデータを作る。"""
    fresh_images = training_code_image_paths(find_class_directory(dataset_dir, "Fresh"))
    rotten_images = training_code_image_paths(find_class_directory(dataset_dir, "Rotten"))
    if not fresh_images or not rotten_images:
        raise FileNotFoundError(
            "Fresh/Rottenフォルダーが見つかりません。"
            f" dataset_dir={dataset_dir}"
        )

    random.seed(seed)
    random.shuffle(fresh_images)
    random.shuffle(rotten_images)

    def split(paths: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
        train_end = int(len(paths) * 0.70)
        val_end = int(len(paths) * 0.85)
        return paths[:train_end], paths[train_end:val_end], paths[val_end:]

    _fresh_train, _fresh_val, fresh_test = split(fresh_images)
    _rotten_train, _rotten_val, rotten_test = split(rotten_images)
    test_data = [(path, "fresh") for path in fresh_test] + [(path, "rotten") for path in rotten_test]
    random.shuffle(test_data)
    return test_data


def download_dataset(dataset_dir: Path) -> None:
    """Kaggleからデータセットを新規フォルダーへダウンロードする。"""
    if image_files(dataset_dir):
        print(f"既存データを使用: {dataset_dir}")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    api = KaggleApi()
    api.authenticate()
    print(f"Kaggleからダウンロード: {KAGGLE_DATASET}")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(dataset_dir),
        unzip=True,
        quiet=False,
    )


def label_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "rotten" in name or "rotten" in str(path.parent).lower():
        return "rotten"
    if "fresh" in name or "fresh" in str(path.parent).lower():
        return "fresh"
    return ""


def prediction_tensor(output: object) -> torch.Tensor:
    while isinstance(output, (list, tuple)):
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"BiRefNetの出力形式に対応していません: {type(output)}")
    return output


def largest_component(mask: np.ndarray, min_area_ratio: float = 0.005) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), None

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    if int(area) < max(1, int(mask.shape[0] * mask.shape[1] * min_area_ratio)):
        return np.zeros_like(mask), None

    result = np.where(labels == component, 255, 0).astype(np.uint8)
    return result, (int(x), int(y), int(x + width), int(y + height))


def predict_quality(model: tf.keras.Model, image: Image.Image) -> tuple[str, float, float, float]:
    resized = image.convert("RGB").resize((224, 224))
    array = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    prediction = np.asarray(model.predict(array, verbose=0)[0], dtype=np.float32)
    if not np.isclose(float(prediction.sum()), 1.0, atol=1e-3):
        prediction = tf.nn.softmax(prediction).numpy()
    class_index = int(np.argmax(prediction))
    return QUALITY_CLASSES[class_index], float(prediction[class_index]), float(prediction[0]), float(prediction[1])


def evaluate(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。--device cpu を指定してください。")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--thresholdは0より大きく1未満にしてください。")

    data_dir = args.data_dir.resolve()
    dataset_dir = data_dir / "dataset"
    crop_dir = data_dir / "cnn_crops"
    foreground_dir = data_dir / "birefnet_foregrounds"
    result_dir = data_dir / "result_images"
    correct_dir = data_dir / "correct_images"
    incorrect_dir = data_dir / "incorrect_images"
    csv_path = data_dir / "cnn_evaluation_results.csv"

    if args.max_per_group < 1:
        raise ValueError("--max-per-groupには1以上を指定してください。")

    if args.clean_output:
        for directory in (crop_dir, foreground_dir, result_dir, correct_dir, incorrect_dir):
            if directory.exists():
                shutil.rmtree(directory)
    for directory in (crop_dir, foreground_dir, result_dir, correct_dir, incorrect_dir):
        directory.mkdir(parents=True, exist_ok=True)

    download_dataset(dataset_dir)
    candidates = make_test_split(data_dir / "dataset", args.seed)
    if args.num_images > 0:
        candidates = random.Random(args.seed).sample(candidates, min(args.num_images, len(candidates)))

    print(f"CNNモデルを読み込み: {args.cnn_model}")
    cnn_model = tf.keras.models.load_model(args.cnn_model, compile=False)
    print(f"BiRefNetを読み込み: {args.birefnet_model}")
    birefnet = AutoModelForImageSegmentation.from_pretrained(args.birefnet_model, trust_remote_code=True)
    birefnet.to(args.device).eval()
    if args.device == "cpu":
        birefnet.float()
    model_dtype = next(birefnet.parameters()).dtype
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows: list[dict[str, object]] = []
    correct_count = 0
    incorrect_count = 0
    for index, (image_path, expected) in enumerate(candidates, start=1):
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device=args.device, dtype=model_dtype)
        with torch.inference_mode():
            output = prediction_tensor(birefnet(tensor))
            probability = output.sigmoid()[0, 0].float().cpu().numpy()

        probability = cv2.resize(probability, image.size, interpolation=cv2.INTER_LINEAR)
        binary = np.where(probability >= args.threshold, 255, 0).astype(np.uint8)
        mask, bbox = largest_component(binary)
        stem = image_path.stem
        row: dict[str, object] = {
            "image": str(image_path.relative_to(data_dir / "dataset")),
            "detected": bbox is not None,
            "bbox": "" if bbox is None else ",".join(map(str, bbox)),
            "expected": expected,
            "predicted": "",
            "confidence": "",
            "fresh_probability": "",
            "rotten_probability": "",
            "correct": "",
            "error": "",
        }
        if bbox is None:
            row["error"] = "foreground_not_detected"
        else:
            x1, y1, x2, y2 = bbox
            crop = image.crop((x1, y1, x2, y2))
            predicted, confidence, fresh_probability, rotten_probability = predict_quality(cnn_model, crop)
            row.update({
                "predicted": predicted,
                "confidence": f"{confidence:.6f}",
                "fresh_probability": f"{fresh_probability:.6f}",
                "rotten_probability": f"{rotten_probability:.6f}",
                "correct": expected != "" and expected == predicted,
            })
            crop.save(crop_dir / f"{stem}.jpg", quality=95)

            foreground = image.convert("RGBA")
            foreground.putalpha(Image.fromarray(mask))
            foreground.save(foreground_dir / f"{stem}.png")

            result_image = image.copy()
            draw = ImageDraw.Draw(result_image)
            draw.rectangle(bbox, outline="red", width=max(3, min(image.size) // 100))
            draw.text((x1, max(0, y1 - 20)), f"{predicted} {confidence * 100:.1f}%", fill="red")
            result_image.save(result_dir / f"{stem}.jpg", quality=95)

        # 正解・不正解を最大200枚ずつ保存する。不検出は不正解側へ入れる。
        is_correct = row["correct"] is True
        group_dir = correct_dir if is_correct else incorrect_dir
        group_count = correct_count if is_correct else incorrect_count
        if group_count < args.max_per_group:
            group_name = "correct" if is_correct else "incorrect"
            output_name = f"{group_name}_{index:04d}_{expected}_{stem}.jpg"
            if bbox is None:
                image.save(group_dir / output_name, quality=95)
            else:
                result_image.save(group_dir / output_name, quality=95)
            if is_correct:
                correct_count += 1
            else:
                incorrect_count += 1

        rows.append(row)
        print(f"[{index}/{len(candidates)}] {image_path.name}: {row['predicted'] or row['error']}")

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labeled = [row for row in rows if row["expected"] in QUALITY_CLASSES and isinstance(row["correct"], bool)]
    correct = sum(bool(row["correct"]) for row in labeled)
    print(f"\n結果CSV: {csv_path}")
    print(f"評価画像数: {len(rows)}")
    print(f"クロップ成功数: {sum(bool(row['detected']) for row in rows)}")
    print(f"正解画像フォルダー: {correct_dir} ({correct_count}枚)")
    print(f"不正解画像フォルダー: {incorrect_dir} ({incorrect_count}枚)")
    if labeled:
        print(f"正解ラベル付き評価数: {len(labeled)}")
        print(f"CNN正解率: {correct / len(labeled) * 100:.2f}% ({correct}/{len(labeled)})")
    else:
        print("ファイル名から正解ラベルを取得できる画像がないため、正解率は計算していません。")


if __name__ == "__main__":
    evaluate(parse_args())
