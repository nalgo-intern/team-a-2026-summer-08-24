"""既存のCNN評価結果とBiRefNetクロップ後のCNN結果を比較する。"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from make_confusion_matrix import create_confusion_matrix


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "odaira_keiji/cnn_evaluate"
DEFAULT_RESULT_CSV = SOURCE_DIR / "result_full.csv"
DEFAULT_CNN_MODEL = SOURCE_DIR / "model.keras"
DEFAULT_OUTPUT_DIR = Path("data/cnn_evaluate_comparison")
QUALITY_CLASSES = ("fresh", "rotten")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--cnn-model", type=Path, default=DEFAULT_CNN_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--birefnet-model", default="ZhengPeng7/BiRefNet")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--birefnet-size", type=int, default=512, help="BiRefNet入力画像の一辺。既定値は512")
    parser.add_argument("--max-images", type=int, default=0, help="評価枚数。0ならresult_full.csvの全画像")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


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
    return np.where(labels == component, 255, 0).astype(np.uint8), (int(x), int(y), int(x + width), int(y + height))


def expand_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int], padding_ratio: float = 0.12) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    image_width, image_height = image_size
    padding_x, padding_y = int(width * padding_ratio), int(height * padding_ratio)
    return (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(image_width, x2 + padding_x),
        min(image_height, y2 + padding_y),
    )


def find_source_image(source_dir: Path, filename: str) -> Path:
    """評価CSVのFilenameに、信頼度が付加された保存画像を対応付ける。"""
    original = Path(filename).stem
    candidates = [
        path for path in source_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.stem.startswith(original + "_")
    ]
    if not candidates:
        raise FileNotFoundError(f"元画像が見つかりません: {filename}")
    if len(candidates) > 1:
        exact_parent = [path for path in candidates if Path(filename).parent.name.lower() in str(path.parent).lower()]
        if exact_parent:
            candidates = exact_parent
    return candidates[0]


def predict_quality(model: tf.keras.Model, image: Image.Image) -> tuple[str, float, float, float]:
    resized = image.convert("RGB").resize((224, 224))
    array = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    prediction = np.asarray(model.predict(array, verbose=0)[0], dtype=np.float32)
    if not np.isclose(float(prediction.sum()), 1.0, atol=1e-3):
        prediction = tf.nn.softmax(prediction).numpy()
    index = int(np.argmax(prediction))
    return QUALITY_CLASSES[index], float(prediction[index]), float(prediction[0]), float(prediction[1])


def save_transition_report(rows: list[dict[str, object]], output_dir: Path) -> None:
    def as_bool(value: object) -> bool:
        return value is True or str(value).lower() == "true"

    csv_path = output_dir / "comparison_report.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        "correct_to_correct": [row for row in rows if row["transition"] == "correct_to_correct"],
        "correct_to_incorrect": [row for row in rows if row["transition"] == "correct_to_incorrect"],
        "incorrect_to_correct": [row for row in rows if row["transition"] == "incorrect_to_correct"],
        "incorrect_to_incorrect": [row for row in rows if row["transition"] == "incorrect_to_incorrect"],
    }
    report_path = output_dir / "comparison_report.txt"
    baseline_correct = sum(as_bool(row["baseline_correct"]) for row in rows)
    detected_rows = [row for row in rows if row["crop_status"] == "ok"]
    crop_correct = sum(as_bool(row["crop_correct"]) for row in rows)
    detected_crop_correct = sum(as_bool(row["crop_correct"]) for row in detected_rows)
    final_correct = sum(as_bool(row["final_correct"]) for row in rows)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("元画像判定とBiRefNetクロップ後判定の比較\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"評価画像数: {len(rows)}\n")
        file.write(f"元画像CNN正解率: {baseline_correct / len(rows) * 100:.2f}% ({baseline_correct}/{len(rows)})\n")
        file.write(f"BiRefNet検出成功: {len(detected_rows)}/{len(rows)}\n")
        file.write(f"クロップ後CNN正解率（検出成功のみ）: {detected_crop_correct / len(detected_rows) * 100:.2f}% ({detected_crop_correct}/{len(detected_rows)})\n")
        file.write(f"クロップ後正解率（検出失敗を不正解として含む）: {crop_correct / len(rows) * 100:.2f}% ({crop_correct}/{len(rows)})\n\n")
        file.write(f"修正後最終正解率（フォールバック・安全判定）: {final_correct / len(rows) * 100:.2f}% ({final_correct}/{len(rows)})\n\n")
        file.write("修正後の判定遷移\n")
        file.write(f"正解 → 正解: {len(groups['correct_to_correct'])}枚\n")
        file.write(f"正解 → 不正解: {len(groups['correct_to_incorrect'])}枚\n")
        file.write(f"不正解 → 正解: {len(groups['incorrect_to_correct'])}枚\n")
        file.write(f"不正解 → 不正解: {len(groups['incorrect_to_incorrect'])}枚\n\n")
        file.write("改善した画像（不正解 → 正解）\n")
        for row in groups["incorrect_to_correct"]:
            file.write(f"- {row['filename']}\n")
        file.write("\n悪化した画像（正解 → 不正解）\n")
        for row in groups["correct_to_incorrect"]:
            file.write(f"- {row['filename']}\n")

    for group_name, group_rows in groups.items():
        group_dir = output_dir / group_name
        original_dir = group_dir / "original"
        cropped_dir = group_dir / "cropped"
        original_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)
        for row in group_rows:
            source = Path(str(row["source_image"]))
            output_name = f"{int(row['number']):04d}_{source.name}"
            shutil.copy2(source, original_dir / output_name)
            crop_source = row.get("crop_image")
            if crop_source:
                shutil.copy2(Path(str(crop_source)), cropped_dir / output_name)

    print(f"比較CSV: {csv_path}")
    print(f"比較レポート: {report_path}")
    print(f"元画像CNN正解率: {baseline_correct / len(rows) * 100:.2f}%")
    print(f"BiRefNet検出成功: {len(detected_rows)}/{len(rows)}")
    print(f"クロップ後CNN正解率（検出成功のみ）: {detected_crop_correct / len(detected_rows) * 100:.2f}%")
    print(f"クロップ後正解率（検出失敗を不正解として含む）: {crop_correct / len(rows) * 100:.2f}%")
    print(f"修正後最終正解率（フォールバック・安全判定）: {final_correct / len(rows) * 100:.2f}%")
    for group_name, group_rows in groups.items():
        print(f"{group_name}: {len(group_rows)}枚")


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません。--device cpuを指定してください。")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--thresholdは0より大きく1未満にしてください。")
    if args.birefnet_size < 64:
        raise ValueError("--birefnet-sizeには64以上を指定してください。")

    output_dir = args.output_dir.resolve()
    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = output_dir / "cropped_images"
    crop_dir.mkdir(parents=True, exist_ok=True)

    with args.result_csv.open(encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))
    if args.max_images > 0:
        source_rows = source_rows[:args.max_images]

    print(f"CNNモデル: {args.cnn_model}")
    cnn_model = tf.keras.models.load_model(args.cnn_model, compile=False)
    print(f"BiRefNetモデル: {args.birefnet_model}")
    birefnet = AutoModelForImageSegmentation.from_pretrained(args.birefnet_model, trust_remote_code=True)
    birefnet.to(args.device).eval()
    if args.device == "cpu":
        birefnet.float()
    model_dtype = next(birefnet.parameters()).dtype
    transform = transforms.Compose([
        transforms.Resize((args.birefnet_size, args.birefnet_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows: list[dict[str, object]] = []
    for index, source_row in enumerate(source_rows, start=1):
        source_image = find_source_image(args.source_dir, source_row["Filename"])
        image = Image.open(source_image).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device=args.device, dtype=model_dtype)
        with torch.inference_mode():
            output = prediction_tensor(birefnet(tensor))
            probability = output.sigmoid()[0, 0].float().cpu().numpy()
        probability = cv2.resize(probability, image.size, interpolation=cv2.INTER_LINEAR)
        binary = np.where(probability >= args.threshold, 255, 0).astype(np.uint8)
        mask, bbox = largest_component(binary)

        expected = source_row["True Class"].lower()
        baseline_predicted = source_row["Predicted Class"].lower()
        baseline_correct = source_row["Result"].lower() == "correct"
        crop_predicted = ""
        crop_confidence = ""
        crop_correct = False
        status = "ok"
        if bbox is None:
            status = "foreground_not_detected"
            final_predicted = baseline_predicted
            final_correct = baseline_correct
            crop_image_path = ""
        else:
            padded_bbox = expand_bbox(bbox, image.size)
            x1, y1, x2, y2 = padded_bbox
            crop = image.crop(padded_bbox)
            crop_image_path = crop_dir / f"{index:04d}_{source_image.name}"
            crop.save(crop_image_path, quality=95)
            crop_predicted, confidence, fresh_probability, rotten_probability = predict_quality(cnn_model, crop)
            crop_confidence = f"{confidence:.6f}"
            crop_correct = crop_predicted == expected
            final_predicted = "rotten" if "rotten" in (baseline_predicted, crop_predicted) else "fresh"
            final_correct = final_predicted == expected

        if baseline_correct and final_correct:
            transition = "correct_to_correct"
        elif baseline_correct and not final_correct:
            transition = "correct_to_incorrect"
        elif not baseline_correct and final_correct:
            transition = "incorrect_to_correct"
        else:
            transition = "incorrect_to_incorrect"

        rows.append({
            "number": index,
            "filename": source_row["Filename"],
            "source_image": str(source_image),
            "crop_image": str(crop_image_path),
            "expected": expected,
            "baseline_predicted": baseline_predicted,
            "baseline_correct": baseline_correct,
            "crop_predicted": crop_predicted,
            "crop_confidence": crop_confidence,
            "crop_correct": crop_correct,
            "final_predicted": final_predicted,
            "final_correct": final_correct,
            "crop_status": status,
            "transition": transition,
        })
        print(f"[{index}/{len(source_rows)}] {source_row['Filename']}: {transition}")

    save_transition_report(rows, output_dir)
    comparison_csv = output_dir / "comparison_report.csv"
    create_confusion_matrix(comparison_csv, output_dir / "crop_confusion_matrix")


if __name__ == "__main__":
    main()
