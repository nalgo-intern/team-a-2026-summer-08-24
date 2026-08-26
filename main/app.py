"""食品画像の前景抽出と鮮度判定を行うFlaskアプリ。"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, ImageDraw
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


ROOT_DIR = Path(__file__).resolve().parents[1]
CNN_MODEL_PATH = Path(
    os.getenv(
        "CNN_MODEL_PATH",
        Path(__file__).resolve().parent / "model" / "best_freshness_model_wada_best.keras",
    )
)
BIREFNET_MODEL_NAME = os.getenv("BIREFNET_MODEL_NAME", "ZhengPeng7/BiRefNet")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (224, 224)
QUALITY_CLASSES = ("Fresh", "Rotten")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

_cnn_model: tf.keras.Model | None = None
_birefnet_model: torch.nn.Module | None = None
_birefnet_transform: transforms.Compose | None = None


def load_models() -> None:
    """モデルをアプリ起動後の最初の推論時に一度だけ読み込む。"""
    global _cnn_model, _birefnet_model, _birefnet_transform

    if _cnn_model is None:
        if not CNN_MODEL_PATH.exists():
            raise FileNotFoundError(f"CNNモデルが見つかりません: {CNN_MODEL_PATH}")
        _cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)

    if _birefnet_model is None:
        if DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDAが利用できません。DEVICE=cpuで起動してください。")
        _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_MODEL_NAME,
            trust_remote_code=True,
        )
        _birefnet_model.to(DEVICE).eval()
        if DEVICE == "cpu":
            _birefnet_model.float()
        _birefnet_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def prediction_tensor(output: object) -> torch.Tensor:
    while isinstance(output, (list, tuple)):
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"BiRefNetの出力形式に対応していません: {type(output)}")
    return output


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), None

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    minimum_area = max(1, int(mask.shape[0] * mask.shape[1] * 0.005))
    if int(area) < minimum_area:
        return np.zeros_like(mask), None

    component_mask = np.where(labels == component, 255, 0).astype(np.uint8)
    return component_mask, (int(x), int(y), int(x + width), int(y + height))


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


def make_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg" if image_format == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def run_birefnet(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    assert _birefnet_model is not None
    assert _birefnet_transform is not None

    model_dtype = next(_birefnet_model.parameters()).dtype
    tensor = _birefnet_transform(image).unsqueeze(0).to(device=DEVICE, dtype=model_dtype)
    with torch.inference_mode():
        output = prediction_tensor(_birefnet_model(tensor))
        probability = output.sigmoid()[0, 0].float().cpu().numpy()

    probability = cv2.resize(probability, image.size, interpolation=cv2.INTER_LINEAR)
    binary = np.where(probability >= 0.5, 255, 0).astype(np.uint8)
    mask, bbox = largest_component(binary)

    foreground = image.convert("RGBA")
    foreground.putalpha(Image.fromarray(mask))
    return foreground, bbox


def run_cnn(crop: Image.Image) -> tuple[str, float, dict[str, float]]:
    assert _cnn_model is not None
    resized = crop.convert("RGB").resize(IMAGE_SIZE)
    pixels = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    prediction = np.asarray(_cnn_model.predict(pixels, verbose=0)[0], dtype=np.float32)

    # 学習済みモデルの出力が確率でない場合にも表示できるようにする。
    if not np.isclose(float(prediction.sum()), 1.0, atol=1e-3):
        prediction = tf.nn.softmax(prediction).numpy()
    class_index = int(np.argmax(prediction))
    probabilities = {
        name: float(probability)
        for name, probability in zip(QUALITY_CLASSES, prediction)
    }
    return QUALITY_CLASSES[class_index], float(prediction[class_index]), probabilities


def analyze(image: Image.Image) -> dict[str, object]:
    load_models()
    image = image.convert("RGB")
    foreground, bbox = run_birefnet(image)
    original_quality, original_confidence, original_probabilities = run_cnn(image)
    if bbox is None:
        quality = original_quality
        confidence = original_confidence
        probabilities = original_probabilities
        crop = image
        crop_quality = None
        warning = "食品を切り出せなかったため、元画像で判定しました。"
        display_bbox = None
    else:
        display_bbox = expand_bbox(bbox, image.size)
        crop = image.crop(display_bbox)
        crop_quality, crop_confidence, crop_probabilities = run_cnn(crop)
        quality = "Rotten" if "Rotten" in (original_quality, crop_quality) else "Fresh"
        combined_probabilities = {
            name: max(original_probabilities[name], crop_probabilities[name])
            for name in QUALITY_CLASSES
        }
        probability_total = sum(combined_probabilities.values())
        probabilities = {
            name: value / probability_total
            for name, value in combined_probabilities.items()
        }
        confidence = probabilities[quality]
        warning = None if original_quality == crop_quality else (
            f"元画像判定（{original_quality}）と切り出し判定（{crop_quality}）が異なるため、安全側に判定しました。"
        )

    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    if display_bbox is not None:
        draw.rectangle(display_bbox, outline=(220, 38, 38), width=max(3, min(image.size) // 100))

    return {
        "original_url": make_data_url(image),
        "foreground_url": make_data_url(foreground),
        "boxed_url": make_data_url(boxed),
        "crop_url": make_data_url(crop),
        "quality": quality,
        "confidence": confidence,
        "probabilities": probabilities,
        "bbox": display_bbox,
        "original_quality": original_quality,
        "crop_quality": crop_quality,
        "warning": warning,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    uploaded = next(
        (file for file in request.files.getlist("image") if file.filename),
        None,
    )
    if uploaded is None or not uploaded.filename:
        return render_template("index.html", error="画像を選択または撮影してください。"), 400

    extension = Path(uploaded.filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="JPEG、PNG、WEBP画像を使用してください。"), 400

    try:
        image = Image.open(uploaded.stream)
        result = analyze(image)
    except (OSError, ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        app.logger.exception("推論に失敗しました")
        return render_template("index.html", error=str(error)), 500

    return render_template("index.html", result=result)


@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", error="画像サイズは10MB以下にしてください。"), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
