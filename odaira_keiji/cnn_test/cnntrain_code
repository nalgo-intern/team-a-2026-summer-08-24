# ============================================================
# 果物・野菜検出 ＋ 品質判定システム
#
# YOLO：
#   ・果物・野菜の種類を検出
#   ・Bounding Boxを取得
#
# CNN：
#   ・YOLOで検出された部分を切り出す
#   ・Fresh / Rotten を判定
#
# Google Colab用
# ============================================================

# ============================================================
# 1. 必要なライブラリをインストール
# ============================================================

!pip install -q kagglehub ultralytics

# ============================================================
# 2. ライブラリをインポート
# ============================================================

import os
import shutil
import random
from pathlib import Path

import kagglehub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 【変更点①】GPU使用時はMixed Precisionを有効化して学習を高速化
# FP16を活用し、GPU上のCNN計算を軽量化します。
from tensorflow.keras import mixed_precision

if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed Precision: ON")
else:
    print("Mixed Precision: OFF（GPUが検出されませんでした）")

from ultralytics import YOLO
from PIL import Image, ImageDraw

# ============================================================
# 3. TensorFlowのバージョン確認
# ============================================================

print("========================================")
print("TensorFlow version:", tf.__version__)
print("========================================")

# ============================================================
# 4. 基本設定
# ============================================================

# CNN入力画像サイズ
IMG_SIZE = 224

# 1回の学習で処理する画像数
BATCH_SIZE = 256

# 学習回数
EPOCHS = 20

# 学習率
LEARNING_RATE = 0.001

# Dropout率
DROPOUT_RATE = 0.5

# EarlyStopping
PATIENCE = 5

# CNNモデル保存先
CNN_MODEL_PATH = "best_freshness_model.keras"

# YOLOモデル保存先
YOLO_MODEL_PATH = "best_yolo_model.pt"

# テスト画像
TEST_IMAGE_PATH = "test.jpg"

# 結果画像
OUTPUT_IMAGE_PATH = "result.jpg"

# ============================================================
# 5. Food Freshness Datasetをダウンロード
# ============================================================

print("\nFood Freshness Datasetをダウンロードします...")

path = kagglehub.dataset_download(
    "ulnnproject/food-freshness-dataset"
)

print("Dataset Path:")
print(path)

# ============================================================
# 6. データセットのフォルダ構造を確認
# ============================================================

print("\n========================================")
print("Dataset Structure")
print("========================================")

for root, dirs, files in os.walk(path):

    level = root.replace(path, "").count(os.sep)

    indent = " " * 4 * level

    print(f"{indent}{os.path.basename(root)}/")

    subindent = " " * 4 * (level + 1)

    for file in files[:3]:
        print(f"{subindent}{file}")

# ============================================================
# ============================================================
# 7. Fresh / Rotten フォルダを指定
# ============================================================

DATASET_DIR = os.path.join(
    path,
    "Dataset"
)

FRESH_DIR = os.path.join(
    DATASET_DIR,
    "Fresh"
)

ROTTEN_DIR = os.path.join(
    DATASET_DIR,
    "Rotten"
)

print("Freshフォルダ:", FRESH_DIR)
print("Rottenフォルダ:", ROTTEN_DIR)

# ============================================================
# 8. 画像ファイルを取得する関数
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

def get_image_paths(folder):

    image_paths = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(IMAGE_EXTENSIONS):

                image_path = os.path.join(
                    root,
                    file
                )

                image_paths.append(
                    image_path
                )

    return image_paths

# ============================================================
# 9. Fresh / Rotten画像を取得
# ============================================================

fresh_images = get_image_paths(
    FRESH_DIR
)

rotten_images = get_image_paths(
    ROTTEN_DIR
)

print("\n========================================")
print("Fresh画像数 :", len(fresh_images))
print("Rotten画像数:", len(rotten_images))
print("========================================")

# ============================================================
# 10. データセット確認
# ============================================================

if len(fresh_images) == 0:

    raise ValueError(
        "Fresh画像が見つかりません。"
    )

if len(rotten_images) == 0:

    raise ValueError(
        "Rotten画像が見つかりません。"
    )

# ============================================================
# 10. データセットを
#     train / validation / test に分割
#
# train : 70%
# val   : 15%
# test  : 15%
# ============================================================

random.seed(42)

random.shuffle(fresh_images)
random.shuffle(rotten_images)

def split_dataset(image_list):

    total = len(image_list)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train = image_list[:train_end]
    val = image_list[train_end:val_end]
    test = image_list[val_end:]

    return train, val, test

fresh_train, fresh_val, fresh_test = split_dataset(
    fresh_images
)

rotten_train, rotten_val, rotten_test = split_dataset(
    rotten_images
)

# ============================================================
# 11. TensorFlow Datasetを作成
# ============================================================

def create_dataset(image_paths, labels, shuffle=True):

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def load_image(image_path, label):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_image(
            image,
            channels=3,
            expand_animations=False
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(image_paths),
                1000
            )
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

# ============================================================
# 12. train / validation / test データ作成
#
# Fresh  = 0
# Rotten = 1
# ============================================================

train_paths = fresh_train + rotten_train

train_labels = (
    [0] * len(fresh_train)
    +
    [1] * len(rotten_train)
)

val_paths = fresh_val + rotten_val

val_labels = (
    [0] * len(fresh_val)
    +
    [1] * len(rotten_val)
)

test_paths = fresh_test + rotten_test

test_labels = (
    [0] * len(fresh_test)
    +
    [1] * len(rotten_test)
)

# データを混ぜる
train_data = list(
    zip(train_paths, train_labels)
)

val_data = list(
    zip(val_paths, val_labels)
)

test_data = list(
    zip(test_paths, test_labels)
)

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

train_paths, train_labels = zip(*train_data)
val_paths, val_labels = zip(*val_data)
test_paths, test_labels = zip(*test_data)

# TensorFlow Dataset作成
train_ds = create_dataset(
    list(train_paths),
    list(train_labels),
    shuffle=True
)

val_ds = create_dataset(
    list(val_paths),
    list(val_labels),
    shuffle=False
)

test_ds = create_dataset(
    list(test_paths),
    list(test_labels),
    shuffle=False
)

print("\n========================================")
print("Dataset Split")
print("========================================")

print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))

# ============================================================
# 13. Data Augmentation
#
# 学習時に画像を少し変化させて
# 過学習を防ぐ
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),

    tf.keras.layers.RandomContrast(
        0.1
    )

])

# ============================================================
# 14. CNNモデルを作成
# ============================================================

model = tf.keras.Sequential([

    # --------------------------------------------------------
    # 入力層
    # --------------------------------------------------------

    tf.keras.layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    # --------------------------------------------------------
    # Data Augmentation
    # --------------------------------------------------------

    data_augmentation,

    # --------------------------------------------------------
    # 正規化
    #
    # 0～255
    #   ↓
    # 0～1
    # --------------------------------------------------------

    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # ========================================================
    # Conv Block 1
    # ========================================================

    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # ========================================================
    # Conv Block 2
    # ========================================================

    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # ========================================================
    # Conv Block 3
    # ========================================================

    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # ========================================================
    # Conv Block 4
    # ========================================================

    tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # ========================================================
    # 特徴量をまとめる
    # ========================================================

    tf.keras.layers.GlobalAveragePooling2D(),

    # ========================================================
    # 全結合層
    # ========================================================

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # ========================================================
    # Dropout
    # ========================================================

    tf.keras.layers.Dropout(
        DROPOUT_RATE
    ),

    # ========================================================
    # 出力層
    #
    # 0 : Fresh
    # 1 : Rotten
    # ========================================================

    # 【変更点③】Mixed Precision使用時も最終出力はFP32に固定
    # Fresh / Rottenの確率計算を安定させます。
    tf.keras.layers.Dense(
        2,
        activation="softmax",
        dtype="float32"
    )

])

# ============================================================
# 15. モデル構造を表示
# ============================================================

print("\n========================================")
print("CNN Model Summary")
print("========================================")

model.summary()

# ============================================================
# 16. モデルをコンパイル
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],
)

# ============================================================
# 17. EarlyStopping
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=PATIENCE,

    restore_best_weights=True

)

# ============================================================
# 18. CNNモデルを学習
# ============================================================

print("\n========================================")
print("CNN Training Start")
print("========================================")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stopping
    ]

)

# ============================================================
# 19. CNNモデルを保存
# ============================================================

print("\n========================================")
print("CNN Model Save")
print("========================================")

# CNNモデルを保存
model.save(CNN_MODEL_PATH)

# 【変更点】保存したファイルの絶対パスを取得
absolute_path = os.path.abspath(CNN_MODEL_PATH)

print("CNNモデルを保存しました。")
print("絶対パス:")
print(absolute_path)

# 【変更点】ファイルが実際に存在するか確認
if os.path.exists(absolute_path):

    file_size = os.path.getsize(
        absolute_path
    )

    print(
        f"ファイルサイズ: "
        f"{file_size / (1024 * 1024):.2f} MB"
    )

else:

    print("WARNING: ファイルが確認できません。")

# ============================================================
# 20. テストデータで評価
# ============================================================

print("\n========================================")
print("CNN Test Evaluation")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_ds
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ============================================================
# 21. 学習結果をグラフで表示
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("CNN Training Accuracy")

plt.legend()

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("CNN Training Loss")

plt.legend()

plt.show()

# ============================================================
# 22. 品質クラス
# ============================================================

quality_classes = [
    "Fresh",
    "Rotten"
]

# ============================================================
# 23. CNNで品質判定する関数
# ============================================================

def predict_freshness(crop_image):

    # PIL Imageを224×224にリサイズ
    resized_image = crop_image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    # RGBに変換
    resized_image = resized_image.convert(
        "RGB"
    )

    # NumPy配列に変換
    image_array = np.array(
        resized_image
    )

    # バッチ次元を追加
    #
    # (224, 224, 3)
    #        ↓
    # (1, 224, 224, 3)
    #
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # CNNで予測
    prediction = model.predict(
        image_array,
        verbose=0
    )

    # 最大確率のクラス番号
    predicted_index = np.argmax(
        prediction[0]
    )

    # クラス名
    predicted_class = quality_classes[
        predicted_index
    ]

    # 信頼度
    confidence = float(
        prediction[0][predicted_index]
    )

    return (
        predicted_class,
        confidence,
        prediction[0]
    )

# ============================================================
# 24. YOLOモデルを読み込む
#
# ここでは自分で学習したYOLOモデル
# best_yolo_model.pt を使用
# ============================================================

if not os.path.exists(YOLO_MODEL_PATH):

    print("\n========================================")
    print("YOLOモデルが見つかりません")
    print(YOLO_MODEL_PATH)
    print("========================================")

    print(
        "YOLO統合を行う場合は、"
        "学習済みモデルをアップロードしてください。"
    )

else:

    yolo_model = YOLO(
        YOLO_MODEL_PATH
    )

    print("\nYOLOモデルを読み込みました。")

    # ========================================================
    # 25. テスト画像が存在するか確認
    # ========================================================

    if not os.path.exists(TEST_IMAGE_PATH):

        print("\n========================================")
        print("テスト画像が見つかりません")
        print(TEST_IMAGE_PATH)
        print("========================================")

        print(
            "YOLO + CNNによる判定を行う場合は、"
            "test.jpg をアップロードしてください。"
        )

    else:

        # ====================================================
        # 26. テスト画像を読み込む
        # ====================================================

        image = Image.open(
            TEST_IMAGE_PATH
        ).convert(
            "RGB"
        )

        # 結果表示用
        result_image = image.copy()

        draw = ImageDraw.Draw(
            result_image
        )

        # ====================================================
        # 27. YOLOで物体検出
        # ====================================================

        print("\n========================================")
        print("YOLO Object Detection Start")
        print("========================================")

        results = yolo_model(
            TEST_IMAGE_PATH
        )

        detection_count = 0

        # ====================================================
        # 28. 検出結果を1つずつ処理
        # ====================================================

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:

                detection_count += 1

                # ============================================
                # Bounding Box取得
                # ============================================

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.width, int(x2))
                y2 = min(image.height, int(y2))

                # ============================================
                # YOLOクラス番号
                # ============================================

                class_id = int(
                    box.cls[0]
                )

                # ============================================
                # 食品名
                # ============================================

                food_name = yolo_model.names[
                    class_id
                ]

                # ============================================
                # YOLO検出信頼度
                # ============================================

                detection_confidence = float(
                    box.conf[0]
                )

                # ============================================
                # Bounding Boxで画像を切り出す
                # ============================================

                crop_image = image.crop(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                # ============================================
                # CNNで品質判定
                # ============================================

                quality_result, quality_confidence, probabilities = (
                    predict_freshness(
                        crop_image
                    )
                )

                # ============================================
                # コンソールに結果を表示
                # ============================================

                print("\n----------------------------------------")

                print(
                    "検出番号:",
                    detection_count
                )

                print(
                    "食品名:",
                    food_name
                )

                print(
                    f"YOLO信頼度: "
                    f"{detection_confidence * 100:.2f}%"
                )

                print(
                    "品質判定:",
                    quality_result
                )

                print(
                    f"CNN信頼度: "
                    f"{quality_confidence * 100:.2f}%"
                )

                print(
                    f"Fresh: "
                    f"{probabilities[0] * 100:.2f}%"
                )

                print(
                    f"Rotten: "
                    f"{probabilities[1] * 100:.2f}%"
                )

                # ============================================
                # Bounding Boxを描画
                # ============================================

                draw.rectangle(

                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    outline="red",

                    width=3

                )

                # ============================================
                # 表示ラベル
                # ============================================

                label = (
                    f"{food_name}\n"
                    f"YOLO: "
                    f"{detection_confidence * 100:.1f}%\n"
                    f"{quality_result}: "
                    f"{quality_confidence * 100:.1f}%"
                )

                # ============================================
                # テキスト表示位置
                # ============================================

                text_x = x1

                text_y = max(
                    0,
                    y1 - 60
                )

                # ============================================
                # 結果を画像に描画
                # ============================================

                draw.text(

                    (
                        text_x,
                        text_y
                    ),

                    label,

                    fill="red"

                )

        # ====================================================
        # 29. 結果を保存
        # ====================================================

        result_image.save(
            OUTPUT_IMAGE_PATH
        )

        print("\n========================================")
        print("検出された物体数:", detection_count)
        print("結果画像:", OUTPUT_IMAGE_PATH)
        print("========================================")

        # ====================================================
        # 30. 結果画像を表示
        # ====================================================

        plt.figure(
            figsize=(12, 8)
        )

        plt.imshow(
            result_image
        )

        plt.axis(
            "off"
        )

        plt.title(
            "Fruit / Vegetable Detection + Quality Classification"
        )

        plt.show()
