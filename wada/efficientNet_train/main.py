# ============================================================
# 果物・野菜の品質判定システム
#
# CNN：
#   ・画像全体を入力
#   ・Fresh / Rotten を判定
#
# 改善版
#
# ・EfficientNetB0 転移学習
# ・Data Augmentation
# ・Mixed Precision
# ・ReduceLROnPlateau
# ・EarlyStopping
# ・ModelCheckpoint
# ・Fine-tuning
#
# Google Colab用
# ============================================================


# ============================================================
# 1. 必要なライブラリをインストール
# ============================================================

# !pip install -q kagglehub


# ============================================================
# 2. ライブラリをインポート
# ============================================================

import os
import random

import kagglehub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from tensorflow.keras import mixed_precision


# ============================================================
# 3. Mixed Precision設定
#
# GPU使用時はMixed Precisionを有効化
# ============================================================

if tf.config.list_physical_devices("GPU"):

    mixed_precision.set_global_policy(
        "mixed_float16"
    )

    print("Mixed Precision: ON")

else:

    print(
        "Mixed Precision: OFF（GPUが検出されませんでした）"
    )


# ============================================================
# 4. TensorFlowのバージョン確認
# ============================================================

print("========================================")

print(
    "TensorFlow version:",
    tf.__version__
)

print("========================================")


# ============================================================
# 5. 基本設定
# ============================================================

# CNN入力画像サイズ
IMG_SIZE = 224


# ------------------------------------------------------------
# BATCH_SIZEは変更しない
# ------------------------------------------------------------

BATCH_SIZE = 256


# ------------------------------------------------------------
# 学習回数
# ------------------------------------------------------------

EPOCHS = 5


# ------------------------------------------------------------
# Fine-tuning時の学習回数
# ------------------------------------------------------------

FINE_TUNE_EPOCHS = 20


# ------------------------------------------------------------
# 学習率は変更しない
# ------------------------------------------------------------

LEARNING_RATE = 0.001


# Fine-tuning用の学習率
FINE_TUNE_LEARNING_RATE = 1e-5


# Dropout率
DROPOUT_RATE = 0.3


# EarlyStopping
PATIENCE = 7


# CNNモデル保存先
CNN_MODEL_PATH = "best_freshness_model.keras"


# テスト画像
TEST_IMAGE_PATH = "test.jpg"


# ============================================================
# 6. 再現性のための乱数固定
# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# 7. Food Freshness Datasetをダウンロード
# ============================================================

print(
    "\nFood Freshness Datasetをダウンロードします..."
)

path = kagglehub.dataset_download(
    "ulnnproject/food-freshness-dataset"
)

print(
    "Dataset Path:"
)

print(path)


# ============================================================
# 8. データセットのフォルダ構造を確認
# ============================================================

print("\n========================================")

print(
    "Dataset Structure"
)

print("========================================")


for root, dirs, files in os.walk(path):

    level = root.replace(
        path,
        ""
    ).count(
        os.sep
    )

    indent = " " * 4 * level

    print(
        f"{indent}{os.path.basename(root)}/"
    )

    subindent = " " * 4 * (
        level + 1
    )

    for file in files[:3]:

        print(
            f"{subindent}{file}"
        )


# ============================================================
# 9. Fresh / Rotten フォルダを指定
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


print(
    "\nFreshフォルダ:",
    FRESH_DIR
)

print(
    "Rottenフォルダ:",
    ROTTEN_DIR
)


# ============================================================
# 10. 画像ファイルを取得する関数
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

            if file.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                image_path = os.path.join(
                    root,
                    file
                )

                image_paths.append(
                    image_path
                )

    return image_paths


# ============================================================
# 11. Fresh / Rotten画像を取得
# ============================================================

fresh_images = get_image_paths(
    FRESH_DIR
)


rotten_images = get_image_paths(
    ROTTEN_DIR
)


print("\n========================================")

print(
    "Fresh画像数 :",
    len(fresh_images)
)

print(
    "Rotten画像数:",
    len(rotten_images)
)

print("========================================")


# ============================================================
# 12. データセット確認
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
# 13. データセットを
# train / validation / test に分割
#
# train : 70%
# val   : 15%
# test  : 15%
# ============================================================

random.shuffle(
    fresh_images
)

random.shuffle(
    rotten_images
)


def split_dataset(image_list):

    total = len(
        image_list
    )

    train_end = int(
        total * 0.70
    )

    val_end = int(
        total * 0.85
    )

    train = image_list[
        :train_end
    ]

    val = image_list[
        train_end:val_end
    ]

    test = image_list[
        val_end:
    ]

    return (
        train,
        val,
        test
    )


fresh_train, fresh_val, fresh_test = (
    split_dataset(
        fresh_images
    )
)


rotten_train, rotten_val, rotten_test = (
    split_dataset(
        rotten_images
    )
)


# ============================================================
# 14. Train / Validation / Test データ作成
#
# Fresh  = 0
# Rotten = 1
# ============================================================


# Train

train_paths = (

    fresh_train

    +

    rotten_train

)


train_labels = (

    [0] * len(
        fresh_train
    )

    +

    [1] * len(
        rotten_train
    )

)


# Validation

val_paths = (

    fresh_val

    +

    rotten_val

)


val_labels = (

    [0] * len(
        fresh_val
    )

    +

    [1] * len(
        rotten_val
    )

)


# Test

test_paths = (

    fresh_test

    +

    rotten_test

)


test_labels = (

    [0] * len(
        fresh_test
    )

    +

    [1] * len(
        rotten_test
    )

)


# ============================================================
# 15. データを混ぜる
# ============================================================

train_data = list(

    zip(
        train_paths,
        train_labels
    )

)


val_data = list(

    zip(
        val_paths,
        val_labels
    )

)


test_data = list(

    zip(
        test_paths,
        test_labels
    )

)


random.shuffle(
    train_data
)

random.shuffle(
    val_data
)

random.shuffle(
    test_data
)


train_paths, train_labels = zip(
    *train_data
)

val_paths, val_labels = zip(
    *val_data
)

test_paths, test_labels = zip(
    *test_data
)


# ============================================================
# 16. TensorFlow Datasetを作成
# ============================================================

def create_dataset(
    image_paths,
    labels,
    shuffle=True
):

    dataset = tf.data.Dataset.from_tensor_slices(

        (
            image_paths,
            labels
        )

    )

    def load_image(
        image_path,
        label
    ):

        image = tf.io.read_file(
            image_path
        )

        image = tf.image.decode_image(

            image,

            channels=3,

            expand_animations=False

        )

        image.set_shape(
            [
                None,
                None,
                3
            ]
        )

        image = tf.image.resize(

            image,

            (
                IMG_SIZE,
                IMG_SIZE
            ),

            antialias=True

        )

        image = tf.cast(
            image,
            tf.float32
        )

        return (
            image,
            label
        )

    dataset = dataset.map(

        load_image,

        num_parallel_calls=tf.data.AUTOTUNE

    )

    if shuffle:

        dataset = dataset.shuffle(

            buffer_size=min(

                len(image_paths),

                5000

            ),

            seed=SEED,

            reshuffle_each_iteration=True

        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 17. TensorFlow Dataset作成
# ============================================================

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

print(
    "Dataset Split"
)

print("========================================")


print(
    "Train:",
    len(train_paths)
)

print(
    "Validation:",
    len(val_paths)
)

print(
    "Test:",
    len(test_paths)
)


# ============================================================
# 18. Data Augmentation
#
# 学習時のみ適用される
# ============================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.15
    ),

    tf.keras.layers.RandomZoom(
        height_factor=0.15,
        width_factor=0.15
    ),

    tf.keras.layers.RandomContrast(
        0.15
    ),

    tf.keras.layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    )

])


# ============================================================
# 19. EfficientNetB0を読み込む
#
# ImageNetで事前学習済みのモデル
# ============================================================

base_model = tf.keras.applications.EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(

        IMG_SIZE,

        IMG_SIZE,

        3

    )

)


# ============================================================
# 第1段階
#
# EfficientNet本体は固定
# ============================================================

base_model.trainable = False


# ============================================================
# 20. 改良版CNNモデル
# ============================================================

inputs = tf.keras.layers.Input(

    shape=(

        IMG_SIZE,

        IMG_SIZE,

        3

    )

)


# ------------------------------------------------------------
# Data Augmentation
# ------------------------------------------------------------

x = data_augmentation(
    inputs
)


# ------------------------------------------------------------
# EfficientNetB0
#
# EfficientNetB0には内部にRescaling処理があるため
# 追加の 1/255 正規化は行わない
# ------------------------------------------------------------

x = base_model(

    x,

    training=False

)


# ------------------------------------------------------------
# 特徴量を1次元に変換
# ------------------------------------------------------------

x = tf.keras.layers.GlobalAveragePooling2D()(
    x
)


# ------------------------------------------------------------
# Batch Normalization
# ------------------------------------------------------------

x = tf.keras.layers.BatchNormalization()(
    x
)


# ------------------------------------------------------------
# 全結合層
# ------------------------------------------------------------

x = tf.keras.layers.Dense(

    256,

    activation="relu"

)(
    x
)


# ------------------------------------------------------------
# Dropout
# ------------------------------------------------------------

x = tf.keras.layers.Dropout(

    DROPOUT_RATE

)(
    x
)


# ------------------------------------------------------------
# 出力層
#
# Mixed Precisionでも出力はfloat32
# ------------------------------------------------------------

outputs = tf.keras.layers.Dense(

    2,

    activation="softmax",

    dtype="float32"

)(
    x
)


model = tf.keras.Model(

    inputs,

    outputs

)


# ============================================================
# 21. モデル構造を表示
# ============================================================

print("\n========================================")

print(
    "CNN Model Summary"
)

print("========================================")


model.summary()


# ============================================================
# 22. モデルをコンパイル
#
# 学習率は指定通り 0.001
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=LEARNING_RATE

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)


# ============================================================
# 23. Callbacks
# ============================================================


# ------------------------------------------------------------
# EarlyStopping
# ------------------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=PATIENCE,

    restore_best_weights=True,

    verbose=1

)


# ------------------------------------------------------------
# Learning Rateを自動調整
#
# 初期値は0.001のまま
# Validation Lossが改善しない場合に減少
# ------------------------------------------------------------

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1

)


# ------------------------------------------------------------
# 最も良いモデルを保存
# ------------------------------------------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    CNN_MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


# ============================================================
# 24. 第1段階の学習
#
# EfficientNetB0は固定
# 分類部分のみ学習
# ============================================================

print("\n========================================")

print(
    "Stage 1 Training Start"
)

print("EfficientNetB0: Frozen")

print("Batch Size:",
      BATCH_SIZE)

print("Learning Rate:",
      LEARNING_RATE)

print("========================================")


history_stage1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[

        early_stopping,

        reduce_lr,

        checkpoint

    ]

)


# ============================================================
# 25. 第2段階：Fine-tuning
#
# EfficientNetB0の後半だけ再学習
# ============================================================

print("\n========================================")

print(
    "Fine-tuning Start"
)

print("========================================")


base_model.trainable = True


# ------------------------------------------------------------
# 前半の層を固定
#
# 後半の20層だけ学習
# ------------------------------------------------------------

for layer in base_model.layers[:-20]:

    layer.trainable = False


# ------------------------------------------------------------
# BatchNormalization層は固定
#
# Fine-tuning時の学習安定化
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(

        layer,

        tf.keras.layers.BatchNormalization

    ):

        layer.trainable = False


# ============================================================
# Fine-tuning用に再コンパイル
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=FINE_TUNE_LEARNING_RATE

    ),

    loss="sparse_categorical_crossentropy",

    metrics=[

        "accuracy"

    ]

)


# ============================================================
# Fine-tuning Callback
# ============================================================

fine_tune_early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1

)


fine_tune_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1

)


fine_tune_checkpoint = tf.keras.callbacks.ModelCheckpoint(

    CNN_MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


# ============================================================
# Fine-tuning実行
# ============================================================

history_stage2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[

        fine_tune_early_stopping,

        fine_tune_reduce_lr,

        fine_tune_checkpoint

    ]

)


# ============================================================
# 26. 最良モデルを読み込む
# ============================================================

print("\n========================================")

print(
    "Best Model Loading"
)

print("========================================")


model = tf.keras.models.load_model(
    CNN_MODEL_PATH
)


# ============================================================
# 27. テストデータで評価
# ============================================================

print("\n========================================")

print(
    "CNN Test Evaluation"
)

print("========================================")


test_loss, test_accuracy = model.evaluate(

    test_ds,

    verbose=1

)


print(
    "Test Loss:",
    test_loss
)


print(
    "Test Accuracy:",
    test_accuracy
)


# ============================================================
# 28. 学習結果をグラフで表示
# ============================================================

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

train_accuracy = (

    history_stage1.history["accuracy"]

    +

    history_stage2.history["accuracy"]

)


val_accuracy = (

    history_stage1.history["val_accuracy"]

    +

    history_stage2.history["val_accuracy"]

)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    train_accuracy,

    label="Train Accuracy"

)


plt.plot(

    val_accuracy,

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "Training Accuracy"
)


plt.legend()


plt.grid()


plt.show()


# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

train_loss = (

    history_stage1.history["loss"]

    +

    history_stage2.history["loss"]

)


val_loss = (

    history_stage1.history["val_loss"]

    +

    history_stage2.history["val_loss"]

)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    train_loss,

    label="Train Loss"

)


plt.plot(

    val_loss,

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "Training Loss"
)


plt.legend()


plt.grid()


plt.show()


# ============================================================
# 29. 品質クラス
# ============================================================

quality_classes = [

    "Fresh",

    "Rotten"

]


# ============================================================
# 30. CNNで品質判定する関数
# ============================================================

def predict_freshness(image):

    # --------------------------------------------------------
    # 224 × 224 にリサイズ
    # --------------------------------------------------------

    resized_image = image.resize(

        (

            IMG_SIZE,

            IMG_SIZE

        )

    )

    # --------------------------------------------------------
    # RGBに変換
    # --------------------------------------------------------

    resized_image = resized_image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # NumPy配列に変換
    # --------------------------------------------------------

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # バッチ次元を追加
    #
    # (224, 224, 3)
    #        ↓
    # (1, 224, 224, 3)
    # --------------------------------------------------------

    image_array = np.expand_dims(

        image_array,

        axis=0

    )

    # --------------------------------------------------------
    # CNNで予測
    # --------------------------------------------------------

    prediction = model.predict(

        image_array,

        verbose=0

    )

    # --------------------------------------------------------
    # 最大確率のクラス番号
    # --------------------------------------------------------

    predicted_index = np.argmax(
        prediction[0]
    )

    # --------------------------------------------------------
    # クラス名
    # --------------------------------------------------------

    predicted_class = quality_classes[
        predicted_index
    ]

    # --------------------------------------------------------
    # 信頼度
    # --------------------------------------------------------

    confidence = float(

        prediction[0][
            predicted_index
        ]

    )

    return (

        predicted_class,

        confidence,

        prediction[0]

    )


# ============================================================
# 31. テスト画像の品質判定
# ============================================================

print("\n========================================")

print(
    "Freshness Prediction"
)

print("========================================")


if not os.path.exists(
    TEST_IMAGE_PATH
):

    print(
        "テスト画像が見つかりません。"
    )

    print(
        TEST_IMAGE_PATH
    )

    print(
        "\ntest.jpg をアップロードしてください。"
    )


else:

    # ========================================================
    # テスト画像を読み込む
    # ========================================================

    image = Image.open(

        TEST_IMAGE_PATH

    ).convert(

        "RGB"

    )

    # ========================================================
    # CNNで品質判定
    # ========================================================

    predicted_class, confidence, probabilities = (

        predict_freshness(
            image
        )

    )

    # ========================================================
    # 結果表示
    # ========================================================

    print("\n========================================")

    print(
        "品質判定結果"
    )

    print("========================================")

    print(
        "判定:",
        predicted_class
    )

    print(

        f"信頼度: "
        f"{confidence * 100:.2f}%"

    )

    print(

        f"Fresh: "
        f"{probabilities[0] * 100:.2f}%"

    )

    print(

        f"Rotten: "
        f"{probabilities[1] * 100:.2f}%"

    )

    # ========================================================
    # テスト画像表示
    # ========================================================

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        image
    )

    plt.axis(
        "off"
    )

    plt.title(

        f"Prediction: "
        f"{predicted_class} "
        f"({confidence * 100:.2f}%)"

    )

    plt.show()


# ============================================================
# 32. 保存モデルの確認
# ============================================================

print("\n========================================")

print(
    "Model Save Confirmation"
)

print("========================================")


absolute_path = os.path.abspath(
    CNN_MODEL_PATH
)


print(
    "モデル保存先:"
)


print(
    absolute_path
)


if os.path.exists(
    absolute_path
):

    file_size = os.path.getsize(
        absolute_path
    )

    print(

        f"ファイルサイズ: "
        f"{file_size / (1024 * 1024):.2f} MB"

    )

    print(
        "モデル保存確認: OK"
    )


else:

    print(
        "WARNING: モデルが見つかりません。"
    )
