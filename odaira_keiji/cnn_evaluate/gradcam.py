import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

MODEL_PATH = "model.keras"

INPUT_IMAGE_BASE = "input"

OUTPUT_IMAGE = "input_gradcam.jpg"

CLASS_NAMES = [
    "Fresh",
    "Rotten"
]

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
]


# ============================================================
# 入力画像を探す
# ============================================================

def find_input_image():

    for ext in IMAGE_EXTENSIONS:

        path = INPUT_IMAGE_BASE + ext

        if os.path.exists(path):
            return path

    return None


# ============================================================
# モデル読み込み
# ============================================================

print("=" * 60)
print("Grad-CAM")
print("=" * 60)
print()

print("モデルを読み込んでいます...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("モデルの読み込みが完了しました。")
print()


# ============================================================
# 入力サイズ
# ============================================================

input_shape = model.input_shape

print(
    "モデル入力サイズ:",
    input_shape
)

IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]

print(
    f"使用する画像サイズ: {IMG_WIDTH} x {IMG_HEIGHT}"
)

print()


# ============================================================
# EfficientNetB0を取得
# ============================================================

efficientnet = model.get_layer(
    "efficientnetb0"
)

print(
    "ベースモデル:",
    efficientnet.name
)


# ============================================================
# Grad-CAM対象レイヤー
# ============================================================

last_conv_layer = efficientnet.get_layer(
    "top_conv"
)

print(
    "Grad-CAM対象レイヤー:",
    last_conv_layer.name
)

print(
    "レイヤー種類:",
    last_conv_layer.__class__.__name__
)

print(
    "レイヤー出力形状:",
    last_conv_layer.output.shape
)

print()


# ============================================================
# EfficientNet内部の
# 「top_conv → EfficientNet出力」を作る
# ============================================================

# top_convの出力からEfficientNetの最終出力までを
# 別のFunctional Modelとして構築する。

efficientnet_tail = tf.keras.models.Model(
    inputs=last_conv_layer.output,
    outputs=efficientnet.output
)


# ============================================================
# EfficientNetの出力から
# 最終分類までを作る
# ============================================================

# model.layersの中からefficientnetb0より後ろにある
# レイヤーを取得する。

efficientnet_index = None

for i, layer in enumerate(model.layers):

    if layer.name == "efficientnetb0":

        efficientnet_index = i
        break


if efficientnet_index is None:

    raise ValueError(
        "efficientnetb0がモデル内に見つかりません。"
    )


# EfficientNetの後ろにあるレイヤー
classifier_layers = model.layers[
    efficientnet_index + 1:
]


print(
    "分類部分のレイヤー:"
)

for layer in classifier_layers:

    print(
        f"  {layer.name} "
        f"({layer.__class__.__name__})"
    )

print()


# ============================================================
# Grad-CAM計算
# ============================================================

def make_gradcam(
    img_array,
    class_index
):

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # 1. EfficientNetのtop_convまで計算
        # ----------------------------------------------------

        conv_outputs = tf.keras.Model(
            inputs=efficientnet.input,
            outputs=last_conv_layer.output
        )(
            img_array,
            training=False
        )

        # 勾配計算対象として明示的に監視
        tape.watch(conv_outputs)

        # ----------------------------------------------------
        # 2. top_conv以降を計算
        # ----------------------------------------------------

        x = efficientnet_tail(
            conv_outputs,
            training=False
        )

        # ----------------------------------------------------
        # 3. EfficientNet以降の分類層
        # ----------------------------------------------------

        for layer in classifier_layers:

            x = layer(
                x,
                training=False
            )

        predictions = x

        # 指定クラスの出力
        class_output = predictions[
            :,
            class_index
        ]

    # --------------------------------------------------------
    # 勾配を取得
    # --------------------------------------------------------

    grads = tape.gradient(
        class_output,
        conv_outputs
    )

    if grads is None:

        raise ValueError(
            "Grad-CAMの勾配を取得できませんでした。"
        )

    # --------------------------------------------------------
    # 各チャンネルの重要度
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # 特徴マップを重要度で重み付け
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # 0～1に正規化
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:

        heatmap /= max_value

    return heatmap.numpy()


# ============================================================
# 画像読み込み
# ============================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = np.array(
        image
    )

    resized_image = image.resize(
        (
            IMG_WIDTH,
            IMG_HEIGHT
        )
    )

    img_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # バッチ次元
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return original_image, img_array


# ============================================================
# Grad-CAM画像保存
# ============================================================

def save_gradcam(
    original_image,
    heatmap,
    output_path
):

    # --------------------------------------------------------
    # ヒートマップを元画像サイズに拡大
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        (
            original_image.shape[1],
            original_image.shape[0]
        ),
        Image.Resampling.BILINEAR
    )

    heatmap = np.array(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # カラーマップ
    # --------------------------------------------------------

    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap)[:, :, :3]

    # --------------------------------------------------------
    # 元画像
    # --------------------------------------------------------

    original = (
        original_image.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # 合成
    # --------------------------------------------------------

    overlay = (
        original * 0.55
        + colored_heatmap * 0.45
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    output_image = Image.fromarray(
        np.uint8(
            overlay * 255
        )
    )

    output_image.save(
        output_path,
        "JPEG",
        quality=95
    )


# ============================================================
# メイン
# ============================================================

def main():

    # --------------------------------------------------------
    # 入力画像検索
    # --------------------------------------------------------

    input_path = find_input_image()

    if input_path is None:

        print(
            "エラー: input.jpg / input.png などの"
            "入力画像が見つかりません。"
        )

        print()

        print(
            "対応形式:"
        )

        for ext in IMAGE_EXTENSIONS:

            print(
                f"  input{ext}"
            )

        return


    print(
        "入力画像:",
        input_path
    )

    print()


    # --------------------------------------------------------
    # 画像読み込み
    # --------------------------------------------------------

    original_image, img_array = (
        load_image(
            input_path
        )
    )


    # --------------------------------------------------------
    # 通常のモデル推論
    # --------------------------------------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )


    # --------------------------------------------------------
    # 推論結果
    # --------------------------------------------------------

    print(
        "========== 推論結果 =========="
    )

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{class_name}: "
            f"{probabilities[i]:.4f}"
        )

    print()

    print(
        "予測:",
        predicted_class
    )

    print(
        f"信頼度: {confidence:.4f}"
    )

    print()


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print(
        "Grad-CAMを計算しています..."
    )

    heatmap = make_gradcam(
        img_array,
        predicted_index
    )


    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_gradcam(
        original_image,
        heatmap,
        OUTPUT_IMAGE
    )


    # --------------------------------------------------------
    # 完了
    # --------------------------------------------------------

    print()

    print(
        "Grad-CAMの作成が完了しました。"
    )

    print(
        "保存先:",
        os.path.abspath(
            OUTPUT_IMAGE
        )
    )

    print()

    print("=" * 60)


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()