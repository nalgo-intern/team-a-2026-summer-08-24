import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

MODEL_PATH = "model.keras"

INPUT_DIR = "input"

OUTPUT_DIR = "output"

CLASS_NAMES = [
    "fresh",
    "rotten"
]

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


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
# 入力サイズ取得
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
# EfficientNetB0取得
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
# EfficientNetのtop_convまで取得するモデル
# ============================================================

conv_model = tf.keras.models.Model(
    inputs=efficientnet.input,
    outputs=last_conv_layer.output
)


# ============================================================
# EfficientNetのtop_conv以降
# ============================================================

efficientnet_tail = tf.keras.models.Model(
    inputs=last_conv_layer.output,
    outputs=efficientnet.output
)


# ============================================================
# 分類層取得
# ============================================================

efficientnet_index = None

for i, layer in enumerate(model.layers):

    if layer.name == "efficientnetb0":
        efficientnet_index = i
        break


if efficientnet_index is None:

    raise ValueError(
        "efficientnetb0がモデル内に見つかりません。"
    )


classifier_layers = model.layers[
    efficientnet_index + 1:
]


print("分類部分のレイヤー:")

for layer in classifier_layers:

    print(
        f"  {layer.name} "
        f"({layer.__class__.__name__})"
    )

print()


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

    # ========================================================
    # 注意
    #
    # ここでは /255.0 を行わない
    # ========================================================

    img_array = np.array(
        resized_image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return original_image, img_array


# ============================================================
# Grad-CAM
# ============================================================

def make_gradcam(
    img_array,
    class_index
):

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # top_convまで
        # ----------------------------------------------------

        conv_outputs = conv_model(
            img_array,
            training=False
        )

        tape.watch(
            conv_outputs
        )

        # ----------------------------------------------------
        # EfficientNet後半
        # ----------------------------------------------------

        x = efficientnet_tail(
            conv_outputs,
            training=False
        )

        # ----------------------------------------------------
        # 分類層
        # ----------------------------------------------------

        for layer in classifier_layers:

            x = layer(
                x,
                training=False
            )

        predictions = x

        class_output = predictions[
            :,
            class_index
        ]

    # --------------------------------------------------------
    # 勾配取得
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
    # チャンネルごとの重要度
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # 重み付け
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
    # 0～1へ正規化
    #
    # これは「入力画像の正規化」ではなく、
    # Grad-CAMのヒートマップを表示するための正規化。
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if max_value > 0:

        heatmap /= max_value

    return heatmap.numpy()


# ============================================================
# Grad-CAM画像保存
# ============================================================

def save_gradcam(
    original_image,
    heatmap,
    output_path
):

    # --------------------------------------------------------
    # ヒートマップを元画像サイズへ
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
        output_path
    )


# ============================================================
# 入力フォルダの画像取得
# ============================================================

def get_image_files():

    if not os.path.exists(INPUT_DIR):

        raise FileNotFoundError(
            f"入力フォルダがありません: {INPUT_DIR}"
        )

    image_files = []

    for filename in os.listdir(INPUT_DIR):

        filepath = os.path.join(
            INPUT_DIR,
            filename
        )

        if not os.path.isfile(filepath):
            continue

        if not filename.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            continue

        image_files.append(
            filepath
        )

    # ファイル名順に処理
    image_files.sort()

    return image_files


# ============================================================
# メイン
# ============================================================

def main():

    # --------------------------------------------------------
    # 出力フォルダ作成
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # 入力画像取得
    # --------------------------------------------------------

    image_files = get_image_files()

    if len(image_files) == 0:

        print(
            "inputフォルダに画像がありません。"
        )

        return


    print(
        f"処理対象画像: {len(image_files)}枚"
    )

    print()


    # --------------------------------------------------------
    # 全画像処理
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            print(
                f"[{index}/{len(image_files)}] "
                f"{os.path.basename(image_path)}"
            )

            # ------------------------------------------------
            # 画像読み込み
            # ------------------------------------------------

            original_image, img_array = (
                load_image(
                    image_path
                )
            )


            # ------------------------------------------------
            # 推論
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            heatmap = make_gradcam(
                img_array,
                predicted_index
            )


            # ------------------------------------------------
            # ファイル名作成
            # ------------------------------------------------

            filename = os.path.basename(
                image_path
            )

            name, extension = os.path.splitext(
                filename
            )

            # 小数第1位
            confidence_percent = (
                confidence * 100
            )

            confidence_text = (
                f"{confidence_percent:.1f}"
            )

            output_filename = (
                f"{name}_"
                f"{predicted_class}_"
                f"{confidence_text}"
                f"{extension}"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_filename
            )


            # ------------------------------------------------
            # 保存
            # ------------------------------------------------

            save_gradcam(
                original_image,
                heatmap,
                output_path
            )


            # ------------------------------------------------
            # 結果表示
            # ------------------------------------------------

            print(
                f"  → {predicted_class}"
            )

            print(
                f"  → 信頼度: "
                f"{confidence_percent:.1f}%"
            )

            print(
                f"  → 保存: "
                f"{output_filename}"
            )

            print()


        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            print()


    # --------------------------------------------------------
    # 完了
    # --------------------------------------------------------

    print("=" * 60)
    print("すべての処理が完了しました。")
    print(
        "出力先:",
        os.path.abspath(OUTPUT_DIR)
    )
    print("=" * 60)


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()