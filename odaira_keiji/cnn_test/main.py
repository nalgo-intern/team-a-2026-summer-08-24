import os
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# 設定
# ============================================================

# Kerasモデル
MODEL_PATH = "model.keras"

# 判定する画像
IMAGE_PATH = "input.jpg"

# 判定結果を保存するファイル
RESULT_PATH = "result.txt"

# モデルに入力する画像サイズ
IMAGE_SIZE = (224, 224)

# クラス名
# ※ 学習時のクラス番号に合わせて変更してください
CLASS_NAMES = [
    "fresh",
    "not_fresh"
]


# ============================================================
# モデル読み込み
# ============================================================

def load_model():
    print("モデルを読み込んでいます...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"モデルが見つかりません: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    print("モデルの読み込み完了")

    return model


# ============================================================
# 画像読み込み・前処理
# ============================================================

def preprocess_image(image_path):
    print(f"画像を読み込んでいます: {image_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"画像が見つかりません: {image_path}"
        )

    # RGB画像として読み込み
    image = Image.open(image_path).convert("RGB")

    # モデルの入力サイズに変更
    image = image.resize(IMAGE_SIZE)

    # NumPy配列へ変換
    image_array = np.array(image, dtype=np.float32)

    # 0～1に正規化
    image_array = image_array / 255.0

    # バッチ次元を追加
    #
    # 224 × 224 × 3
    #       ↓
    # 1 × 224 × 224 × 3
    #
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# 判定
# ============================================================

def predict(model, image):
    print("判定しています...")

    # 推論
    prediction = model.predict(image, verbose=0)

    # 出力結果
    probabilities = prediction[0]

    # 最も確率が高いクラス
    class_index = int(np.argmax(probabilities))

    # クラス名
    class_name = CLASS_NAMES[class_index]

    # 信頼度
    confidence = float(probabilities[class_index])

    return class_name, confidence, probabilities


# ============================================================
# 結果保存
# ============================================================

def save_result(class_name, confidence, probabilities):
    print(f"結果を保存しています: {RESULT_PATH}")

    with open(RESULT_PATH, "w", encoding="utf-8") as f:

        f.write("=== 判定結果 ===\n")
        f.write(f"判定: {class_name}\n")
        f.write(f"信頼度: {confidence:.4f}\n")

        f.write("\n=== クラス別確率 ===\n")

        for i, probability in enumerate(probabilities):
            f.write(
                f"{CLASS_NAMES[i]}: "
                f"{float(probability):.4f}\n"
            )

    print("結果を保存しました")


# ============================================================
# メイン処理
# ============================================================

def main():

    print("========================================")
    print("       Keras 画像判定プログラム")
    print("========================================")

    try:

        # 1. モデル読み込み
        model = load_model()

        # 2. 画像読み込み・前処理
        image = preprocess_image(IMAGE_PATH)

        # 3. 判定
        class_name, confidence, probabilities = predict(
            model,
            image
        )

        # 4. 結果表示
        print()
        print("========================================")
        print("判定結果")
        print("========================================")
        print(f"判定      : {class_name}")
        print(f"信頼度    : {confidence:.4f}")
        print(f"信頼度    : {confidence * 100:.2f}%")
        print("========================================")

        # 5. result.txtへ保存
        save_result(
            class_name,
            confidence,
            probabilities
        )

        print()
        print("処理が完了しました。")

    except Exception as e:

        print()
        print("========================================")
        print("エラーが発生しました")
        print("========================================")
        print(e)


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    main()