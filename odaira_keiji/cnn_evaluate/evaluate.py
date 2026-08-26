import os
import csv
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# 設定
# ============================================================

# Kerasモデル
MODEL_PATH = "model.keras"

# テスト画像フォルダ
TEST_DIR = "test"

# Fresh / Rotten
FRESH_DIR = os.path.join(
    TEST_DIR,
    "Fresh"
)

ROTTEN_DIR = os.path.join(
    TEST_DIR,
    "Rotten"
)

# 各クラスから使用する画像枚数
TEST_IMAGE_COUNT = 150

# 結果ファイル
RESULT_TXT_PATH = "result.txt"
RESULT_CSV_PATH = "result_full.csv"

# モデル入力サイズ
IMAGE_SIZE = (224, 224)

# クラス名
#
# 0 = Fresh
# 1 = Rotten
#
CLASS_NAMES = [
    "fresh",
    "rotten"
]

# 対応する画像拡張子
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# 画像ファイル取得
# ============================================================

def get_image_paths(folder):

    image_paths = []

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f"フォルダが見つかりません: {folder}"
        )

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
# モデル読み込み
# ============================================================

def load_model():

    print("モデルを読み込んでいます...")

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"モデルが見つかりません: "
            f"{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("モデルの読み込み完了")

    return model


# ============================================================
# 画像前処理
# ============================================================

def preprocess_image(image_path):

    # --------------------------------------------------------
    # 画像読み込み
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # 224×224へリサイズ
    # --------------------------------------------------------

    image = image.resize(
        IMAGE_SIZE
    )

    # --------------------------------------------------------
    # NumPy配列へ変換
    # --------------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # バッチ次元を追加
    #
    # (224, 224, 3)
    #       ↓
    # (1, 224, 224, 3)
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # 重要
    #
    # 学習モデル内部に
    #
    # Rescaling(1.0 / 255)
    #
    # が存在するため、ここでは
    # /255 を行わない。
    # --------------------------------------------------------

    return image_array


# ============================================================
# 1枚の画像を判定
# ============================================================

def predict(
    model,
    image_path
):

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image,
        verbose=0
    )

    # 1枚目の結果
    probabilities = prediction[0]

    # 最大確率のクラス
    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    # クラス名
    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    # 信頼度
    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    return (
        predicted_index,
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# CSV保存
# ============================================================

def save_csv(results):

    print()
    print(
        f"CSVを保存しています: "
        f"{RESULT_CSV_PATH}"
    )

    with open(
        RESULT_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        # ----------------------------------------------------
        # CSVヘッダー
        # ----------------------------------------------------

        writer.writerow([
            "No",
            "Filename",
            "Path",
            "True Class",
            "Predicted Class",
            "Result",
            "Confidence",
            "Confidence (%)",
            "Fresh Probability",
            "Fresh Probability (%)",
            "Rotten Probability",
            "Rotten Probability (%)"
        ])

        # ----------------------------------------------------
        # 各画像の結果
        # ----------------------------------------------------

        for result in results:

            writer.writerow([

                result["number"],

                result["filename"],

                result["path"],

                result["true_class"],

                result["predicted_class"],

                "Correct"
                if result["correct"]
                else "Incorrect",

                f"{result['confidence']:.10f}",

                f"{result['confidence'] * 100:.6f}",

                f"{result['fresh_probability']:.10f}",

                f"{result['fresh_probability'] * 100:.6f}",

                f"{result['rotten_probability']:.10f}",

                f"{result['rotten_probability'] * 100:.6f}"

            ])

    print(
        "result_full.csvを保存しました。"
    )

    print(
        "絶対パス:"
    )

    print(
        os.path.abspath(
            RESULT_CSV_PATH
        )
    )


# ============================================================
# TXT保存
# ============================================================

def save_txt(

    total_count,

    correct_count,

    fresh_total,
    fresh_correct,

    rotten_total,
    rotten_correct,

    accuracy,
    fresh_accuracy,
    rotten_accuracy,

    average_confidence,
    correct_average_confidence,
    incorrect_average_confidence

):

    print()
    print(
        f"結果を保存しています: "
        f"{RESULT_TXT_PATH}"
    )

    # --------------------------------------------------------
    # 不正解数
    # --------------------------------------------------------

    incorrect_count = (
        total_count -
        correct_count
    )

    # --------------------------------------------------------
    # TXT書き込み
    # --------------------------------------------------------

    with open(
        RESULT_TXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "========================================\n"
        )

        f.write(
            "Keras Fresh / Rotten 評価結果\n"
        )

        f.write(
            "========================================\n"
        )

        f.write("\n")


        # ====================================================
        # 評価条件
        # ====================================================

        f.write(
            "【評価条件】\n"
        )

        f.write(
            f"モデル: {MODEL_PATH}\n"
        )

        f.write(
            f"Fresh評価枚数: "
            f"{fresh_total}\n"
        )

        f.write(
            f"Rotten評価枚数: "
            f"{rotten_total}\n"
        )

        f.write(
            f"合計評価枚数: "
            f"{total_count}\n"
        )

        f.write("\n")


        # ====================================================
        # 全体結果
        # ====================================================

        f.write(
            "========================================\n"
        )

        f.write(
            "【全体結果】\n"
        )

        f.write(
            "========================================\n"
        )

        f.write(
            f"評価画像数: "
            f"{total_count}\n"
        )

        f.write(
            f"正解数: "
            f"{correct_count}\n"
        )

        f.write(
            f"不正解数: "
            f"{incorrect_count}\n"
        )

        f.write(
            f"全体正解率: "
            f"{accuracy * 100:.4f}%\n"
        )

        f.write("\n")


        # ====================================================
        # Fresh結果
        # ====================================================

        f.write(
            "========================================\n"
        )

        f.write(
            "【Fresh結果】\n"
        )

        f.write(
            "========================================\n"
        )

        f.write(
            f"評価枚数: "
            f"{fresh_total}\n"
        )

        f.write(
            f"正解数: "
            f"{fresh_correct}\n"
        )

        f.write(
            f"不正解数: "
            f"{fresh_total - fresh_correct}\n"
        )

        f.write(
            f"正解率: "
            f"{fresh_accuracy * 100:.4f}%\n"
        )

        f.write("\n")


        # ====================================================
        # Rotten結果
        # ====================================================

        f.write(
            "========================================\n"
        )

        f.write(
            "【Rotten結果】\n"
        )

        f.write(
            "========================================\n"
        )

        f.write(
            f"評価枚数: "
            f"{rotten_total}\n"
        )

        f.write(
            f"正解数: "
            f"{rotten_correct}\n"
        )

        f.write(
            f"不正解数: "
            f"{rotten_total - rotten_correct}\n"
        )

        f.write(
            f"正解率: "
            f"{rotten_accuracy * 100:.4f}%\n"
        )

        f.write("\n")


        # ====================================================
        # 信頼度
        # ====================================================

        f.write(
            "========================================\n"
        )

        f.write(
            "【信頼度】\n"
        )

        f.write(
            "========================================\n"
        )

        f.write(
            f"全体の信頼度の平均: "
            f"{average_confidence * 100:.4f}%\n"
        )

        f.write(
            f"正解だった画像の信頼度の平均: "
            f"{correct_average_confidence * 100:.4f}%\n"
        )

        f.write(
            f"不正解だった画像の信頼度の平均: "
            f"{incorrect_average_confidence * 100:.4f}%\n"
        )

        f.write("\n")


        # ====================================================
        # 完了
        # ====================================================

        f.write(
            "========================================\n"
        )

        f.write(
            "個別画像の詳細情報は "
            "result_full.csv を参照してください。\n"
        )

        f.write(
            "========================================\n"
        )


    print(
        "result.txtを保存しました。"
    )

    print(
        "絶対パス:"
    )

    print(
        os.path.abspath(
            RESULT_TXT_PATH
        )
    )


# ============================================================
# メイン処理
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "   Keras Fresh / Rotten 評価プログラム"
    )

    print(
        "========================================"
    )


    try:

        # ====================================================
        # 1. Fresh画像取得
        # ====================================================

        print()
        print(
            "Fresh画像を検索しています..."
        )

        fresh_images = get_image_paths(
            FRESH_DIR
        )


        # ====================================================
        # 2. Rotten画像取得
        # ====================================================

        print(
            "Rotten画像を検索しています..."
        )

        rotten_images = get_image_paths(
            ROTTEN_DIR
        )


        print()
        print(
            f"Fresh画像数  : "
            f"{len(fresh_images)}"
        )

        print(
            f"Rotten画像数 : "
            f"{len(rotten_images)}"
        )


        # ====================================================
        # 3. 150枚以上あるか確認
        # ====================================================

        if len(fresh_images) < TEST_IMAGE_COUNT:

            raise ValueError(
                "Fresh画像が150枚未満です。"
            )


        if len(rotten_images) < TEST_IMAGE_COUNT:

            raise ValueError(
                "Rotten画像が150枚未満です。"
            )


        # ====================================================
        # 4. 先頭150枚を使用
        # ====================================================

        selected_fresh = fresh_images[
            :TEST_IMAGE_COUNT
        ]

        selected_rotten = rotten_images[
            :TEST_IMAGE_COUNT
        ]


        # ====================================================
        # 5. 評価データ作成
        #
        # Fresh = 0
        # Rotten = 1
        # ====================================================

        test_data = []


        for image_path in selected_fresh:

            test_data.append(
                (
                    image_path,
                    0
                )
            )


        for image_path in selected_rotten:

            test_data.append(
                (
                    image_path,
                    1
                )
            )


        # ====================================================
        # 6. モデル読み込み
        # ====================================================

        model = load_model()


        # ====================================================
        # 7. 集計変数
        # ====================================================

        total_count = len(
            test_data
        )

        correct_count = 0

        fresh_total = 0
        fresh_correct = 0

        rotten_total = 0
        rotten_correct = 0


        # ----------------------------------------------------
        # 信頼度
        # ----------------------------------------------------

        confidence_values = []

        correct_confidence_values = []

        incorrect_confidence_values = []


        # ----------------------------------------------------
        # 個別結果
        # ----------------------------------------------------

        results = []


        # ====================================================
        # 8. 評価開始
        # ====================================================

        print()
        print(
            "========================================"
        )

        print(
            "評価開始"
        )

        print(
            "========================================"
        )


        for index, (
            image_path,
            true_label
        ) in enumerate(
            test_data,
            start=1
        ):


            # ------------------------------------------------
            # 正解クラス
            # ------------------------------------------------

            true_class = CLASS_NAMES[
                true_label
            ]


            # ------------------------------------------------
            # 推論
            # ------------------------------------------------

            (
                predicted_index,
                predicted_class,
                confidence,
                probabilities
            ) = predict(
                model,
                image_path
            )


            # ------------------------------------------------
            # 正解判定
            # ------------------------------------------------

            correct = (
                predicted_index ==
                true_label
            )


            # ------------------------------------------------
            # 全体集計
            # ------------------------------------------------

            if correct:

                correct_count += 1


            # ------------------------------------------------
            # Fresh集計
            # ------------------------------------------------

            if true_label == 0:

                fresh_total += 1

                if correct:

                    fresh_correct += 1


            # ------------------------------------------------
            # Rotten集計
            # ------------------------------------------------

            elif true_label == 1:

                rotten_total += 1

                if correct:

                    rotten_correct += 1


            # =================================================
            # 信頼度集計
            #
            # 「信頼度」は判定されたクラスの確率
            # =================================================

            confidence_values.append(
                confidence
            )


            if correct:

                correct_confidence_values.append(
                    confidence
                )

            else:

                incorrect_confidence_values.append(
                    confidence
                )


            # ------------------------------------------------
            # 個別結果
            # ------------------------------------------------

            results.append({

                "number":
                    index,

                "path":
                    image_path,

                "filename":
                    os.path.basename(
                        image_path
                    ),

                "true_class":
                    true_class,

                "predicted_class":
                    predicted_class,

                "correct":
                    correct,

                "confidence":
                    confidence,

                "fresh_probability":
                    float(
                        probabilities[0]
                    ),

                "rotten_probability":
                    float(
                        probabilities[1]
                    )

            })


            # ------------------------------------------------
            # コンソール表示
            # ------------------------------------------------

            print(
                f"[{index:3d}/{total_count}] "
                f"{'正解' if correct else '不正解':4s} | "
                f"正解={true_class:6s} | "
                f"判定={predicted_class:6s} | "
                f"信頼度={confidence * 100:6.2f}% | "
                f"{os.path.basename(image_path)}"
            )


        # ====================================================
        # 9. 正解率計算
        # ====================================================

        accuracy = (
            correct_count /
            total_count
        )

        fresh_accuracy = (
            fresh_correct /
            fresh_total
        )

        rotten_accuracy = (
            rotten_correct /
            rotten_total
        )


        # ====================================================
        # 10. 信頼度平均
        # ====================================================

        # ----------------------------------------------------
        # 全体
        # ----------------------------------------------------

        average_confidence = float(
            np.mean(
                confidence_values
            )
        )


        # ----------------------------------------------------
        # 正解画像
        # ----------------------------------------------------

        if len(
            correct_confidence_values
        ) > 0:

            correct_average_confidence = float(
                np.mean(
                    correct_confidence_values
                )
            )

        else:

            correct_average_confidence = 0.0


        # ----------------------------------------------------
        # 不正解画像
        # ----------------------------------------------------

        if len(
            incorrect_confidence_values
        ) > 0:

            incorrect_average_confidence = float(
                np.mean(
                    incorrect_confidence_values
                )
            )

        else:

            incorrect_average_confidence = 0.0


        # ====================================================
        # 11. 結果表示
        # ====================================================

        print()
        print(
            "========================================"
        )

        print(
            "評価結果"
        )

        print(
            "========================================"
        )

        print(
            f"評価画像数 : "
            f"{total_count}"
        )

        print(
            f"正解数     : "
            f"{correct_count}"
        )

        print(
            f"不正解数   : "
            f"{total_count - correct_count}"
        )

        print()

        print(
            f"全体正解率 : "
            f"{accuracy * 100:.2f}%"
        )

        print()

        print(
            "Fresh"
        )

        print(
            f"  正解数 : "
            f"{fresh_correct}/{fresh_total}"
        )

        print(
            f"  正解率 : "
            f"{fresh_accuracy * 100:.2f}%"
        )

        print()

        print(
            "Rotten"
        )

        print(
            f"  正解数 : "
            f"{rotten_correct}/{rotten_total}"
        )

        print(
            f"  正解率 : "
            f"{rotten_accuracy * 100:.2f}%"
        )

        print()

        print(
            "信頼度"
        )

        print(
            f"  全体平均       : "
            f"{average_confidence * 100:.2f}%"
        )

        print(
            f"  正解画像平均   : "
            f"{correct_average_confidence * 100:.2f}%"
        )

        print(
            f"  不正解画像平均 : "
            f"{incorrect_average_confidence * 100:.2f}%"
        )

        print(
            "========================================"
        )


        # ====================================================
        # 12. result.txt保存
        # ====================================================

        save_txt(

            total_count,

            correct_count,

            fresh_total,
            fresh_correct,

            rotten_total,
            rotten_correct,

            accuracy,
            fresh_accuracy,
            rotten_accuracy,

            average_confidence,
            correct_average_confidence,
            incorrect_average_confidence

        )


        # ====================================================
        # 13. result_full.csv保存
        # ====================================================

        save_csv(
            results
        )


        # ====================================================
        # 14. 完了
        # ====================================================

        print()
        print(
            "========================================"
        )

        print(
            "評価処理が完了しました。"
        )

        print(
            "========================================"
        )


    except Exception as e:

        print()
        print(
            "========================================"
        )

        print(
            "エラーが発生しました"
        )

        print(
            "========================================"
        )

        print(
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# プログラム開始
# ============================================================

if __name__ == "__main__":

    main()