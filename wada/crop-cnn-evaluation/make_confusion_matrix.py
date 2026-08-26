"""評価CSVから混同行列をCSV・PNG・テキストで作成する。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/food_freshness_evaluation/cnn_evaluation_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/food_freshness_evaluation"),
    )
    return parser.parse_args()


def create_confusion_matrix(input_path: Path, output_dir: Path) -> None:
    labels = ["fresh", "rotten"]
    y_true: list[str] = []
    y_pred: list[str] = []
    with input_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            expected = row.get("expected", "").lower()
            predicted = (row.get("final_predicted") or row.get("predicted") or row.get("crop_predicted") or "").lower()
            if expected in labels and predicted in labels:
                y_true.append(expected)
                y_pred.append(predicted)

    # 行が正解ラベル、列が予測ラベルになるように固定する。
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", "fresh", "rotten"])
        writer.writerow(["fresh", matrix[0, 0], matrix[0, 1]])
        writer.writerow(["rotten", matrix[1, 0], matrix[1, 1]])

    total = int(matrix.sum())
    correct = int(matrix.trace()) if total else 0
    accuracy = correct / total * 100 if total else 0.0
    text_path = output_dir / "confusion_matrix.txt"
    text_path.write_text(
        "混同行列:\n\n"
        "                予測Fresh  予測Rotten\n"
        f"実際Fresh        {matrix[0, 0]:>4}       {matrix[0, 1]:>4}\n"
        f"実際Rotten       {matrix[1, 0]:>4}       {matrix[1, 1]:>4}\n\n"
        f"Accuracy: {accuracy:.2f}% ({correct}/{total})\n",
        encoding="utf-8",
    )

    # 参照画像と同じ、行方向を割合にしたBluesヒートマップを作る。
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(normalized_matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"overall accuracy:{correct / total if total else 0.0:.10f}",
    )
    threshold = normalized_matrix.max() / 2.0
    for row_index in range(normalized_matrix.shape[0]):
        for column_index in range(normalized_matrix.shape[1]):
            color = "white" if normalized_matrix[row_index, column_index] > threshold else "black"
            axis.text(
                column_index,
                row_index,
                f"{normalized_matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color=color,
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close(figure)

    print(f"混同行列CSV: {csv_path}")
    print(f"混同行列画像: {output_dir / 'confusion_matrix.png'}")
    print(f"混同行列テキスト: {text_path}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    args = parse_args()
    create_confusion_matrix(args.input, args.output_dir)
