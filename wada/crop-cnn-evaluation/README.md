# BiRefNetクロップ + CNN評価

## 概要

1. `food_freshness_evaluation/dataset` にKaggleのFood Freshness Datasetをダウンロード（展開後の`Dataset/Fresh`、`Dataset/Rotten`も自動検出）
2. BiRefNetで画像内の食品を前景抽出
3. 最大前景成分のBBoxで元画像をクロップ
4. クロップ画像をCNNへ入力して `fresh / rotten` を判定
5. クロップ画像、前景画像、BBox付き画像、CSVを保存

評価対象は、学習コード `odaira_keiji/cnn_test/cnntrain_code.py` と同じ `seed=42`、Fresh/Rottenごとの `70% / 15% / 15%` 分割で作成したテストデータから、既定で200枚を抽出します。

## 実行

`.env.example` を `.env` にコピーし、Kaggle APIの認証情報を設定してから実行します。

```bash
cp .env.example .env
```

`.env` はGit管理対象外です。

```bash
cd wada/crop-cnn-evaluation
uv sync
uv run python evaluate.py --device cpu
```

初回実行時にKaggleデータセットとBiRefNetモデルをダウンロードします。

GPUを使う場合:

```bash
uv run python evaluate.py --device cuda
```

## オプション

```bash
# テストデータから200枚だけ評価（既定値）
uv run python evaluate.py --device cpu --seed 42

# 既存CSVから混同行列だけを作成
uv run python make_confusion_matrix.py

# 学習時と異なるseedでテスト分割を作る場合
uv run python evaluate.py --device cpu --seed 42

# 出力を削除して再評価
uv run python evaluate.py --device cpu --clean-output

# CNNモデルを変更
uv run python evaluate.py --device cpu --cnn-model ../../main/model/best_freshness_model_odaira.keras
```

## 出力先

`data/food_freshness_evaluation/` 以下に保存されます。

- `dataset/`: ダウンロードしたデータセット
- `cnn_crops/`: CNNへ入力したクロップ画像
- `birefnet_foregrounds/`: BiRefNetの前景抽出画像
- `result_images/`: BBoxと判定結果を描画した画像
- `correct_images/`: CNNの正解画像（最大200枚）
- `incorrect_images/`: CNNの不正解・不検出画像（最大200枚）
- `cnn_evaluation_results.csv`: 全画像の評価結果
- `confusion_matrix.csv`: 混同行列の数値
- `confusion_matrix.png`: 混同行列の画像
- `confusion_matrix.txt`: 日本語の混同行列

CSVの正解ラベルは、テスト分割時の `Fresh` / `Rotten` フォルダーから設定し、クロップ成功後のCNN予測と比較して正解率を計算します。

## 既存のcnn_evaluate結果との比較

`odaira_keiji/cnn_evaluate/result_full.csv` と、その `correct` / `false` フォルダー内の画像を使って、元画像判定とBiRefNetクロップ後判定を比較します。Kaggleからの再ダウンロードは行いません。

```bash
uv run python compare_existing_results.py --device cpu --clean-output
```

出力先は `data/cnn_evaluate_comparison/` です。

- `comparison_report.csv`: 各画像の元判定・クロップ後判定・判定遷移
- `comparison_report.txt`: 正解→不正解、不正解→正解などの集計レポート
- `correct_to_incorrect/`: クロップで正解から不正解になった画像
- `incorrect_to_correct/`: クロップで不正解から正解になった画像
- 各遷移フォルダー内の `original/` と `cropped/` で、元画像と実際にCNNへ入力したクロップ画像を比較可能
- `cropped_images/`: 全クロップ画像
- `crop_confusion_matrix/confusion_matrix.png`: 参照画像と同じ正規化ヒートマップ

比較時の修正方針:

- BiRefNetで検出できない場合は元画像CNNへフォールバック
- BBoxの上下左右に12%の余白を追加
- 元画像CNNとクロップ画像CNNの両方がRottenならRottenを採用
- 2つの判定が異なる場合も安全側としてRottenを採用
