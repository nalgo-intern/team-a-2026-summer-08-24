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

CSVの正解ラベルは、テスト分割時の `Fresh` / `Rotten` フォルダーから設定し、クロップ成功後のCNN予測と比較して正解率を計算します。
