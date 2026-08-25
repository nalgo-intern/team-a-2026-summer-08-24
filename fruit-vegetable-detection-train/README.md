# 果物・野菜物体検出モデル学習

通常のCUDA GPU環境で、対象9品目をYOLO26n / YOLO26s / YOLO26mで学習・比較するスクリプトです。W&Bには各学習Run、val/testの評価指標、正解・予測Bounding Box付き画像を記録します。

学習済みの最終モデルは `models/yolo26m-30epochs-best.pt` にあります。学習データ、実行ログ、中間モデルは容量と個人情報保護のためGitには含めません。

実行前に Kaggle API トークンと W&B APIキーを、プロジェクト直下の `.env` に保存してください。

```dotenv
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-api-key
WANDB_API_KEY=your-wandb-api-key
```

依存関係を導入してから実行します。

```bash
uv sync
uv run python main.py
```

既定では各モデルを30 epochsで学習します。データ取得、9品目への整形、可視化画像の保存、val/test評価、W&B比較まで連続して実行されます。GPUメモリ不足時は各モデルでbatch 16、8、4の順に再試行します。

```bash
# 例: YOLO26sだけを10 epochsで試す
uv run python main.py --models yolo26s --epochs 10
```
