# img-score-ai

CLIP 埋め込みを使用した画像好み判定 AI プロジェクト

## 概要

このプロジェクトは、CLIP モデルの画像埋め込みを利用して、
ユーザーの好み（like/dislike）を学習し、新しい画像を自動分類・スコアリングするツールです。

## ディレクトリ構成

```
img-score-ai/
├── data/
│   ├── train/
│   │   ├── like/          # 学習用: 好きな画像
│   │   └── dislike/       # 学習用: 嫌いな画像
│   ├── unlabeled/         # 推論用: 分類/スコアリングしたい画像
│   ├── classified/        # classify.py出力先
│   │   ├── like/
│   │   └── dislike/
│   └── scores/            # score.py出力先 (CSV)
├── models/
│   ├── clip/              # CLIPモデル (open_clip_model.safetensors)
│   └── trained/           # 学習済みモデル出力先
├── scripts/
│   ├── train.py           # 学習スクリプト
│   ├── classify.py        # 画像分類スクリプト
│   ├── score.py           # スコアレポート出力スクリプト
│   └── check_env.py       # 環境確認スクリプト
└── docs/
    ├── README.md          # このファイル
    └── 学習.md            # アクティブラーニングの説明
```

## セットアップ

### 1. 依存関係のインストール

python uv を使用

```bash
uv sync
```

### 2. PyTorch のインストール（環境に合わせて選択）

PyTorch は環境（CUDA バージョン、CPU/GPU）によってインストール方法が異なります。  
[PyTorch 公式サイト](https://pytorch.org/get-started/locally/) で確認してください。
(念のため uninstall を先に行うことを推奨)

```bash
# 例: CUDA 12.1 の場合
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. モデルファイルの配置

`Hugging Face: laion/CLIP-ViT-B-16-laion2B-s34B-b88K`の`open_clip_model.safetensors`を  
`models/clip/` フォルダに配置してください。

ダウンロード先: https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K

### 4. 環境確認

```bash
uv run python scripts/check_env.py
```

## 使い方

### 学習

1. `data/train/like/` に好きな画像を配置
2. `data/train/dislike/` に嫌いな画像を配置
3. 学習を実行:

```bash
uv run python scripts/train.py
```

学習済みモデルは `models/trained/preference_head_v1.pt` に保存されます。

### 推論（画像分類）

画像を like/dislike フォルダに自動振り分け:

```bash
# デフォルト設定で実行
uv run python scripts/classify.py

# オプション指定
uv run python scripts/classify.py --input data/my_images --threshold 0.6 --copy
```

オプション:

- `--input, -i`: 入力フォルダ（デフォルト: data/unlabeled）
- `--output, -o`: 出力フォルダ（デフォルト: data/classified）
- `--threshold, -t`: like/dislike 判定閾値（デフォルト: 0.5）
- `--copy`: 移動ではなくコピーモード
- `--batch-size, -b`: バッチサイズ（デフォルト: 32）

### 推論（スコアレポート）

画像のスコア（0-100%）を CSV 形式で出力:

```bash
# デフォルト設定で実行
uv run python scripts/score.py

# オプション指定
uv run python scripts/score.py --input data/my_images --sort score_desc
```

オプション:

- `--input, -i`: 入力フォルダ（デフォルト: data/unlabeled）
- `--output, -o`: 出力フォルダ（デフォルト: data/scores）
- `--csv-name`: 出力 CSV ファイル名（デフォルト: scores.csv）
- `--threshold, -t`: like/dislike 判定閾値（デフォルト: 0.5）
- `--sort, -s`: ソート順（score_desc, score_asc, name）
- `--batch-size, -b`: バッチサイズ（デフォルト: 32）
- `--quiet, -q`: 統計サマリーを非表示

## 出力例

### classify.py

```
data/classified/
├── like/
│   ├── image001.jpg
│   └── image003.png
└── dislike/
    ├── image002.jpg
    └── image004.webp
```

### score.py (scores.csv)

```csv
filename,score,label
best_image.jpg,95.2,like
nice_photo.png,82.1,like
average.jpg,48.3,dislike
bad_image.png,12.5,dislike
```

## アクティブラーニング

効率的な学習のために、アクティブラーニング手法を推奨します。
詳細は `docs/学習.md` を参照してください。
