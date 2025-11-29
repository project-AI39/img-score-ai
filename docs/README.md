# img-score-ai

## セットアップ

### 1. 依存関係のインストール

python uv を使用
```bash
uv sync
```

### 2. PyTorch のインストール（環境に合わせて選択）

PyTorch は環境（CUDA バージョン、CPU/GPU）によってインストール方法が異なります。  
[PyTorch 公式サイト](https://pytorch.org/get-started/locally/) で確認してください。
(念のためuninnstallを先に行うことを推奨)

### 3. モデルファイルの配置

`Hugging Face: laion/CLIP-ViT-B-16-laion2B-s34B-b88K`の`open_clip_model.safetensors`モデルを `models/clip` に配置してください。
