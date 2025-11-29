"""
画像分類推論スクリプト（好き/嫌いフォルダに振り分け）

このスクリプトは、学習済みモデルを使用して画像を「好き」「嫌い」に分類し、
それぞれのフォルダに振り分けます。

【使用方法】
    uv run python scripts/classify.py

【入出力】
    入力: data/unlabeled/         <- 判別したい画像を配置
    出力: data/classified/
          ├── like/              <- 好きと判定された画像
          └── dislike/           <- 嫌いと判定された画像

【オプション】
    --input, -i   : 入力フォルダのパス（デフォルト: data/unlabeled）
    --output, -o  : 出力フォルダのパス（デフォルト: data/classified）
    --threshold   : 分類閾値 0.0-1.0（デフォルト: 0.5）
    --copy        : 移動ではなくコピーする
    --batch-size  : バッチサイズ（デフォルト: 32）
"""

from __future__ import annotations

import argparse
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import open_clip
from tqdm import tqdm


# ===== 設定 =====
CONFIG = {
    "clip_model": "ViT-B-16",
    "clip_checkpoint": Path("models/clip/open_clip_model.safetensors"),
    "trained_model": Path("models/trained/preference_head_v1.pt"),
    "input_dir": Path("data/unlabeled"),
    "output_dir": Path("data/classified"),
    "hidden_dim": 256,
    "dropout": 0.3,
    "batch_size": 32,
    "threshold": 0.5,
}


# ===== モデル定義（train.py と同じ構造） =====
class PreferenceHead(nn.Module):
    """画像の好み度を予測するニューラルネットワークヘッド"""

    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_image_paths(directory: Path) -> list[Path]:
    """
    ディレクトリから画像ファイルのパスを取得

    大文字・小文字を区別せずに検出する（.JPG, .Jpg なども対応）

    Args:
        directory (Path): 検索するディレクトリ

    Returns:
        list[Path]: 画像ファイルのパスリスト（ソート済み）
    """
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
    }
    paths = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    return sorted(paths)


def load_models(device: str) -> tuple:
    """CLIPモデルと学習済みモデルをロード"""
    # CLIPモデルのロード
    print("Loading CLIP model...")
    if not CONFIG["clip_checkpoint"].exists():
        raise FileNotFoundError(
            f"CLIPモデルが見つかりません: {CONFIG['clip_checkpoint']}\n"
            f"models/clip/README.txt を参照してダウンロードしてください。"
        )

    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name=CONFIG["clip_model"],
        pretrained=str(CONFIG["clip_checkpoint"]),
    )
    clip_model = clip_model.to(device)
    clip_model.eval()

    # 学習済みモデルのロード
    print("Loading trained model...")
    if not CONFIG["trained_model"].exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {CONFIG['trained_model']}\n"
            f"先に train.py で学習を実行してください。"
        )

    head = PreferenceHead(
        input_dim=512,
        hidden_dim=CONFIG["hidden_dim"],
        dropout=CONFIG["dropout"],
    ).to(device)
    head.load_state_dict(torch.load(CONFIG["trained_model"], weights_only=True))
    head.eval()

    return clip_model, preprocess, head


def predict_batch(
    image_paths: list[Path],
    clip_model,
    preprocess,
    head: nn.Module,
    device: str,
    batch_size: int,
) -> list[tuple[Path, float]]:
    """バッチ処理で画像の好み度を予測"""
    results = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []
            valid_paths = []

            # 画像を読み込んで前処理
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            if not batch_tensors:
                continue

            # バッチ処理
            batch_tensor = torch.stack(batch_tensors).to(device)

            # CLIP埋め込み計算
            features = clip_model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

            # 好み度予測
            logits = head(features)
            probs = torch.sigmoid(logits).cpu().numpy()

            # 結果を記録
            for path, prob in zip(valid_paths, probs):
                results.append((path, float(prob)))

            # GPUメモリ解放
            del batch_tensor, features
            if device == "cuda":
                torch.cuda.empty_cache()

    return results


def classify_images(
    results: list[tuple[Path, float]],
    output_dir: Path,
    threshold: float,
    copy_mode: bool,
) -> tuple[int, int]:
    """画像を好き/嫌いフォルダに振り分け"""
    like_dir = output_dir / "like"
    dislike_dir = output_dir / "dislike"

    # 出力ディレクトリを作成
    like_dir.mkdir(parents=True, exist_ok=True)
    dislike_dir.mkdir(parents=True, exist_ok=True)

    like_count = 0
    dislike_count = 0
    processed_paths: set[Path] = set()  # 処理済みパスを記録

    for img_path, prob in tqdm(results, desc="Classifying"):
        # 既に処理済みのパスはスキップ（重複対策）
        if img_path in processed_paths:
            continue
        processed_paths.add(img_path)

        # 移動モードの場合、ファイルが存在するか確認
        if not copy_mode and not img_path.exists():
            print(f"Warning: File not found, skipping: {img_path}")
            continue

        if prob >= threshold:
            dest_dir = like_dir
            like_count += 1
        else:
            dest_dir = dislike_dir
            dislike_count += 1

        dest_path = dest_dir / img_path.name

        # ファイル名が重複する場合は番号を付ける
        if dest_path.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if copy_mode:
            shutil.copy2(img_path, dest_path)
        else:
            shutil.move(str(img_path), dest_path)

    return like_count, dislike_count


def main():
    parser = argparse.ArgumentParser(description="画像を好き/嫌いに分類して振り分け")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=CONFIG["input_dir"],
        help=f"入力フォルダ（デフォルト: {CONFIG['input_dir']}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=CONFIG["output_dir"],
        help=f"出力フォルダ（デフォルト: {CONFIG['output_dir']}）",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=CONFIG["threshold"],
        help=f"分類閾値 0.0-1.0（デフォルト: {CONFIG['threshold']}）",
    )
    parser.add_argument(
        "--copy", "-c", action="store_true", help="移動ではなくコピーする"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=CONFIG["batch_size"],
        help=f"バッチサイズ（デフォルト: {CONFIG['batch_size']}）",
    )

    args = parser.parse_args()

    # デバイス選択
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 入力フォルダの確認
    if not args.input.exists():
        print(f"Error: 入力フォルダが見つかりません: {args.input}")
        print(f"  {args.input}/ に判別したい画像を配置してください")
        return 1

    # 画像パスを取得
    image_paths = get_image_paths(args.input)
    if not image_paths:
        print(f"Error: 画像が見つかりません: {args.input}")
        return 1

    print(f"Found {len(image_paths)} images")

    # モデルをロード
    try:
        clip_model, preprocess, head = load_models(device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # 推論実行
    print("\nPredicting preferences...")
    results = predict_batch(
        image_paths=image_paths,
        clip_model=clip_model,
        preprocess=preprocess,
        head=head,
        device=device,
        batch_size=args.batch_size,
    )

    # 分類実行
    print(f"\nClassifying images (threshold: {args.threshold})...")
    action = "Copying" if args.copy else "Moving"
    print(f"{action} to {args.output}/")

    like_count, dislike_count = classify_images(
        results=results,
        output_dir=args.output,
        threshold=args.threshold,
        copy_mode=args.copy,
    )

    # 結果表示
    print("\n" + "=" * 50)
    print("Classification Complete!")
    print("=" * 50)
    print(f"  Like:    {like_count} images → {args.output}/like/")
    print(f"  Dislike: {dislike_count} images → {args.output}/dislike/")
    print(f"  Total:   {like_count + dislike_count} images")

    return 0


if __name__ == "__main__":
    exit(main())
