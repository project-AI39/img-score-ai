"""
画像スコアリング推論スクリプト（評価スコアをCSV出力）

このスクリプトは、学習済みモデルを使用して画像の好み度スコア（0-100%）を計算し、
結果をCSVファイルに出力します。

【使用方法】
    uv run python scripts/score.py

【入出力】
    入力: data/unlabeled/              <- スコアリングしたい画像を配置
    出力: data/scores/
          └── scores.csv               <- ファイル名とスコアの一覧

【CSVフォーマット】
    filename,score,label
    image001.jpg,87.5,like
    image002.png,23.1,dislike
    ...

【オプション】
    --input, -i     : 入力フォルダのパス（デフォルト: data/unlabeled）
    --output, -o    : 出力フォルダのパス（デフォルト: data/scores）
    --csv-name      : 出力CSVファイル名（デフォルト: scores.csv）
    --threshold     : like/dislike判定の閾値（デフォルト: 0.5）
    --sort          : ソート順（score_desc, score_asc, name）
    --batch-size    : バッチサイズ（デフォルト: 32）
"""

from __future__ import annotations

import argparse
import csv
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
    "output_dir": Path("data/scores"),
    "csv_name": "scores.csv",
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
    """ディレクトリから画像ファイルのパスを取得"""
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
        "*.WEBP",
    ]
    paths = []
    for ext in image_extensions:
        paths.extend(directory.glob(ext))
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
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Scoring"):
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


def save_scores_csv(
    results: list[tuple[Path, float]],
    output_path: Path,
    threshold: float,
    sort_by: str,
) -> None:
    """スコア結果をCSVファイルに保存"""
    # ソート
    if sort_by == "score_desc":
        results = sorted(results, key=lambda x: x[1], reverse=True)
    elif sort_by == "score_asc":
        results = sorted(results, key=lambda x: x[1])
    elif sort_by == "name":
        results = sorted(results, key=lambda x: x[0].name)

    # CSV出力
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "score", "label"])

        for img_path, prob in results:
            score = prob * 100  # 0-100%に変換
            label = "like" if prob >= threshold else "dislike"
            writer.writerow([img_path.name, f"{score:.1f}", label])


def print_summary(results: list[tuple[Path, float]], threshold: float) -> None:
    """統計サマリーを表示"""
    if not results:
        return

    scores = [prob * 100 for _, prob in results]
    like_count = sum(1 for _, prob in results if prob >= threshold)
    dislike_count = len(results) - like_count

    print("\n" + "=" * 50)
    print("Score Statistics")
    print("=" * 50)
    print(f"  Total images:  {len(results)}")
    print(f"  Like:          {like_count} ({like_count / len(results) * 100:.1f}%)")
    print(
        f"  Dislike:       {dislike_count} ({dislike_count / len(results) * 100:.1f}%)"
    )
    print("-" * 50)
    print(f"  Average score: {sum(scores) / len(scores):.1f}%")
    print(f"  Max score:     {max(scores):.1f}%")
    print(f"  Min score:     {min(scores):.1f}%")

    # Top 5 / Bottom 5
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    print("\n" + "-" * 50)
    print("Top 5 (Most Liked)")
    print("-" * 50)
    for img_path, prob in sorted_results[:5]:
        print(f"  {prob * 100:5.1f}%  {img_path.name}")

    print("\n" + "-" * 50)
    print("Bottom 5 (Least Liked)")
    print("-" * 50)
    for img_path, prob in sorted_results[-5:]:
        print(f"  {prob * 100:5.1f}%  {img_path.name}")


def main():
    parser = argparse.ArgumentParser(description="画像の好み度スコアを計算してCSV出力")
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
        "--csv-name",
        type=str,
        default=CONFIG["csv_name"],
        help=f"出力CSVファイル名（デフォルト: {CONFIG['csv_name']}）",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=CONFIG["threshold"],
        help=f"like/dislike判定の閾値（デフォルト: {CONFIG['threshold']}）",
    )
    parser.add_argument(
        "--sort",
        "-s",
        choices=["score_desc", "score_asc", "name"],
        default="score_desc",
        help="ソート順（デフォルト: score_desc）",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=CONFIG["batch_size"],
        help=f"バッチサイズ（デフォルト: {CONFIG['batch_size']}）",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="統計サマリーを表示しない"
    )

    args = parser.parse_args()

    # デバイス選択
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 入力フォルダの確認
    if not args.input.exists():
        print(f"Error: 入力フォルダが見つかりません: {args.input}")
        print(f"  {args.input}/ にスコアリングしたい画像を配置してください")
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
    print("\nCalculating scores...")
    results = predict_batch(
        image_paths=image_paths,
        clip_model=clip_model,
        preprocess=preprocess,
        head=head,
        device=device,
        batch_size=args.batch_size,
    )

    # 出力ディレクトリを作成
    args.output.mkdir(parents=True, exist_ok=True)

    # CSV出力
    output_path = args.output / args.csv_name
    save_scores_csv(
        results=results,
        output_path=output_path,
        threshold=args.threshold,
        sort_by=args.sort,
    )

    print(f"\nScores saved to: {output_path}")

    # 統計サマリー表示
    if not args.quiet:
        print_summary(results, args.threshold)

    print("\n" + "=" * 50)
    print("Scoring Complete!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
