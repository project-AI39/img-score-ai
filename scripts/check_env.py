"""
環境構築完了確認スクリプト

このスクリプトは、プロジェクトに必要なライブラリとファイルが
すべて正しくセットアップされているかを確認します。

使用方法:
    uv run python scripts/check_env.py
"""

from pathlib import Path
import sys


def print_header(title: str) -> None:
    """セクションヘッダーを表示"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(name: str, status: bool, detail: str = "") -> None:
    """ステータスを表示"""
    icon = "✓" if status else "✗"
    status_text = "OK" if status else "NG"
    print(f"  [{icon}] {name}: {status_text}")
    if detail:
        print(f"      → {detail}")


def check_python_version() -> bool:
    """Pythonバージョンを確認"""
    print_header("Python バージョン確認")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    # Python 3.12以上が必要
    is_ok = version.major >= 3 and version.minor >= 12
    print_status(
        "Python",
        is_ok,
        f"バージョン {version_str}" + ("" if is_ok else " (3.12以上が必要)"),
    )
    return is_ok


def check_pytorch() -> bool:
    """PyTorchの確認"""
    print_header("PyTorch 確認")

    all_ok = True

    # torch
    try:
        import torch

        version = torch.__version__
        print_status("torch", True, f"バージョン {version}")

        # CUDA確認
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            print_status("CUDA", True, f"バージョン {cuda_version}")
            print_status("GPU", True, device_name)
        else:
            print_status("CUDA", False, "利用不可（CPUモードで動作します）")
            print(
                "      ⚠ GPUを使用する場合はCUDA対応のPyTorchをインストールしてください"
            )
            print("        参考: https://pytorch.org/get-started/locally/")
    except ImportError:
        print_status("torch", False, "インストールされていません")
        print(
            "      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
        all_ok = False

    # torchvision
    try:
        import torchvision

        version = torchvision.__version__
        print_status("torchvision", True, f"バージョン {version}")
    except ImportError:
        print_status("torchvision", False, "インストールされていません")
        all_ok = False

    return all_ok


def check_libraries() -> bool:
    """必要なライブラリの確認"""
    print_header("必要なライブラリ確認")

    all_ok = True

    # open_clip
    try:
        import open_clip

        version = open_clip.__version__
        print_status("open-clip-torch", True, f"バージョン {version}")
    except ImportError:
        print_status("open-clip-torch", False, "インストールされていません")
        print("      uv sync または pip install open-clip-torch")
        all_ok = False
    except AttributeError:
        # バージョン属性がない場合
        print_status("open-clip-torch", True, "インストール済み（バージョン不明）")

    # PIL (Pillow)
    try:
        import PIL

        version = PIL.__version__
        print_status("Pillow", True, f"バージョン {version}")
    except ImportError:
        print_status("Pillow", False, "インストールされていません")
        print("      uv sync または pip install pillow")
        all_ok = False

    # numpy
    try:
        import numpy as np

        version = np.__version__
        print_status("numpy", True, f"バージョン {version}")
    except ImportError:
        print_status("numpy", False, "インストールされていません")
        all_ok = False

    # tqdm
    try:
        import tqdm

        version = tqdm.__version__
        print_status("tqdm", True, f"バージョン {version}")
    except ImportError:
        print_status("tqdm", False, "インストールされていません")
        all_ok = False

    return all_ok


def check_model_files() -> bool:
    """モデルファイルの確認"""
    print_header("モデルファイル確認")

    all_ok = True

    # CLIPモデル
    clip_model_path = Path("models/clip/open_clip_model.safetensors")
    if clip_model_path.exists():
        size_mb = clip_model_path.stat().st_size / (1024 * 1024)
        print_status("CLIPモデル", True, f"{clip_model_path} ({size_mb:.1f} MB)")
    else:
        print_status("CLIPモデル", False, f"{clip_model_path} が見つかりません")
        print("      以下からダウンロードしてください:")
        print("      https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K")
        print("      ファイル名: open_clip_model.safetensors")
        all_ok = False

    return all_ok


def check_directory_structure() -> bool:
    """ディレクトリ構造の確認"""
    print_header("ディレクトリ構造確認")

    all_ok = True

    # 必須ディレクトリ（存在しない場合はエラー）
    required_dirs = [
        ("data/train/like", "学習用: 好きな画像を配置"),
        ("data/train/dislike", "学習用: 嫌いな画像を配置"),
        ("data/unlabeled", "推論用: 分類/スコアリングしたい画像を配置"),
        ("models/clip", "CLIPモデルを配置"),
        ("models/trained", "学習済みモデルの出力先"),
    ]

    # オプションディレクトリ（推論実行時に自動生成される）
    optional_dirs = [
        ("data/classified/like", "classify.py出力: 好きと判定された画像"),
        ("data/classified/dislike", "classify.py出力: 嫌いと判定された画像"),
        ("data/scores", "score.py出力: スコアレポート(CSV)"),
    ]

    print("  [必須ディレクトリ]")
    for dir_path, description in required_dirs:
        path = Path(dir_path)
        if path.exists():
            # ディレクトリ内のファイル数をカウント（README.txtを除く）
            if path.is_dir():
                files = [f for f in path.glob("*") if f.name != "README.txt"]
                file_count = len(files)
                print_status(dir_path, True, f"{description} ({file_count} ファイル)")
            else:
                print_status(dir_path, True, description)
        else:
            print_status(dir_path, False, f"{description} - ディレクトリがありません")
            print(f"      mkdir -p {dir_path}")
            all_ok = False

    print("\n  [オプションディレクトリ（推論時に自動生成）]")
    for dir_path, description in optional_dirs:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                files = [f for f in path.glob("*") if f.name != "README.txt"]
                file_count = len(files)
                print_status(dir_path, True, f"{description} ({file_count} ファイル)")
            else:
                print_status(dir_path, True, description)
        else:
            print_status(dir_path, True, f"{description} - 未作成（自動生成されます）")

    return all_ok


def check_training_data() -> bool:
    """学習データの確認"""
    print_header("学習データ確認")

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

    like_dir = Path("data/train/like")
    dislike_dir = Path("data/train/dislike")

    # likeディレクトリの画像数
    like_count = 0
    if like_dir.exists():
        for ext in image_extensions:
            like_count += len(list(like_dir.glob(ext)))

    # dislikeディレクトリの画像数
    dislike_count = 0
    if dislike_dir.exists():
        for ext in image_extensions:
            dislike_count += len(list(dislike_dir.glob(ext)))

    total = like_count + dislike_count

    if like_count > 0:
        print_status("Like画像", True, f"{like_count} 枚")
    else:
        print_status("Like画像", False, "画像がありません")
        print("      data/train/like/ に好きな画像を配置してください")

    if dislike_count > 0:
        print_status("Dislike画像", True, f"{dislike_count} 枚")
    else:
        print_status("Dislike画像", False, "画像がありません")
        print("      data/train/dislike/ に嫌いな画像を配置してください")

    if total > 0:
        print(f"\n  合計: {total} 枚 (Like: {like_count}, Dislike: {dislike_count})")

        # データバランスの確認
        if like_count > 0 and dislike_count > 0:
            ratio = max(like_count, dislike_count) / min(like_count, dislike_count)
            if ratio > 3:
                print(f"  ⚠ データの偏りが大きいです (比率 {ratio:.1f}:1)")
                print("    バランスを取ることを推奨します")

    return like_count > 0 and dislike_count > 0


def check_unlabeled_data() -> bool:
    """推論用データの確認"""
    print_header("推論用データ確認")

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

    unlabeled_dir = Path("data/unlabeled")

    # unlabeledディレクトリの画像数
    unlabeled_count = 0
    if unlabeled_dir.exists():
        for ext in image_extensions:
            unlabeled_count += len(list(unlabeled_dir.glob(ext)))

    if unlabeled_count > 0:
        print_status("推論用画像", True, f"{unlabeled_count} 枚")
        print("      推論を実行できます:")
        print("        uv run python scripts/classify.py  (フォルダ分類)")
        print("        uv run python scripts/score.py     (スコアCSV出力)")
    else:
        print_status("推論用画像", True, "画像がありません（オプション）")
        print("      推論を行う場合は data/unlabeled/ に画像を配置してください")

    # 常にTrueを返す（オプションのため）
    return True


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("   img-score-ai 環境構築確認ツール")
    print("=" * 60)

    results = []

    # 各チェックを実行
    results.append(("Python バージョン", check_python_version()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("必要なライブラリ", check_libraries()))
    results.append(("モデルファイル", check_model_files()))
    results.append(("ディレクトリ構造", check_directory_structure()))
    results.append(("学習データ", check_training_data()))
    results.append(("推論用データ", check_unlabeled_data()))

    # 結果サマリー
    print_header("確認結果サマリー")

    all_passed = True
    for name, passed in results:
        print_status(name, passed)
        if not passed:
            all_passed = False

    print("\n" + "-" * 60)

    if all_passed:
        print("\n  🎉 環境構築が完了しています！")
        print("\n  次のステップ:")
        print("    1. 学習を開始する場合:")
        print("       uv run python scripts/train.py")
        print("    2. 推論を実行する場合（学習済みモデルが必要）:")
        print("       uv run python scripts/classify.py  (フォルダ分類)")
        print("       uv run python scripts/score.py     (スコアCSV出力)")
    else:
        print("\n  ⚠ 一部の項目が未完了です")
        print("    上記のエラーを確認し、必要な設定を行ってください")

    print("\n" + "=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
