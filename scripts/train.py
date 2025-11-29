"""
画像好み度予測モデルの学習スクリプト

このスクリプトは、CLIP (Contrastive Language-Image Pre-training) モデルを使用して
画像の「好み度」を予測する分類ヘッドを学習します。

【全体の流れ】
1. CLIP モデルで画像を512次元のベクトル（埋め込み）に変換
2. その埋め込みベクトルを入力として、好き/嫌いを予測する小さなニューラルネットワーク（PreferenceHead）を学習
3. 学習済みモデルを保存

【なぜCLIPを使うのか】
- CLIP は大量の画像とテキストで事前学習されており、画像の意味的な特徴を捉えることができる
- CLIP の特徴量を使うことで、少ないデータでも効果的に学習できる（転移学習）
- CLIP 自体は凍結（学習しない）し、軽量なヘッドのみを学習するため、学習が高速

【ストリーミング方式について】
- 埋め込みは計算しながらファイルに保存される（.pt ファイル）
- 処理が途中で止まっても、次回実行時にキャッシュから再開できる
- 大量の画像（数十万枚以上）でもメモリを圧迫しない
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import open_clip
from tqdm import tqdm
import json
import random
import numpy as np
from typing import Optional, Any


def set_seed(seed: int = 42) -> None:
    """
    乱数シードを固定して再現性を確保する

    【なぜシードを固定するのか】
    - 同じデータ・設定で実行した場合に同じ結果が得られる
    - デバッグや実験の比較が容易になる
    - 学習結果の再現性が重要な研究や本番環境で必須

    Args:
        seed (int): 乱数シード値。デフォルトは42。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 再現性のための設定（若干速度が低下する可能性あり）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_or_create_seed(seed_file: Path) -> int:
    """
    学習用の乱数シードを取得または生成する

    【動作】
    - シードファイルが存在する場合: 保存されたシードを読み込む
    - シードファイルが存在しない場合: 新しいシードをランダム生成して保存

    【なぜランダムに生成するのか】
    - 固定のシード（42など）だと、常に同じデータ分割になる
    - ランダムに生成することで、学習ごとに異なる分割になる
    - ただし、同じ学習セッション内では再現性を保つ

    【なぜファイルに保存するのか】
    - 学習が中断されて再開した場合、同じシードで再現できる
    - データのシャッフル順序が保持される

    Args:
        seed_file (Path): シードを保存するファイルのパス

    Returns:
        int: 乱数シード値
    """
    if seed_file.exists():
        # 既存のシードを読み込み
        with open(seed_file, "r") as f:
            data = json.load(f)
            seed = data["seed"]
            print(f"Loaded existing seed: {seed}")
            return seed
    else:
        # 新しいシードをランダム生成
        # os.urandom を使ってより良いエントロピーを得る
        # （Python の random は起動時刻に依存するため）
        seed = int.from_bytes(os.urandom(4), byteorder="big") % (2**31)

        # ディレクトリを作成して保存
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(seed_file, "w") as f:
            json.dump({"seed": seed}, f, indent=2)

        print(f"Generated new seed: {seed}")
        return seed


# ===== 設定 =====
# すべてのハイパーパラメータと設定を一箇所にまとめて管理
# これにより、実験時にパラメータを変更しやすくなる
CONFIG = {
    # ----- CLIP モデル設定 -----
    # ViT-B-16: Vision Transformer の Base サイズ、16x16 パッチ
    # 画像を 16x16 ピクセルのパッチに分割して処理する
    "clip_model": "ViT-B-16",
    # 事前学習済みの CLIP モデルの重みファイル
    "clip_checkpoint": Path("models/clip/open_clip_model.safetensors"),
    # ----- データパス -----
    # 学習データのディレクトリ構成:
    # data/train/
    #   ├── like/     <- 好きな画像を入れる
    #   └── dislike/  <- 嫌いな画像を入れる
    "data_dir": Path("data/train"),
    "like_dir": Path("data/train/like"),
    "dislike_dir": Path("data/train/dislike"),
    # ----- 学習設定 -----
    # batch_size: 一度に処理する画像の数
    # 大きいほど学習が安定するが、メモリを多く消費する
    "batch_size": 64,
    # embedding_batch_size: CLIP で埋め込みを計算する際のバッチサイズ
    # GPU メモリが足りない場合は小さくする（16, 8 など）
    "embedding_batch_size": 32,
    # epochs: データセット全体を何回繰り返し学習するか
    "epochs": 300,
    # learning_rate: 学習率（パラメータの更新幅）
    # 大きすぎると発散、小さすぎると収束が遅い
    "learning_rate": 1e-3,
    # weight_decay: L2正則化の強さ（過学習を防ぐ）
    "weight_decay": 1e-4,
    # val_split: バリデーション用に分けるデータの割合（20%）
    # 学習中にモデルの汎化性能を評価するために使用
    "val_split": 0.2,
    # ----- モデル保存 -----
    # 学習済みモデルの保存先ディレクトリ
    "save_dir": Path("models/trained"),
    "model_name": "preference_head_v1.pt",
    # ----- 埋め込みキャッシュ -----
    # 計算済みの埋め込みを保存するディレクトリ
    # 処理が途中で止まっても、次回実行時にキャッシュから再開できる
    "cache_dir": Path("data/cache"),
    # ----- 学習チェックポイント -----
    # 学習の途中状態を保存するファイル
    # 処理が途中で止まっても、次回実行時に途中から再開できる
    "checkpoint_path": Path("models/trained/checkpoint.pt"),
    # チェックポイントを保存する間隔（エポック数）
    "checkpoint_interval": 1,
    # ----- Early Stopping -----
    # バリデーション精度が改善しないエポックが続いた場合に学習を停止
    # 過学習を防ぎ、計算資源を節約する
    "early_stopping_patience": 15,
    # ----- シードファイル -----
    # 学習の乱数シードを保存するファイル
    # 初回実行時にランダム生成し、再開時は同じシードを使用
    "seed_file": Path("data/cache/seed.json"),
    # ----- ヘッド設計 -----
    # hidden_dim: 隠れ層のニューロン数
    # 512（CLIP出力）→ 256（隠れ層）→ 1（出力）という構造
    "hidden_dim": 256,
    # dropout: ドロップアウト率（過学習を防ぐためにランダムにニューロンを無効化）
    "dropout": 0.3,
}

# デバイスの自動選択
# CUDA（NVIDIA GPU）が使える場合は GPU を使用し、そうでなければ CPU を使用
# GPU を使うと学習が大幅に高速化される
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ===== 好み度予測ヘッド =====
class PreferenceHead(nn.Module):
    """
    画像の好み度を予測するニューラルネットワークヘッド

    【役割】
    CLIP が出力した 512 次元の埋め込みベクトルを入力として受け取り、
    その画像が「好き」か「嫌い」かを予測するスコア（ロジット）を出力する。

    【ネットワーク構造】
    入力 (512) → 線形変換 → ReLU活性化 → Dropout → 線形変換 → 出力 (1)

    - 512 次元: CLIP ViT-B-16 の出力次元
    - ReLU: 非線形性を導入（負の値を0にする活性化関数）
    - Dropout: 過学習を防ぐためにランダムにニューロンを無効化
    - 出力は1次元のロジット（確率に変換する前の値）

    【なぜロジットを出力するのか】
    - BCEWithLogitsLoss を使用するため、シグモイド関数は損失関数内で適用される
    - 数値的に安定した計算が可能になる

    Args:
        input_dim (int): 入力の次元数。デフォルトは512（CLIP ViT-B-16の出力）
        hidden_dim (int): 隠れ層の次元数。デフォルトは256
        dropout (float): ドロップアウト率。デフォルトは0.3（30%のニューロンを無効化）
    """

    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3):
        """
        PreferenceHead の初期化

        ネットワークの各層を定義する。
        nn.Sequential を使って複数の層を順番に並べる。
        """
        super().__init__()

        # nn.Sequential: 複数の層を順番に適用するコンテナ
        self.net = nn.Sequential(
            # 第1層: 線形変換（全結合層）
            # 512次元の入力を256次元に圧縮
            # 重みとバイアスが学習される
            nn.Linear(input_dim, hidden_dim),  # 512 → 256
            # 活性化関数: ReLU (Rectified Linear Unit)
            # f(x) = max(0, x)
            # 負の値を0にすることで非線形性を導入
            nn.ReLU(),
            # Dropout: 過学習を防ぐ正則化手法
            # 学習時にランダムに一部のニューロンを無効化（0にする）
            # これにより、特定のニューロンへの依存を減らし、汎化性能を向上
            nn.Dropout(dropout),
            # 第2層: 出力層
            # 256次元を1次元（スカラー）に変換
            # この値が「好き」の度合いを表すロジット
            nn.Linear(hidden_dim, 1),  # 256 → 1 (ロジット)
        )

    def forward(self, x):
        """
        順伝播（フォワードパス）

        入力テンソルをネットワークに通して予測を得る。

        Args:
            x (torch.Tensor): CLIP埋め込み。形状は (batch_size, 512)

        Returns:
            torch.Tensor: 予測ロジット。形状は (batch_size,)
                         正の値: 「好き」の可能性が高い
                         負の値: 「嫌い」の可能性が高い

        Note:
            squeeze(-1) で最後の次元を削除: (B, 1) → (B,)
            これにより、損失関数やメトリクス計算時に形状が合う
        """
        return self.net(x).squeeze(-1)  # (B, 1) → (B,)


# ===== データセット =====
class ImageEmbeddingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    PyTorch Dataset クラス: CLIP埋め込みとラベルのペアを管理

    【役割】
    - 事前に計算された CLIP 埋め込みとそのラベル（好き/嫌い）を保持
    - DataLoader と連携してバッチ単位でデータを提供

    【なぜ Dataset クラスを使うのか】
    - PyTorch の DataLoader と連携し、自動的にバッチ化・シャッフルができる
    - メモリ効率の良いデータ読み込みが可能
    - マルチプロセスでのデータ読み込みをサポート

    【注意】
    このデータセットは画像そのものではなく、事前計算された埋め込みを保持する。
    これにより、学習時に毎回 CLIP で埋め込みを計算する必要がなく、高速化できる。

    Attributes:
        embeddings (torch.Tensor): CLIP埋め込みテンソル。形状は (N, 512)
        labels (torch.Tensor): ラベルテンソル。形状は (N,)
                              1.0 = 好き, 0.0 = 嫌い
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """
        データセットの初期化

        Args:
            embeddings (torch.Tensor): CLIP で計算した画像埋め込み (N, 512)
            labels (torch.Tensor): 各画像のラベル (N,)
                                  1.0 = 好き (like)
                                  0.0 = 嫌い (dislike)
        """
        self.embeddings = embeddings  # (N, 512)
        self.labels = labels  # (N,)

    def __len__(self) -> int:
        """
        データセットのサンプル数を返す

        DataLoader がデータセットの大きさを知るために使用。

        Returns:
            int: データセット内のサンプル数
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのサンプルを返す

        DataLoader がバッチを作成する際に呼び出される。

        Args:
            idx (int): 取得するサンプルのインデックス

        Returns:
            tuple: (埋め込みベクトル, ラベル) のタプル
                   - 埋め込み: torch.Tensor, 形状 (512,)
                   - ラベル: torch.Tensor, スカラー (0.0 または 1.0)
        """
        return self.embeddings[idx], self.labels[idx]


# ===== 埋め込み計算 =====
def get_cache_key(image_path: Path) -> str:
    """
    画像パスからキャッシュキーを生成する

    【キャッシュの仕組み】
    - 画像の絶対パスをそのままキーとして使用
    - これにより、同じ画像は常に同じキーでアクセスされる

    Args:
        image_path (Path): 画像ファイルのパス

    Returns:
        str: キャッシュキー（画像の絶対パス文字列）
    """
    # 画像の絶対パスを文字列化してキャッシュキーとして使用
    return str(image_path.absolute())


# ===== キャッシュ管理クラス =====
class EmbeddingCache:
    """
    埋め込みキャッシュを1つのファイルで管理するクラス

    【設計思想】
    - 1つの .pt ファイルに辞書形式で全埋め込みを保存
    - バッチ処理ごとにファイルを更新（ストリーミング保存）
    - 途中で止まっても再開可能
    - 画像パス（絶対パス文字列）をキーとして使用

    【ファイル構造】
    {
        "D:/path/to/image1.jpg": tensor([...]),  # 512次元
        "D:/path/to/image2.png": tensor([...]),
        ...
    }
    """

    def __init__(self, cache_path: Path):
        """
        キャッシュの初期化

        Args:
            cache_path (Path): キャッシュファイルのパス（.pt ファイル）
        """
        self.cache_path = cache_path
        self.cache: dict[str, torch.Tensor] = {}
        self._load()

    def _load(self) -> None:
        """キャッシュファイルを読み込む"""
        if self.cache_path.exists():
            try:
                self.cache = torch.load(self.cache_path, weights_only=False)
                print(f"Loaded cache with {len(self.cache)} embeddings")
            except Exception as e:
                print(f"Cache corrupted, starting fresh: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def save(self) -> None:
        """キャッシュをファイルに保存（アトミック）"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(".tmp")
        try:
            torch.save(self.cache, temp_path)
            temp_path.replace(self.cache_path)
        except Exception as e:
            print(f"Error saving cache: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def get(self, key: str) -> torch.Tensor | None:
        """キーに対応する埋め込みを取得"""
        return self.cache.get(key)

    def set(self, key: str, embedding: torch.Tensor) -> None:
        """埋め込みをキャッシュに追加（メモリ上のみ）"""
        self.cache[key] = embedding

    def __contains__(self, key: str) -> bool:
        """キーがキャッシュに存在するか確認"""
        return key in self.cache

    def __len__(self) -> int:
        """キャッシュ内のエントリ数"""
        return len(self.cache)


def compute_single_embedding(
    image_path: Path, clip_model, preprocess, device
) -> torch.Tensor:
    """
    単一の画像から CLIP 埋め込みを計算する

    Args:
        image_path (Path): 画像ファイルのパス
        clip_model: ロード済みの CLIP モデル
        preprocess: CLIP の画像前処理関数
        device (str): 計算デバイス

    Returns:
        torch.Tensor: L2正規化された埋め込みベクトル (512,)

    Raises:
        Exception: 画像の読み込みや処理に失敗した場合
    """
    # PIL で画像を開き、RGB に変換
    img = Image.open(image_path).convert("RGB")

    # CLIP の前処理を適用してバッチ次元を追加
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # CLIP で埋め込みを計算
    with torch.no_grad():
        features = clip_model.encode_image(img_tensor)
        # L2 正規化
        features = features / features.norm(dim=-1, keepdim=True)

    # (1, 512) → (512,) に変換して CPU に移動
    return features.squeeze(0).cpu()


def compute_embeddings_streaming(
    image_paths: list,
    clip_model,
    preprocess,
    device: str,
    cache_dir: Path,
    batch_size: int = 32,
    save_interval: int = 5,
) -> torch.Tensor:
    """
    画像リストから CLIP 埋め込みをストリーミング方式で計算する関数

    【ストリーミング方式の利点】
    1. 途中で止まっても再開可能: 計算済みの埋め込みは1つのキャッシュファイルに保存
    2. メモリ効率: バッチごとに処理
    3. 進捗の可視化: どこまで処理が終わったかが明確

    【キャッシュファイル】
    - cache_dir/embeddings.pt に辞書形式で保存
    - キー: 画像の絶対パス（文字列）
    - 値: 512次元の埋め込みテンソル

    【処理の流れ】
    1. キャッシュファイルを読み込む
    2. 各画像についてキャッシュを確認
    3. キャッシュがあればそれを使用
    4. キャッシュがなければバッチ計算してキャッシュに追加
    5. 一定間隔でキャッシュをファイルに保存
    6. 最後にすべての埋め込みを連結して返す

    Args:
        image_paths (list): 画像ファイルのパスリスト
        clip_model: ロード済みの CLIP モデル
        preprocess: CLIP の画像前処理関数
        device (str): 計算デバイス ("cuda" または "cpu")
        cache_dir (Path): キャッシュディレクトリのパス
        batch_size (int): バッチサイズ。デフォルトは32。
        save_interval (int): キャッシュを保存する間隔（バッチ数）。デフォルトは5。

    Returns:
        torch.Tensor: すべての画像の埋め込みを連結したテンソル (N, 512)

    Note:
        キャッシュを削除したい場合は cache_dir/embeddings.pt を削除してください。
    """
    # キャッシュを初期化
    cache_path = cache_dir / "embeddings.pt"
    cache = EmbeddingCache(cache_path)

    embeddings = []
    cache_hits = 0
    cache_misses = 0

    # 推論モードに設定
    clip_model.eval()

    # ----- 画像を処理 -----
    # キャッシュミスした画像を一時的に保持するバッファ
    batch_paths = []
    batch_indices = []  # 元のインデックスを記録（順序を保持するため）

    print(f"Processing {len(image_paths)} images (cache: {cache_path})")

    # まず、キャッシュの状態を確認
    for idx, img_path in enumerate(tqdm(image_paths, desc="Checking cache")):
        cache_key = get_cache_key(img_path)

        if cache_key in cache:
            # キャッシュヒット: 保存済みの埋め込みを使用
            embedding = cache.get(cache_key)
            embeddings.append((idx, embedding))
            cache_hits += 1
        else:
            # キャッシュミス: 後でバッチ処理する
            batch_paths.append(img_path)
            batch_indices.append(idx)
            cache_misses += 1

    print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")

    # ----- キャッシュミスした画像をバッチ処理 -----
    if batch_paths:
        print(f"Computing embeddings for {len(batch_paths)} uncached images...")
        batches_since_save = 0

        with torch.no_grad():
            for i in tqdm(
                range(0, len(batch_paths), batch_size), desc="Computing embeddings"
            ):
                # このバッチで処理する画像
                current_batch_paths = batch_paths[i : i + batch_size]
                current_batch_indices = batch_indices[i : i + batch_size]
                batch_tensors = []
                valid_paths = []
                valid_indices = []

                # 画像を読み込んで前処理
                for img_path, idx in zip(current_batch_paths, current_batch_indices):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_tensor = preprocess(img)
                        batch_tensors.append(img_tensor)
                        valid_paths.append(img_path)
                        valid_indices.append(idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

                if not batch_tensors:
                    continue

                # バッチテンソルを作成して GPU に転送
                batch_tensor = torch.stack(batch_tensors).to(device)

                # CLIP で埋め込みを計算
                features = clip_model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu()

                # 各画像の埋め込みをキャッシュに追加
                for j, (img_path, idx) in enumerate(zip(valid_paths, valid_indices)):
                    embedding = features[j]
                    cache_key = get_cache_key(img_path)
                    cache.set(cache_key, embedding)
                    embeddings.append((idx, embedding))

                # GPU メモリを解放
                del batch_tensor, features
                if device == "cuda":
                    torch.cuda.empty_cache()

                # 一定間隔でキャッシュを保存
                batches_since_save += 1
                if batches_since_save >= save_interval:
                    cache.save()
                    batches_since_save = 0

        # 最後に残りを保存
        cache.save()

    # ----- 埋め込みが空の場合のエラーハンドリング -----
    if not embeddings:
        raise ValueError("No valid images found. Please check the image directory.")

    # ----- インデックス順にソートして連結 -----
    # バッチ処理とキャッシュ読み込みで順序がバラバラになっているため、
    # 元の順序に戻す
    embeddings.sort(key=lambda x: x[0])
    embeddings_tensor = torch.stack([emb for _, emb in embeddings])

    print(f"Total embeddings: {len(embeddings_tensor)}")
    return embeddings_tensor


def compute_embeddings(
    image_paths: list[Path],
    clip_model: Any,
    preprocess: Any,
    device: str,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    後方互換性のためのラッパー関数

    ストリーミング方式の compute_embeddings_streaming を呼び出す。
    キャッシュディレクトリは CONFIG から取得する。
    """
    return compute_embeddings_streaming(
        image_paths=image_paths,
        clip_model=clip_model,
        preprocess=preprocess,
        device=device,
        cache_dir=CONFIG["cache_dir"],
        batch_size=batch_size,
    )


# ===== データ準備 =====
def prepare_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    like / dislike ディレクトリから埋め込み＋ラベルを作成する関数

    【処理の流れ】
    1. CLIP モデルをロード
    2. like / dislike ディレクトリから画像パスを収集
    3. 各画像の CLIP 埋め込みを計算
    4. ラベルを作成（like=1, dislike=0）
    5. データをシャッフル
    6. 訓練データとバリデーションデータに分割

    【なぜシャッフルするのか】
    - 元々 like と dislike が別々のブロックになっているため
    - シャッフルしないと、学習の最初は全部 like、後半は全部 dislike となる
    - これを防ぐことで、学習が安定する

    【なぜ訓練/バリデーションを分けるのか】
    - 訓練データ: モデルのパラメータを更新するために使用
    - バリデーションデータ: 学習中にモデルの汎化性能を評価するために使用
    - 同じデータで評価すると、過学習を検出できない

    Returns:
        tuple: 4つのテンソルを含むタプル
            - train_embeddings (torch.Tensor): 訓練用埋め込み (N_train, 512)
            - train_labels (torch.Tensor): 訓練用ラベル (N_train,)
            - val_embeddings (torch.Tensor): バリデーション用埋め込み (N_val, 512)
            - val_labels (torch.Tensor): バリデーション用ラベル (N_val,)
    """
    # ----- CLIP モデルのロード -----
    print("Loading CLIP model...")

    # open_clip.create_model_and_transforms:
    # - model: CLIP モデル本体
    # - preprocess: 画像の前処理関数
    # - _: tokenizer（テキスト用、今回は使わない）
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        model_name=CONFIG["clip_model"],  # "ViT-B-16"
        pretrained=str(CONFIG["clip_checkpoint"]),  # 事前学習済み重み
    )
    # モデルを指定デバイス（GPU/CPU）に移動
    clip_model = clip_model.to(device)

    # ----- 画像パス収集 -----
    # 複数の拡張子に対応（jpg, jpeg, png, webp）
    # 大文字・小文字両方に対応するため、複数のパターンを使用
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

    like_paths = []
    for ext in image_extensions:
        like_paths.extend(CONFIG["like_dir"].glob(ext))
    like_paths = sorted(like_paths)  # 順序を安定させる

    dislike_paths = []
    for ext in image_extensions:
        dislike_paths.extend(CONFIG["dislike_dir"].glob(ext))
    dislike_paths = sorted(dislike_paths)  # 順序を安定させる

    print(f"Found {len(like_paths)} like images, {len(dislike_paths)} dislike images")

    # データ不均衡の警告
    if len(like_paths) > 0 and len(dislike_paths) > 0:
        ratio = max(len(like_paths), len(dislike_paths)) / min(
            len(like_paths), len(dislike_paths)
        )
        if ratio > 3:
            print(
                f"Warning: Data imbalance detected (ratio: {ratio:.1f}:1). "
                f"Consider balancing the dataset for better results."
            )

    # データが空の場合はエラー
    if len(like_paths) == 0:
        raise ValueError(
            f"No images found in {CONFIG['like_dir']}. Please add images to train."
        )
    if len(dislike_paths) == 0:
        raise ValueError(
            f"No images found in {CONFIG['dislike_dir']}. Please add images to train."
        )

    # ----- 埋め込み計算 -----
    # like 画像の埋め込みを計算（バッチ処理で効率化）
    print("Computing embeddings for LIKE images...")
    like_embeddings = compute_embeddings(
        like_paths,
        clip_model,
        preprocess,
        device,
        batch_size=CONFIG["embedding_batch_size"],
    )

    # dislike 画像の埋め込みを計算（バッチ処理で効率化）
    print("Computing embeddings for DISLIKE images...")
    dislike_embeddings = compute_embeddings(
        dislike_paths,
        clip_model,
        preprocess,
        device,
        batch_size=CONFIG["embedding_batch_size"],
    )

    # ----- ラベル作成 -----
    # like = 1.0 (正例)
    # dislike = 0.0 (負例)
    # ones/zeros でテンソルを作成
    like_labels = torch.ones(len(like_embeddings))  # すべて 1.0
    dislike_labels = torch.zeros(len(dislike_embeddings))  # すべて 0.0

    # ----- データの結合 -----
    # like と dislike のデータを1つのテンソルに連結
    all_embeddings = torch.cat([like_embeddings, dislike_embeddings], dim=0)
    all_labels = torch.cat([like_labels, dislike_labels], dim=0)

    # ----- シャッフル -----
    # ランダムな順列を生成してデータを並び替え
    # これにより、like と dislike が混在した状態になる
    perm = torch.randperm(len(all_labels))  # [0, 1, 2, ...] をランダムに並び替え
    all_embeddings = all_embeddings[perm]
    all_labels = all_labels[perm]

    # ----- Train / Val 分割 -----
    # バリデーションデータのサイズを計算
    val_size = int(len(all_labels) * CONFIG["val_split"])  # 全体の20%

    # バリデーションデータが少なすぎる場合は最低1件確保
    if val_size == 0 and len(all_labels) > 1:
        val_size = 1

    # 先頭 val_size 個をバリデーション、残りを訓練データに
    train_embeddings = all_embeddings[val_size:]
    train_labels = all_labels[val_size:]
    val_embeddings = all_embeddings[:val_size]
    val_labels = all_labels[:val_size]

    # データが少なすぎる場合の警告
    if len(train_labels) == 0:
        raise ValueError(
            f"Not enough data for training. Total: {len(all_labels)}, Val: {val_size}"
        )
    if len(val_labels) == 0:
        print("Warning: No validation data. Consider adding more images.")

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}")

    # ----- CLIP モデルのメモリ解放 -----
    # 埋め込み計算が終わったので、CLIP モデルは不要
    # GPU メモリを解放して学習用に確保
    del clip_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return train_embeddings, train_labels, val_embeddings, val_labels


# ===== チェックポイント管理 =====
def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    best_val_acc: float,
    history: dict,
    checkpoint_path: Path,
    epochs_without_improvement: int = 0,
):
    """
    学習の途中状態をチェックポイントとして保存する

    【チェックポイントに保存される情報】
    - epoch: 現在のエポック番号
    - model_state_dict: モデルの重み
    - optimizer_state_dict: 最適化器の状態（モーメンタムなど）
    - scheduler_state_dict: 学習率スケジューラーの状態
    - best_val_acc: これまでの最高バリデーション精度
    - history: 学習履歴（損失、精度）
    - epochs_without_improvement: Early Stopping カウンター

    【なぜ最適化器の状態も保存するのか】
    - Adam などの最適化器は内部状態（モーメンタム）を持つ
    - 再開時に状態を復元しないと、学習が不安定になる可能性がある

    Args:
        epoch (int): 現在のエポック番号
        model (nn.Module): 学習中のモデル
        optimizer (optim.Optimizer): 最適化器
        scheduler (optim.lr_scheduler.LRScheduler): 学習率スケジューラー
        best_val_acc (float): これまでの最高バリデーション精度
        history (dict): 学習履歴
        checkpoint_path (Path): チェックポイントの保存先
        epochs_without_improvement (int): Early Stopping カウンター
    """
    # 保存ディレクトリを作成
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # チェックポイントデータを作成
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "epochs_without_improvement": epochs_without_improvement,
        "config": {
            "hidden_dim": CONFIG["hidden_dim"],
            "dropout": CONFIG["dropout"],
        },
    }

    # アトミックに保存（一時ファイル→リネーム）
    temp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    # Windows では rename() が既存ファイルを上書きできないため replace() を使用
    temp_path.replace(checkpoint_path)

    print(f"  → Checkpoint saved at epoch {epoch + 1}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> Optional[tuple[int, float, dict, int]]:
    """
    チェックポイントから学習状態を復元する

    Args:
        checkpoint_path (Path): チェックポイントのパス
        model (nn.Module): モデル（状態が上書きされる）
        optimizer (optim.Optimizer): 最適化器（状態が上書きされる）
        scheduler (optim.lr_scheduler.LRScheduler): 学習率スケジューラー（状態が上書きされる）

    Returns:
        Optional[tuple]: (start_epoch, best_val_acc, history, epochs_without_improvement)
            - start_epoch: 再開するエポック番号
            - best_val_acc: これまでの最高バリデーション精度
            - history: 学習履歴
            - epochs_without_improvement: Early Stopping カウンター
            チェックポイントが存在しない場合は None を返す
    """
    if not checkpoint_path.exists():
        return None

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # モデルと最適化器の状態を復元
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # 後方互換性: 古いチェックポイントには scheduler_state_dict がない場合がある
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1  # 次のエポックから再開
    best_val_acc = checkpoint["best_val_acc"]
    history = checkpoint["history"]
    # 後方互換性: 古いチェックポイントには epochs_without_improvement がない場合がある
    epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

    print(f"Resuming from epoch {start_epoch + 1}, best_val_acc: {best_val_acc:.4f}")

    return start_epoch, best_val_acc, history, epochs_without_improvement


def clear_checkpoint(checkpoint_path: Path) -> None:
    """
    チェックポイントファイルを削除する

    学習が正常に完了した後に呼び出す。
    次回の学習に影響しないようにする。

    Args:
        checkpoint_path (Path): チェックポイントのパス
    """
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint cleared: {checkpoint_path}")


# ===== キャッシュ管理 =====
def clear_cache(cache_dir: Path) -> None:
    """
    埋め込みキャッシュファイルを削除する

    学習が正常に完了した後に呼び出す。
    次回の学習で古いキャッシュが使われないようにする。

    【なぜキャッシュをクリアするのか】
    - 画像が変更された場合、古いキャッシュが使われてしまう
    - ディスク容量を節約する
    - 次回の学習で確実に最新の埋め込みを使う

    Args:
        cache_dir (Path): キャッシュディレクトリのパス
    """
    cache_file = cache_dir / "embeddings.pt"
    if cache_file.exists():
        cache_file.unlink()
        print(f"Cache cleared: {cache_file}")
    # 一時ファイルも削除
    temp_file = cache_dir / "embeddings.tmp"
    if temp_file.exists():
        temp_file.unlink()


# ===== 学習 =====
def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    epochs: int,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
    history: Optional[dict] = None,
    patience: int = 10,
    epochs_without_improvement: int = 0,
) -> tuple[dict, float]:
    """
    モデルの学習を実行するメイン関数

    【学習の全体像】
    各エポックで以下を繰り返す:
    1. 訓練フェーズ: 訓練データでモデルを更新
    2. 検証フェーズ: バリデーションデータで性能を評価
    3. ベストモデルを保存
    4. チェックポイントを保存（中断からの再開用）

    【学習の仕組み（各バッチ）】
    1. 順伝播: 入力データをモデルに通して予測を得る
    2. 損失計算: 予測とラベルの差を計算
    3. 逆伝播: 損失から各パラメータの勾配を計算
    4. パラメータ更新: 勾配に基づいてパラメータを調整

    【なぜバリデーションが必要か】
    - 訓練データでの性能は過学習を反映しない
    - 未知のデータ（バリデーション）での性能が真の汎化性能
    - バリデーション精度でベストモデルを選択

    【チェックポイント機能】
    - 各エポック終了時にチェックポイントを保存
    - 学習が中断しても、次回実行時に途中から再開可能
    - 学習完了後にチェックポイントは削除される

    Args:
        train_loader (DataLoader): 訓練データのローダー
        val_loader (DataLoader): バリデーションデータのローダー
        model (nn.Module): 学習するモデル（PreferenceHead）
        criterion: 損失関数（BCEWithLogitsLoss）
        optimizer: 最適化アルゴリズム（AdamW）
        scheduler: 学習率スケジューラー（CosineAnnealingLR）
        epochs (int): 学習エポック数
        start_epoch (int): 開始エポック（再開時に使用）
        best_val_acc (float): これまでの最高バリデーション精度（再開時に使用）
        history (dict): 学習履歴（再開時に使用）

    Returns:
        dict: 学習履歴を含む辞書
            - train_loss: 各エポックの訓練損失リスト
            - val_loss: 各エポックのバリデーション損失リスト
            - val_acc: 各エポックのバリデーション精度リスト
    """
    # 学習履歴を初期化または再開時の履歴を使用
    if history is None:
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "learning_rate": []}
    # 後方互換性: 古い履歴には learning_rate がない場合がある
    if "learning_rate" not in history:
        history["learning_rate"] = []

    # Early Stopping 用のカウンター
    # バリデーション精度が改善しないエポックが続いた回数を記録
    # （再開時は引数から復元される）

    # ----- エポックループ -----
    # 1エポック = データセット全体を1回学習
    for epoch in range(start_epoch, epochs):
        # ========== Training Phase ==========
        # モデルを訓練モードに設定
        # - Dropout が有効になる
        # - BatchNorm が訓練時の動作になる
        model.train()
        train_loss = 0.0

        # 訓練データのバッチを順番に処理
        for embeddings, labels in train_loader:
            # ----- データをデバイスに転送 -----
            # GPU を使う場合、データも GPU に移動する必要がある
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # ----- 勾配をリセット -----
            # PyTorch は勾配を累積するため、各バッチの前にリセットが必要
            optimizer.zero_grad()

            # ----- 順伝播 (Forward Pass) -----
            # モデルに入力を通して予測（ロジット）を得る
            logits = model(embeddings)

            # ----- 損失計算 -----
            # BCEWithLogitsLoss: Binary Cross Entropy + Sigmoid
            # 二値分類の標準的な損失関数
            loss = criterion(logits, labels)

            # ----- 逆伝播 (Backward Pass) -----
            # 損失から各パラメータの勾配を計算
            # 自動微分により、計算グラフを辿って勾配を計算
            loss.backward()

            # ----- パラメータ更新 -----
            # 計算された勾配に基づいてパラメータを更新
            # AdamW: 学習率とモーメンタムを適応的に調整
            optimizer.step()

            # 損失を累積（後で平均を計算するため）
            train_loss += loss.item()

        # エポックの平均訓練損失を計算
        train_loss /= len(train_loader)

        # 現在の学習率を取得（進捗表示用）
        current_lr = optimizer.param_groups[0]["lr"]

        # ========== Validation Phase ==========
        # バリデーションデータがない場合はスキップ
        if len(val_loader) == 0:
            val_loss = 0.0
            val_acc = 0.0
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val: N/A (no validation data)"
            )
        else:
            # モデルを評価モードに設定
            # - Dropout が無効になる（すべてのニューロンを使用）
            # - BatchNorm が推論時の動作になる
            model.eval()
            val_loss = 0.0
            correct = 0  # 正解数
            total = 0  # 総サンプル数

            # 勾配計算を無効化（メモリ節約と高速化）
            # バリデーション時はパラメータを更新しないため不要
            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings = embeddings.to(device)
                    labels = labels.to(device)

                    # 順伝播のみ（逆伝播なし）
                    logits = model(embeddings)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    # ----- 精度計算 -----
                    # シグモイド関数でロジットを確率 [0, 1] に変換
                    probs = torch.sigmoid(logits)

                    # 確率 > 0.5 なら予測 = 1（好き）、そうでなければ 0（嫌い）
                    preds = (probs > 0.5).float()

                    # 予測がラベルと一致した数をカウント
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            # エポックの平均バリデーション損失と精度を計算
            val_loss /= len(val_loader)
            val_acc = correct / total  # 正解率 = 正解数 / 総数

            # ----- 進捗表示 -----
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}"
            )

        # ----- 履歴に記録 -----
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)

        # ----- ベストモデル保存 -----
        # バリデーション精度が過去最高を更新したら保存
        # バリデーションデータがない場合は最後のエポックのモデルを保存（Early Stoppingは無効）
        has_validation = len(val_loader) > 0

        if has_validation:
            # バリデーションデータがある場合: 精度が改善したら保存
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                CONFIG["save_dir"].mkdir(exist_ok=True)
                torch.save(
                    model.state_dict(), CONFIG["save_dir"] / CONFIG["model_name"]
                )
                print(f"  → Best model saved (acc: {val_acc:.4f})")
            else:
                epochs_without_improvement += 1
        else:
            # バリデーションデータがない場合: 毎エポック上書き保存
            # Early Stopping は無効（epochs_without_improvement を更新しない）
            CONFIG["save_dir"].mkdir(exist_ok=True)
            torch.save(model.state_dict(), CONFIG["save_dir"] / CONFIG["model_name"])

        # ----- 学習率スケジューラーの更新 -----
        # エポック終了時に学習率を更新（コサイン関数に従って減衰）
        scheduler.step()

        # ----- Early Stopping チェック -----
        # バリデーションデータがある場合のみ Early Stopping を適用
        if has_validation and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"No improvement in validation accuracy for {patience} epochs")
            break

        # ----- チェックポイント保存 -----
        # 指定間隔でチェックポイントを保存（中断からの再開用）
        if (epoch + 1) % CONFIG["checkpoint_interval"] == 0:
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_acc=best_val_acc,
                history=history,
                checkpoint_path=CONFIG["checkpoint_path"],
                epochs_without_improvement=epochs_without_improvement,
            )

    return history, best_val_acc


# ===== メイン =====
def main() -> None:
    """
    学習のメインエントリーポイント

    【処理の流れ】
    1. データ準備: CLIP埋め込みの計算とデータセット分割
    2. DataLoader の作成: バッチ処理のためのラッパー
    3. モデルの初期化: PreferenceHead の作成
    4. 損失関数と最適化器の設定
    5. 学習ループの実行
    6. 学習履歴の保存

    【DataLoader の役割】
    - バッチ化: 複数のサンプルをまとめて処理
    - シャッフル: 各エポックでデータの順序をランダムに
    - マルチプロセス: 複数のワーカーで並列データ読み込み

    【損失関数: BCEWithLogitsLoss】
    Binary Cross Entropy with Logits Loss
    - 二値分類の標準的な損失関数
    - シグモイド + BCE を内部で計算（数値的に安定）
    - 予測と正解の「ズレ」を測定

    【最適化器: AdamW】
    - Adam: Adaptive Moment Estimation（適応的学習率）
    - W: Weight Decay（重み減衰、L2正則化を正しく適用）
    - 各パラメータごとに学習率を調整するため、収束が速い
    """
    # ===== 0. 乱数シードの設定 =====
    # 初回実行時: ランダムなシードを生成してファイルに保存
    # 再開時: 保存されたシードを使用（データ分割の再現性のため）
    seed = get_or_create_seed(CONFIG["seed_file"])
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # ===== 1. データ準備 =====
    # CLIP 埋め込みを計算し、訓練/バリデーションに分割
    train_emb, train_labels, val_emb, val_labels = prepare_data()

    # ===== 2. データセットと DataLoader の作成 =====
    # Dataset: 個々のサンプルへのアクセスを提供
    train_dataset = ImageEmbeddingDataset(train_emb, train_labels)
    val_dataset = ImageEmbeddingDataset(val_emb, val_labels)

    # DataLoader: バッチ処理とシャッフルを担当
    # - batch_size: 一度に処理するサンプル数
    # - shuffle=True: 各エポックでデータをシャッフル（訓練時のみ）
    # - shuffle=False: バリデーション時は一貫性のためシャッフルしない
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ===== 3. モデルの初期化 =====
    # PreferenceHead: CLIP 埋め込みから好み度を予測するネットワーク
    # .to(device) でモデルを GPU/CPU に移動
    model = PreferenceHead(
        input_dim=512,  # CLIP ViT-B-16 の出力次元
        hidden_dim=CONFIG["hidden_dim"],  # 隠れ層のサイズ
        dropout=CONFIG["dropout"],  # ドロップアウト率
    ).to(device)

    # モデルのパラメータ数を表示
    # 小さなモデルなので、学習が高速
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===== 4. 損失関数と最適化器の設定 =====
    # BCEWithLogitsLoss: 二値分類の損失関数
    # - Binary Cross Entropy + Sigmoid を内部で計算
    # - 数値的に安定した実装
    # - 予測ロジットと正解ラベル（0 or 1）から損失を計算
    criterion = nn.BCEWithLogitsLoss()

    # AdamW: 最適化アルゴリズム
    # - lr (learning_rate): パラメータ更新の基本的な大きさ
    # - weight_decay: L2正則化の強さ（過学習を防ぐ）
    optimizer = optim.AdamW(
        model.parameters(),  # 最適化するパラメータ
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # CosineAnnealingLR: 学習率スケジューラー
    # - 学習率をコサイン関数に従って徐々に減衰させる
    # - T_max: 半周期のエポック数（全エポック数を指定すると最後に最小値に）
    # - eta_min: 最小学習率（デフォルトは0）
    # 【なぜ CosineAnnealingLR を使うのか】
    # - 学習初期は大きな学習率で大まかに最適化
    # - 学習終盤は小さな学習率で微調整
    # - 滑らかな減衰で学習が安定する
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"],  # 全エポック数で1周期
        eta_min=1e-6,  # 最小学習率
    )

    # ===== 5. チェックポイントからの再開確認 =====
    # 前回の学習が中断された場合、チェックポイントから再開
    start_epoch = 0
    best_val_acc = 0.0
    history = None
    epochs_without_improvement = 0

    checkpoint_data = load_checkpoint(
        checkpoint_path=CONFIG["checkpoint_path"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if checkpoint_data is not None:
        start_epoch, best_val_acc, history, epochs_without_improvement = checkpoint_data
        print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        print("Starting fresh training")

    # ===== 6. 学習ループの実行 =====
    # train_model 関数で実際の学習を行う
    history, best_val_acc = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=CONFIG["epochs"],
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
        history=history,
        patience=CONFIG["early_stopping_patience"],
        epochs_without_improvement=epochs_without_improvement,
    )

    # ===== 7. 学習履歴の保存 =====
    # JSON 形式で履歴を保存（後で可視化や分析に使用）
    with open(CONFIG["save_dir"] / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ===== 8. キャッシュとチェックポイントのクリア =====
    # 学習が正常に完了したので、一時ファイルを削除
    # これにより、次回の学習に影響しないようにする
    print("\nCleaning up...")
    clear_checkpoint(CONFIG["checkpoint_path"])
    clear_cache(CONFIG["cache_dir"])

    # 完了メッセージ
    print("\nTraining complete!")
    print(f"Best model saved to: {CONFIG['save_dir'] / CONFIG['model_name']}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


# ===== エントリーポイント =====
# このスクリプトが直接実行された場合のみ main() を呼び出す
# 他のスクリプトからインポートされた場合は実行されない
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoint has been saved. Run again to resume.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("Checkpoint has been saved. Fix the error and run again to resume.")
        raise
