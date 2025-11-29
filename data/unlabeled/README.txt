このフォルダには分類・スコアリングしたい画像を配置してください。

【用途】
推論（分類・スコアリング）の入力として使用されます。
学習済みモデルを使って、ここに配置した画像を評価します。

【対応フォーマット】
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

【推論スクリプト】

① 画像分類（like/dislike フォルダに振り分け）
  uv run python scripts/classify.py

② スコアレポート出力（CSV形式）
  uv run python scripts/score.py

【出力先】
- classify.py → data/classified/like/ または data/classified/dislike/
- score.py    → data/scores/scores.csv
