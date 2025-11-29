このフォルダには学習済みモデルが出力されます。

【内容】
train.py で学習したモデルファイルが保存されます。

【生成元スクリプト】
  uv run python scripts/train.py

【出力ファイル】
- preference_head_v1.pt  : 学習済みモデル（推論に使用）
- checkpoint.pt          : チェックポイント（学習再開用）
- history.json           : 学習履歴（損失・精度の推移）

【使用方法】
学習完了後、以下のスクリプトで推論に使用できます：
  uv run python scripts/classify.py
  uv run python scripts/score.py

【注意】
- このフォルダの内容は train.py 実行時に自動生成されます
- 手動でファイルを配置する必要はありません
