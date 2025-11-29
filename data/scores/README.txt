このフォルダは score.py の出力先です。

【内容】
画像のスコアレポート（CSV形式）がここに出力されます。

【生成元スクリプト】
  uv run python scripts/score.py

【出力ファイル例】
  scores.csv
    filename,score,label
    image001.jpg,87.5,like
    image002.png,23.1,dislike
    ...

【CSVの項目】
- filename: 画像ファイル名
- score: 好み度スコア（0-100%）
- label: 判定ラベル（like/dislike）

【注意】
- このフォルダの内容は score.py 実行時に自動生成されます
- 手動でファイルを配置する必要はありません
