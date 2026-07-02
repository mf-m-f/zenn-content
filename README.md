# zenn-content

材料・化学 × 機械学習に関する [Zenn](https://zenn.dev/efta1989) 記事と、記事で紹介するコードの Jupyter Notebook です。

## 記事と Notebook

| Zenn slug（`articles/` のファイル名） | 記事 | Notebook |
|--------------------------------------|------|----------|
| `e53a7b8fcd5ec6` | 材料・化学の実験データをMLに使う前にやること | [`notebooks/materials-data-cleansing/code.ipynb`](notebooks/materials-data-cleansing/code.ipynb) |
| `1166234e9506f7` | 小データでの機械学習——研究現場で使える現実的な戦略 | [`notebooks/small-data-ml/code.ipynb`](notebooks/small-data-ml/code.ipynb) |
| `shap-inverse-analysis-for-materials` | 材料・化学データでのSHAP解析と逆解析 | [`notebooks/shap-inverse-analysis/code.ipynb`](notebooks/shap-inverse-analysis/code.ipynb) |
| `target-variable-design-for-materials` | 材料・化学データの目的変数設計——対数変換バイアスと圧縮スコアの落とし穴 | [`notebooks/target-variable-design/code.ipynb`](notebooks/target-variable-design/code.ipynb) |

正本は [`article-manifest.json`](article-manifest.json) です。`articles/` に置ける Markdown はこの一覧だけに限定してください。

## GitHub 連携の運用ルール

1. **ファイル名 = slug = URL 末尾**（例: `1166234e9506f7.md` → `zenn.dev/efta1989/articles/1166234e9506f7`）
2. **Web で既に公開した記事の slug は変更しない**（リネームすると別記事として重複する）
3. **新規記事だけ**意味のある slug を付けてよい（SHAP 記事など）
4. push 前に `python3 scripts/check_articles.py` で manifest と `articles/` の一致を確認する

下書きの Drive 原稿は `Business/Career/Zenn/` に置き、公開用に整えてから `articles/{slug}.md` へ反映して push します。

## Notebook の実行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

各 Notebook はデモ用のサンプルデータを内部で生成するため、追加のデータファイルは不要です。
