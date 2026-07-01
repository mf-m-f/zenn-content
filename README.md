# zenn-content

材料・化学 × 機械学習に関する [Zenn](https://zenn.dev) 記事と、記事で紹介するコードの Jupyter Notebook です。

## 記事と Notebook

| 記事（`articles/`） | Notebook |
|---------------------|----------|
| 材料・化学の実験データをMLに使う前にやること | [`notebooks/materials-data-cleansing/code.ipynb`](notebooks/materials-data-cleansing/code.ipynb) |
| 小データでの機械学習——研究現場で使える現実的な戦略 | [`notebooks/small-data-ml/code.ipynb`](notebooks/small-data-ml/code.ipynb) |
| 材料・化学データでのSHAP解析と逆解析 | [`notebooks/shap-inverse-analysis/code.ipynb`](notebooks/shap-inverse-analysis/code.ipynb) |

記事本文は `articles/` 以下の Markdown です。Zenn への公開は GitHub 連携で行っています。

## Notebook の実行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

各 Notebook はデモ用のサンプルデータを内部で生成するため、追加のデータファイルは不要です。
