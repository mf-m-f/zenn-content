# zenn-content

個人アカウント [`sas625efta`](https://github.com/sas625efta) 向けの Zenn 記事・Notebook リポジトリです。

## 構成

```
articles/     … Zenn 記事
notebooks/    … 実行用 Jupyter Notebook
requirements.txt
```

## 執筆フロー

1. Google Drive `Career/Zenn/` で下書き（`article.md`, `code.ipynb`, `research.md`）
2. 完成したら本リポジトリへ複製
3. `sas625efta/zenn-content` に push → Zenn GitHub 連携で公開

`research.md` は Drive のみ（GitHub には載せない）。

## 初回 push 手順

GitHub で空リポジトリ `zenn-content` を **sas625efta** アカウントに作成してから:

```bash
cd ~/GitHub/zenn-content
git init
git add .
git commit -m "Initial commit: Zenn articles and notebooks"
git branch -M main
git remote add origin git@github.com:sas625efta/zenn-content.git
git push -u origin main
```

Zenn ダッシュボード → GitHub 連携 → `sas625efta/zenn-content` を選択。

## Python 環境

```bash
python3 -m venv ~/venvs/career-zenn
~/venvs/career-zenn/bin/pip install -r requirements.txt
```
