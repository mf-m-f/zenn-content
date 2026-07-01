---
title: "小データでの機械学習——研究現場で使える現実的な戦略"
emoji: "📊"
type: "tech"
topics: ["python", "機械学習", "材料"]
published: true
---

## はじめに：「精度が出ない」の前に確認すること

マテリアルズ・インフォマティクス（MI）に取り組んでいると、こんな状況にぶつかる。

- モデルを訓練データに当てはめると精度は出るが、新しいサンプルで外れる
- 5-Fold CVで高い R² が出たが、次の実験で全く予測が当たらない
- アルゴリズムを変えてもどれも似たような結果になる

原因の多くは「小データ」に特有の問題だ。サンプル数 n = 50〜200 程度の実験データに対して、一般的な機械学習の手順をそのまま適用すると、評価も実用性も誤った方向に進む。

本記事では、材料・化学の実験データを小データで扱う際の**モデル選択・評価設計・不確実性の扱い・実験設計**について、実務的な視点で整理する。

記事内の Python コードは説明用の断片です。デモデータ生成を含む実行可能版は GitHub の [code.ipynb](https://github.com/mf-m-f/zenn-content/blob/main/notebooks/small-data-ml/code.ipynb) を参照。（[Colab で開く](https://colab.research.google.com/github/mf-m-f/zenn-content/blob/main/notebooks/small-data-ml/code.ipynb)）

---

## 1. 小データ問題の構造

### なぜ小データが常態化するか

材料・化学の実験データが小さくなる理由は、単純に「まだ集めていない」ではない。

- **合成コスト**：新規合金やポリマーの合成は、原料選定から焼成・冷却まで数日〜数週間かかる
- **測定コスト**：X線回折・透過型電子顕微鏡・各種スペクトル測定は、高度なスキルと高額な装置を要する
- **探索空間の希少性**：特定の機能性材料では、世界中の文献を集めても数十件しかない

これらの制約は、実験の効率化で解消できる部分もあるが、本質的に「測定に時間とコストがかかる」という構造は変わらない。小データは材料科学の宿命であり、前提として受け入れた上で戦略を立てる必要がある。

### n << p 問題

サンプル数 n が少ない状況で、記述子（特徴量）数 p が多い場合、機械学習は実際の物理法則ではなく「ノイズとたまたま相関した特徴量の組み合わせ」を学習してしまう。高次元空間では限られたデータ点が互いに遠く離れ（次元の呪い）、モデルの汎化が困難になる。

**実務的な目安：**

| サンプル数 | 状況 | 推奨戦略 |
|:--------:|:-----|:--------|
| n < 50 | 極端な小データ | 線形正則化モデル、ドメイン知識ベースの少数特徴量 |
| n = 50〜200 | 典型的な実験データ | Ridge/Lasso、GPR、制約付きRF |
| n = 200〜1000 | 比較的恵まれた状況 | アンサンブル、NN（慎重に） |
| n > 10000 | 計算データ等 | 深層学習の直接適用が視野に |

### サンプル数と特徴量数の目安

n と p の比率は、どのモデルを使うかによって許容範囲が変わる。よく引用される経験則をまとめると以下のようになる[^pnratio]。

| 手法 | 推奨される p/n の上限 | 備考 |
|:-----|:------------------:|:----|
| 正則化なし線形回帰 | p < n/10 | 係数推定が不安定。実務ではほぼ使えない |
| Ridge / GPR | p < n/5 程度 | 正則化が多重共線性を吸収するが限界あり[^esl] |
| Lasso / ElasticNet | p > n でも動作するが要注意 | 「真に重要な特徴量が少ない」スパース性の仮定が必要[^lasso] |
| Random Forest | p ≈ n 程度まで耐性あり | 特徴量サブサンプリングが効く。ただし過学習には別途対処が必要 |

**材料データへの当てはめ：** サンプル数60件なら特徴量は最大12個（1:5）が目安だ。matminer の Magpie 特徴量は組成から自動で100次元以上を生成するため、そのまま投入すると深刻な n << p 状態になる。モデルを学習させる前に、Lasso による特徴量選択か PLS による次元削減を行うことが材料データでは実質的に必須になる（セクション6参照）。

### 小データで起きる「高分散」の典型的症状

過学習は「訓練精度は高いがテスト精度が低い」だが、小データではさらに**高分散**の問題が出る。データセットへの外れ値の混入1件で回帰係数の符号が逆転したり、データ分割の乱数シードを変えるだけで精度が大きく変わる。

これは「モデルが悪い」のではなく「データが少なすぎてモデルが不安定」なのであり、解決策はモデルの複雑度を下げるか、正則化を強くするか、評価に複数の分割を使うことになる。

---

## 2. モデル選択の戦略：シンプルから始める

小データにおける最大の原則は「オッカムの剃刀」だ。訓練データが少ないほど、複雑なモデルは過学習のリスクが高い。まず最もシンプルなモデルでベースラインを確立し、必要に応じて複雑さを上げていく。

### 各手法の比較

| 手法 | 小データ適性 | メリット | デメリット | 材料データでの用途 |
|:-----|:-----------:|:--------|:---------|:----------------|
| **Ridge回帰** | ◎ | 解釈性が高い、多重共線性に強い | 非線形性を捉えられない | ベースライン、係数の物理的解釈 |
| **Lasso回帰** | ◎ | 自動特徴量選択、スパースな解 | 強相関する特徴量の選択が不安定 | 重要記述子の特定 |
| **SVR** | ○ | 外れ値にロバスト、非線形カーネル | ハイパーパラメータ調整がシビア、不確実性が出ない | 触媒性能予測、MOF特性 |
| **Random Forest** | △〜○ | 非線形・交互作用を自動学習、スケール不変 | 外挿が苦手、max_depth制限必須 | 組成特徴量でのスクリーニング |
| **GPR** | ◎ | 予測値と不確実性を同時出力、過学習しにくい | 計算コストO(n³)（小データでは問題なし） | ベイズ最適化との連携、信頼区間付き予測 |
| **SISSO** | ◎ | 物理的解釈性が極めて高い数式を抽出 | 計算コストが高い、特徴空間設計が必要 | 圧電・磁性材料の記述子発見 |
| **NN** | ✗ | （大データでは強力） | 小データでは深刻な過学習。直接適用は基本NG | 転移学習（事前学習済みモデル）経由でのみ |

### コード①：モデル比較（LOOCV）

```python
# df はあらかじめ実験データが格納されている DataFrame とします
# （batch_id は評価用ラベルなので特徴量から除外する）
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

X = df.drop(columns=["target", "batch_id"]).values
y = df["target"].values

# --- モデル定義 ---
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
models = {
    "Ridge":        Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso":        Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.01, max_iter=5000))]),
    "SVR":          Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=10, epsilon=0.1))]),
    "RF(depth=3)":  Pipeline([("sc", StandardScaler()), ("m", RandomForestRegressor(n_estimators=200, max_depth=3, random_state=42))]),
    "GPR":          Pipeline([("sc", StandardScaler()), ("m", GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42))]),
}

# --- LOOCV で評価 ---
# R² は各フォールドのテストが1件のため cross_val_score では NaN になる。
# 外れフォールド予測を集めてから r2_score を計算する。
loo = LeaveOneOut()
rows = []
for name, model in models.items():
    y_pred = cross_val_predict(model, X, y, cv=loo)
    r2 = r2_score(y, y_pred)
    mae = cross_val_score(model, X, y, cv=loo, scoring="neg_mean_absolute_error")
    rows.append({"Model": name,
                 "LOOCV R2": f"{r2:.3f}",
                 "LOOCV MAE": f"{(-mae).mean():.1f} ± {(-mae).std():.1f}"})

print(pd.DataFrame(rows).to_string(index=False))
```

**ポイント：** Random Forestは `max_depth=3` のように**深さを明示的に制限**する。制限しないと訓練データを記憶してLOOCV R² が見かけ上高くなる（Notebook では学習曲線で比較）。GPRは不確実性も同時に出力できるため、次節のベイズ最適化との相性が良い。

LOOCV は n < 50 で特に有効だが、本記事のデモ（n = 80）でも動作確認に使っている。n = 50〜200 では Repeated 5-Fold CV の方が推定のばらつきを見やすい（セクション3の表参照）。実行可能な全文は [code.ipynb](https://github.com/mf-m-f/zenn-content/blob/main/notebooks/small-data-ml/code.ipynb) を参照。

---

## 3. 正しいモデル評価と過学習の防止

小データ分析で最も危険なのは、評価プロセスの欠陥により「本物の汎化性能と誤認する」ことだ。

### 正則化パラメータの選択

Ridge・Lasso・ElasticNetの正則化強度 α はクロスバリデーションで決める。`RidgeCV`・`LassoCV`・`ElasticNetCV` はこれを内部で効率的に行う。

### コード②：正則化パラメータのCV選択

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = df.drop(columns=["target", "batch_id"]).values
y = df["target"].values

alphas = np.logspace(-3, 3, 50)

ridge_cv = Pipeline([
    ("sc", StandardScaler()),
    ("m",  RidgeCV(alphas=alphas, cv=5, scoring="r2"))
])
lasso_cv = Pipeline([
    ("sc", StandardScaler()),
    ("m",  LassoCV(alphas=alphas, cv=5, max_iter=10000))
])
elastic_cv = Pipeline([
    ("sc", StandardScaler()),
    ("m",  ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=5, max_iter=10000))
])

for name, pipe in [("Ridge", ridge_cv), ("Lasso", lasso_cv), ("ElasticNet", elastic_cv)]:
    pipe.fit(X, y)
    model = pipe.named_steps["m"]
    best_alpha = getattr(model, "alpha_", None)
    print(f"{name}: best α = {best_alpha:.5f}")

# Ridge の係数（特徴量の重みを物理的解釈に使う）
coef = ridge_cv.named_steps["m"].coef_
feature_names = df.drop(columns=["target", "batch_id"]).columns
importance = pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)
print("\n特徴量の重要度（Ridge係数の絶対値）:")
print(importance)
```

### クロスバリデーション戦略の使い分け

| CV手法 | 推奨サンプル数 | メリット | デメリット | 計算コスト |
|:------|:------------:|:--------|:---------|:---------:|
| **LOOCV** | n < 50 | データを最大活用（n-1件で学習）。バイアスが最小 | 推定分散が大きい。シードを変えても結果が変わらない（分割が一意）ため、ばらつきの評価ができない | 低〜中（n回学習） |
| **Repeated 5-Fold CV** | n = 50〜200 | 乱数シードを変えて複数回繰り返すことで分割依存のブレを平均化。安定した推定値が得られる | LOOCVより使えるデータが少し減る（4/5を訓練に使用） | 中（繰り返し数 × 5回学習） |
| **Nested CV** | 厳密な評価が必要な場合 | ハイパーパラメータ選択と汎化性能評価を完全に分離。最も正直な汎化性能の推定値 | 外側の各分割でさらに訓練データが減る。計算コストが最も高い | 高（外側k × 内側k × 候補数） |

### コード③：Nested CV + GroupKFold（データリーク防止）

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV

X = df.drop(columns=["target", "batch_id"]).values
y = df["target"].values
groups = df["batch_id"].values  # 同一バッチ・同一組成系のグループラベル

pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge())])
param_grid = {"m__alpha": np.logspace(-3, 3, 20)}
inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)

# 比較1: 通常の K-Fold（グループ無視）→ リークしやすい
kfold_scores = []
for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X):
    gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="r2")
    gs.fit(X[tr], y[tr])
    kfold_scores.append(gs.score(X[te], y[te]))

# 比較2: GroupKFold（バッチ単位で分割）
# n_splits はグループ数以下（例: バッチ4種なら n_splits=4）
group_scores = []
for tr, te in GroupKFold(n_splits=4).split(X, y, groups):
    gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="r2")
    gs.fit(X[tr], y[tr])
    group_scores.append(gs.score(X[te], y[te]))

print(f"通常K-Fold   Nested CV R²: {np.mean(kfold_scores):.3f} ± {np.std(kfold_scores):.3f}")
print(f"GroupKFold   Nested CV R²: {np.mean(group_scores):.3f} ± {np.std(group_scores):.3f}")
```

**なぜ GroupKFold か：** 同一バッチで合成されたサンプル群や、同一組成系の微量変化データは強い類似性を持つ。これを単純なランダム分割でtrain/testに混在させると、モデルは「記憶」するだけで高精度を出す。`GroupKFold` でグループ単位に分割することで、真の汎化性能を評価できる[^groupkfold]。

`n_splits` はグループ数以下に設定する（グループ数を超えるとエラー）。バッチ間に相関がある実データでは、通常 K-Fold より GroupKFold のスコアが低く（保守的に）出ることが多い。デモデータではバッチ間の差が小さいため、両者の差が小さく出る場合もある。

### コード④：学習曲線による診断

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

X = df.drop(columns=["target", "batch_id"]).values
y = df["target"].values

pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])

train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y,
    train_sizes=np.linspace(0.2, 1.0, 8),
    cv=5, scoring="r2", n_jobs=-1,
    shuffle=True, random_state=42   # 訓練サイズごとにランダムサンプリング（先頭から取る偏りを防ぐ）
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train R²")
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
ax.plot(train_sizes, val_scores.mean(axis=1), "s--", label="CV R²")
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
ax.set_xlabel("Training samples")
ax.set_ylabel("R²")
ax.set_title("Learning Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

[code.ipynb](https://github.com/mf-m-f/zenn-content/blob/main/notebooks/small-data-ml/code.ipynb) では Ridge（適切な α）・Ridge（弱い正則化）・RF（`max_depth` なし）の3モデルを並べて比較している。過学習の違いを視覚的に確認したい場合はセクション5を参照。

学習曲線の読み方：

- **Train R² が高く、CV R² が低いまま乖離している** → 過学習。モデルを単純化 or 正則化を強化
- **Train R² と CV R² がともに低い** → アンダーフィット。特徴量が不足 or モデルが単純すぎる
- **両者が収束している** → データが増えても精度が頭打ち。特徴量の質が問題

---

## 4. 不確実性の定量化

材料探索において、予測値だけ出すモデルは使いにくい。「この組成は性能が高そうだが、モデルがその領域を知らないだけかもしれない」という判断ができないからだ。不確実性を定量化することで、次の実験候補の選び方が変わる。

### GPRの不確実性を活用する

ガウス過程回帰（GPR）は、予測値（事後分布の平均 μ）と予測不確実性（事後分散の標準偏差 σ）を同時に出力できる。学習データから遠い領域ほど σ が大きくなる性質を持つため、「モデルが自信を持って予測しているか、単に外挿しているだけか」を区別できる[^gpr]。

### コード⑤：GPR による予測と信頼区間

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target", "batch_id"]).values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# スケーリング
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_train)
X_te_sc = sc.transform(X_test)

# GPR（カーネル = 定数 × RBF + ノイズ）
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(0.1, (1e-5, 1.0))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gpr.fit(X_tr_sc, y_train)

# 予測 + 標準偏差
y_pred, y_std = gpr.predict(X_te_sc, return_std=True)

# 予測 vs 実測プロット（信頼区間付き）
sorted_idx = np.argsort(y_test)
x_plot = np.arange(len(y_test))

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x_plot, y_test[sorted_idx], color="black", zorder=5, label="Actual", s=40)
ax.plot(x_plot, y_pred[sorted_idx], "o-", color="steelblue", label="GPR mean")
ax.fill_between(x_plot,
                y_pred[sorted_idx] - 2 * y_std[sorted_idx],
                y_pred[sorted_idx] + 2 * y_std[sorted_idx],
                alpha=0.3, color="steelblue", label="95% CI")
ax.set_xlabel("Sample index (sorted by actual)")
ax.set_ylabel("Target property")
ax.set_title("GPR prediction with uncertainty")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 不確実性が大きいサンプルを確認
uncertainty_df = pd.DataFrame({
    "y_actual": y_test,
    "y_pred":   y_pred,
    "σ (std)":  y_std,
    "error":    np.abs(y_pred - y_test)
}).sort_values("σ (std)", ascending=False)
print("不確実性が高いサンプル Top 5:")
print(uncertainty_df.head())
```

---

## 5. ベイズ最適化と能動学習

GPRの不確実性を活用すると、「次にどの実験をやるべきか」を定量的に決定できる。これがベイズ最適化（BO）を用いた能動学習の考え方だ。

### Exploitation と Exploration のトレードオフ

BOでは獲得関数（Acquisition Function）が、次の候補点を選ぶ。

- **Exploitation（活用）**：予測値 μ が高い領域を狙う。現在の最良点の近傍を最適化
- **Exploration（探索）**：不確実性 σ が高い領域を狙う。未踏領域のデータを取得してモデルを改善

**Expected Improvement（EI）** はこの両者を自動的にバランスする最も広く使われる獲得関数だ[^bayesopt]：

$$\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - f^*) \cdot \Phi(Z) + \sigma(\mathbf{x}) \cdot \phi(Z), \quad Z = \frac{\mu(\mathbf{x}) - f^*}{\sigma(\mathbf{x})}$$

ここで f\* は現在の最良値、Φ と φ はそれぞれ標準正規分布の CDF と PDF。

### 材料科学向け BO ライブラリの選択

| ライブラリ | 特徴 | 使いどころ |
|:---------|:----|:---------|
| **PHYSBO** | 材料科学向け国産ツール。大規模離散候補プールへの高速適用 | 組成DBから最適候補を選ぶスクリーニング[^physbo] |
| **Ax / BoTorch** | PyTorchベース、多目的最適化・複雑な制約に対応 | 連続プロセス変数（温度・圧力）の最適化 |
| **scikit-optimize** | 軽量、学習コストが低い | 手元のデータ数十件での簡易BO |

### コード⑥：簡易ベイズ最適化（EI 獲得関数 + GPR）

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

def expected_improvement(X_candidates, gpr, y_best, xi=0.01):
    """EI 獲得関数：候補点 X_candidates の EI を計算して返す"""
    mu, sigma = gpr.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

# 既存のデータで GPR を学習
feature_cols = df.drop(columns=["target", "batch_id"]).columns.tolist()
X_known = df[feature_cols].values
y_known = df["target"].values

sc = StandardScaler()
X_sc = sc.fit_transform(X_known)

kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
gpr.fit(X_sc, y_known)

# 未実験の候補セット（実験可能な組成・条件の全候補）
# X_candidates は事前にドメイン知識で設定した候補リスト
X_candidates_sc = sc.transform(X_candidates)

# EI を計算して上位 5 件を提案
y_best = y_known.max()
ei_scores = expected_improvement(X_candidates_sc, gpr, y_best)

top5_idx = np.argsort(ei_scores)[::-1][:5]
print("次の実験候補（EI順）:")
for rank, idx in enumerate(top5_idx, 1):
    mu, sigma = gpr.predict(X_candidates_sc[[idx]], return_std=True)
    print(f"  Rank {rank}: EI={ei_scores[idx]:.4f}, μ={mu[0]:.3f}, σ={sigma[0]:.3f}")
    print(f"    条件: {dict(zip(feature_cols, X_candidates[idx]))}")
```

:::message
**BO は「次の実験候補を選ぶツール」であり「予測ツール」ではない**

コードで出力している μ（予測値）は参考値であり、外挿領域では信頼性が低い。予測精度を確認したい場合は、コード①②のように LOOCV などで評価したモデル（Ridge / GPR）で行う。BO と予測モデルは役割が異なる。

また μ を実験候補の判断に使う場合は、必ず σ（不確実性）とセットで確認すること。σ が大きい候補は「性能が良い」のではなく「モデルがその領域を知らない」だけの可能性がある。
:::

---

## 6. 特徴量エンジニアリング

小データほど、特徴量の設計がモデルの性能を左右する。アルゴリズムの選定以上に重要だと言っても過言ではない。

### 材料科学で有効な記述子

| 記述子の種類 | ツール | 特徴 |
|:-----------|:------|:----|
| **組成ベース（Magpie）** | matminer | 化学式から元素の電気陰性度・原子半径・融点の平均/分散を自動生成。合成前でも使える |
| **分子記述子** | RDKit | SMILES から Morgan フィンガープリント・トポロジー情報を数値化 |
| **構造ベース（SOAP, CM）** | dscribe | 3D座標から局所化学環境を表現。情報量は多いが次元が増える |

### 組成データの前処理：CLR変換

合金の配合比率やポリマーの組成比のように、**成分の合計が100%（または1）になる制約を持つデータ**は「組成データ（Compositional Data）」として特別な扱いが必要だ。

通常の空間でそのまま回帰モデルに入力すると、ある成分を増やせば必ず別の成分が減るという数学的な制約から、成分間に**偽相関（閉包問題）**が生じる。前記事でも触れたが、この問題への対処として**中心対数比（CLR: Centered Log-Ratio）変換**が有効だ[^coda]。

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np

def clr_transform(X):
    log_X = np.log(X + 1e-10)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean

# 使用例：合金の組成比（wt%）を CLR 変換してからモデルに入力
composition_cols = ["Fe_wt", "Ni_wt", "Cr_wt"]
X_comp = df[composition_cols].values / 100   # wt% -> 比率
X_clr  = clr_transform(X_comp)
```

CLR 変換は前記事（データクレンジング編）でも扱っているが、小データの回帰モデルに組成データを入力する際は**モデル学習の前に必ず適用する**べきステップだ。

### 教師なし vs 教師あり次元削減

特徴量が多い場合、次元削減は有効だが選択を誤ると重要な情報を捨てる。

- **PCA**（教師なし）：分散が大きい方向を抽出するが、ターゲット変数との相関が低い成分が落ちる可能性がある
- **PLS**（教師あり）：ターゲット変数との共分散を最大化する方向で次元削減。小データでの回帰タスクに有効

### コード⑦：PLS vs PCA の次元削減比較

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

X = df.drop(columns=["target", "batch_id"]).values
y = df["target"].values

cv = KFold(n_splits=5, shuffle=True, random_state=42)
n_components_list = [1, 2, 3, 4, 5, 6]  # 特徴量数以下

results = []
for n in n_components_list:
    # PCA + Ridge
    pipe_pca = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=n)),
        ("m",   Ridge(alpha=1.0))
    ])
    r2_pca = cross_val_score(pipe_pca, X, y, cv=cv, scoring="r2").mean()

    # PLS（教師あり次元削減＋回帰を同時実行）
    pls = Pipeline([
        ("sc", StandardScaler()),
        ("m",  PLSRegression(n_components=n))
    ])
    r2_pls = cross_val_score(pls, X, y, cv=cv, scoring="r2").mean()

    results.append({"n_components": n, "CV R² (PCA+Ridge)": round(r2_pca, 3), "CV R² (PLS)": round(r2_pls, 3)})

print(pd.DataFrame(results).to_string(index=False))
```

---

## 7. 転移学習の活用（データ量が絶対的に不足する場合）

実験データが数十件しか取れない場合でも、DFT 計算データベース（Materials Project、OQMD、NOMAD など、数十万〜数百万件規模）で事前学習されたモデルを転用できる。

**流れ：**
1. 大規模計算データベースでグラフニューラルネットワーク（GNN）を事前学習
2. 目的の実験データ（数十〜数百件）でファインチューニング
3. DFT の系統誤差（バンドギャップの過小評価など）を実験値で補正

**利用可能なモデル：** CHGNet、M3GNet、ALIGNN（いずれも事前学習済みモデルが公開されている）

近年は自然言語処理の Transformer アーキテクチャを応用した **CrystalTransformer** などのモデルも登場している。原子の普遍的な特徴表現（Universal Atomic Embeddings）を獲得することで、従来の GNN より予測精度と汎化性能が向上しているケースが報告されている[^crystaltransformer]。

転移学習は「小データを増やす」のではなく「大量の計算データから抽出した物理的規則性を、実験データの少ないモデルに注入する」アプローチだ[^transfer]。計算コストが高い DFT を直接使えない研究室でも、公開モデルを活用することで小データの壁を越えられる。

### データ拡張（Data Augmentation）の罠

画像認識では回転・反転・ノイズ付加などのデータ拡張が一般的だが、**表形式の材料データへの安易な適用は危険**だ。

SMOTE などのオーバーサンプリングや単純なノイズ付加は、物理的・化学的に存在し得ない架空の組成や構造（無効な SMILES・不安定な結晶相）を生成するリスクが高い[^augmentation]。モデルはその架空データを「本物」として学習してしまう。

材料データでデータ拡張が安全に機能するのは以下の限られたケースだ。

- **3次元座標データへの回転・並進操作**：物理的に等価な変換のため情報が壊れない
- **顕微鏡画像からの結晶領域認識**：疑似画像によるドメイン適応
- **同一系統の類似材料データの統合**：厳密にはデータ拡張ではなくデータ収集の工夫

表形式データのサンプル数不足は、データ拡張で解決しようとするより、**ドメイン知識に基づく特徴量設計**か**転移学習**で対処する方が材料科学では現実的だ。

---

## まとめ：小データ ML の実践的チェックリスト

材料・化学の小データで機械学習を使う際の判断フローをまとめる。

**① データ準備**
- [ ] 欠損値・外れ値の処理は完了しているか（前記事参照[^cleaning]）
- [ ] 同一バッチ・同一組成系のグループ構造を `batch_id` などで記録したか
- [ ] 特徴量の次元数は n と比べて適切か（目安：p < n/5）

**② モデル選択**
- [ ] まず Ridge/Lasso でベースラインを確立したか
- [ ] GPR で不確実性を同時推定できる状態になっているか
- [ ] Random Forest を使う場合、`max_depth` を制限したか

**③ 評価設計**
- [ ] `GroupKFold` でデータリークを防止しているか
- [ ] Nested CV でハイパーパラメータ選択と汎化性能評価を分離したか
- [ ] 学習曲線で過学習・アンダーフィットを確認したか

**④ 次の実験設計**
- [ ] GPR の σ（不確実性）を確認し、外挿領域への過信を避けているか
- [ ] EI などの獲得関数で Exploitation/Exploration のバランスを取っているか

小データ ML の本質は「アルゴリズムの最新化」ではなく、**少ないデータから信頼できる知識を引き出す設計**にある。モデルの予測値だけでなく、その不確実性をどう次の実験計画に活かすかが、研究サイクルの効率を左右する。

---

[^pnratio]: 材料インフォマティクスにおけるサンプル数と特徴量数の関係については Benchmark datasets incorporating diverse tasks, sample sizes, material systems, and data heterogeneity for materials informatics (PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC8319566/) を参照。
[^esl]: Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning (2nd ed., Springer, 2009). 正則化の効果と p/n 比の関係についての統計的根拠を提供する標準的な教科書。
[^lasso]: Tibshirani R. Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society, Series B, 58(1):267-288, 1996. p > n でも動作する条件（スパース性）を示した原著論文。
[^groupkfold]: GroupKFold によるデータリーク防止は小データ材料データ解析で特に重要。実験バッチや組成系でグループを定義し、`sklearn.model_selection.GroupKFold` を使用する。
[^gpr]: Gaussian Process Regression の詳細は scikit-learn ドキュメント参照: https://scikit-learn.org/stable/modules/gaussian_process.html
[^bayesopt]: ベイズ最適化の材料探索への応用: A survey of active learning in materials science (arXiv:2601.06971)
[^physbo]: PHYSBO (optimization tools for PHYsics based on Bayesian Optimization): https://www.pasums.issp.u-tokyo.ac.jp/physbo/en/about
[^transfer]: 転移学習による実験データ予測精度の向上: Enhancing Materials Property Prediction by Leveraging Computational and Experimental Data using Deep Transfer Learning (NIST)
[^coda]: 組成データ分析（CoDA）と CLR 変換: CoDaWeb (https://www.compositionaldata.com/)。前記事「材料・化学の実験データをMLに使う前にやること」のセクション「CLR変換」も参照。
[^crystaltransformer]: Crystal Transformer Based Universal Atomic Embedding for Accurate and Transferable Prediction of Materials Properties (arXiv:2401.09755)
[^augmentation]: 材料データに対するデータ拡張の限界と安全なケース: Machine learning of fake micrographs for automated analysis of crystal growth process (Taylor & Francis, https://www.tandfonline.com/doi/full/10.1080/27660400.2022.2082235)
[^cleaning]: 材料・化学の実験データをMLに使う前にやること——欠損・外れ値・単位不統一の実践的対処法（本マガジン前記事）