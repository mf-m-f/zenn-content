---
title: "材料・化学データでのSHAP解析と逆解析——モデルを信じる前に確認すること"
emoji: "🔬"
type: "tech"
topics: ["機械学習", "SHAP", "材料科学", "Python", "逆解析"]
published: false
---

## はじめに：「なぜこの予測値か」を説明できるか

モデルの LOOCV R² が 0.90 を超えた。次は何をするか。

多くの場合、ここで「モデルを使って最適条件を探す」に進む。しかし材料科学の文脈では、この手順には一つ重要なステップが抜けている。

**そのモデルが、どの変数をどう使って予測しているかを確認する**ことだ。

高精度なモデルが「Ni を 35% 入れて 1200℃ で焼成せよ」と提案しても、実験者はそれを無条件に信じない。ドメイン知識と整合しているか、外挿領域への提案ではないか——これを確認できないと、高コストな実験リソースを投入する根拠が薄い。

SHAP（SHapley Additive exPlanations）はその確認作業のためのツールだ。モデルの予測値を「各特徴量の寄与の足し算」に分解し、数値として出力する。さらにその解釈を終えた後、`scipy.optimize` とガウス過程回帰（GPR）の不確実性を使った逆解析まで、本記事では一連の流れとして整理する。

デモデータは前記事（小データ機械学習）と同じ Fe/Ni/Cr 合金の引張強度データ（n=80）を使う。

記事内の Python コードは説明用の断片です。デモデータ生成を含む実行可能版は GitHub の [code.ipynb](https://github.com/mf-m-f/zenn-content/blob/main/notebooks/shap-inverse-analysis/code.ipynb) を参照。（[Colab で開く](https://colab.research.google.com/github/mf-m-f/zenn-content/blob/main/notebooks/shap-inverse-analysis/code.ipynb)）

---

## 1. SHAP の基礎：予測値を「足し算」に分解する

### シャープレイ値とは何か

SHAP は協力ゲーム理論の「シャープレイ値」を機械学習に応用した手法だ。

考え方はシンプルだ。あるサンプルの予測値が「平均予測値（Base value）」とどれだけ違うか——その差を、各特徴量が「どれだけ貢献したか」に公平に分配する。「公平」とは数学的に定義されており、以下の4つの性質を同時に満たす唯一の分配方法がシャープレイ値になる。

- **効率性**：全特徴量の SHAP 値の総和 ＝ 予測値 − 平均予測値
- **対称性**：同じ貢献をした特徴量には同じ SHAP 値が割り当てられる
- **ダミー**：予測に影響しない特徴量の SHAP 値は必ずゼロ
- **加法性**：アンサンブルモデルの SHAP 値 ＝ 各モデルの SHAP 値の和

「Base value」（平均予測値）は、モデルが特徴量の情報を何も受け取っていないときに出す予測値——つまり訓練データ全体の予測平均だ。各特徴量の SHAP 値がプラスであれば「平均よりも予測値を上げる方向に寄与した」、マイナスであれば「下げる方向に寄与した」を意味する。

### Explainer の選び方

SHAP ライブラリには対象モデルに応じた複数の計算アルゴリズム（Explainer）がある。

| Explainer | 対象モデル | 特徴 |
|:---------|:---------|:----|
| **TreeExplainer** | ランダムフォレスト、XGBoost、LightGBM | 決定木の構造を使った高速な厳密計算。材料データでの実質的な標準 |
| **LinearExplainer** | 線形回帰、Ridge、Lasso | 解析的に計算できるため高速。ただし線形モデル専用 |
| **KernelExplainer** | 任意のモデル（SVR、NN 等） | モデル非依存だが計算コストが膨大。特徴量が多いと実用的でない |

Random Forest を使う場合は `TreeExplainer` を選べばよい。

### 小データでの注意点

最初に確認しておくべき大前提がある。

**SHAP はモデルの判断を説明するツールであり、物理法則を説明するツールではない。**

モデルがノイズや疑似相関を学習してしまっていれば、SHAP はその疑似相関に対して大きな寄与度を割り当てる。SHAP の出力がドメイン知識と大きく反している場合、それは SHAP の問題ではなく、モデルの学習品質の問題だ。

小データ（n < 200）では、LOOCV や GroupKFold で汎化性能が確認されたモデルに対して SHAP を適用することを推奨する。

---

## 2. 事前確認：多重共線性チェック

SHAP 解釈の前に、**特徴量間の多重共線性を確認**しておく必要がある。強く相関する特徴量が複数あると、SHAP が重要度を両者に分散させ、真の支配因子が見えにくくなる（Section 8 で実演）。

確認するのは2点だ。相関行列（|r| > 0.8 程度が要注意）と、VIF（分散拡大係数。VIF > 10 は除外・統合の目安）。

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

feature_cols = ['Fe', 'Ni', 'Cr', 'temp', 'time', 'cool_r']
X_df = df[feature_cols]

# 相関行列
corr = X_df.corr()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(feature_cols)))
axes[0].set_yticks(range(len(feature_cols)))
axes[0].set_xticklabels(feature_cols, rotation=45, ha='right')
axes[0].set_yticklabels(feature_cols)
for i in range(len(feature_cols)):
    for j in range(len(feature_cols)):
        axes[0].text(j, i, f'{corr.values[i, j]:.2f}',
                     ha='center', va='center', fontsize=9,
                     color='white' if abs(corr.values[i, j]) > 0.6 else 'black')
plt.colorbar(im, ax=axes[0])
axes[0].set_title('特徴量間の相関行列')

# VIF
vif_data = pd.DataFrame({
    'feature': feature_cols,
    'VIF': [variance_inflation_factor(X_df.values, i)
            for i in range(len(feature_cols))]
}).sort_values('VIF', ascending=False)

colors = ['#d62728' if v > 10 else '#ff7f0e' if v > 5 else '#2ca02c'
          for v in vif_data['VIF']]
axes[1].barh(vif_data['feature'], vif_data['VIF'], color=colors)
axes[1].axvline(x=10, color='red', linestyle='--', label='VIF=10（警戒線）')
axes[1].axvline(x=5,  color='orange', linestyle='--', label='VIF=5（注意線）')
axes[1].set_xlabel('VIF')
axes[1].set_title('VIF（分散拡大係数）')
axes[1].legend()

plt.tight_layout()
plt.show()
```

今回のデモデータでは **Fe の VIF が 51** という高い値になる。これは `Fe = 100 − Ni − Cr`（組成の閉包制約）によって Fe が Ni・Cr で完全に決まってしまうためで、構造的多重共線性だ。

:::message
**今回のデモの設定について（閉包制約）**

Fe・Ni・Cr を3変数すべて特徴量に入れると、この構造的共線性が生じる。今回は「3変数を入れたときに VIF と SHAP がどう振る舞うか」を**意図的に見せるためのデモ設定**だ。実務では、Ni・Cr のみを特徴量に使う（Fe をドロップする）か、CLR 変換（CoDA）で対処することを推奨する。
:::

---

## 3. モデルの訓練と LOOCV 評価

SHAP を適用するモデルは、前記事と同じく Random Forest（RF）と GPR の2種類を使う。評価は LOOCV（Leave-One-Out Cross-Validation）だ。

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
import numpy as np

feature_cols = ['Fe', 'Ni', 'Cr', 'temp', 'time', 'cool_r']
X = df[feature_cols].values
y_arr = df['tensile_strength'].values

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X, y_arr)

loo = LeaveOneOut()
y_pred_rf = cross_val_predict(rf, X, y_arr, cv=loo)
r2_rf = r2_score(y_arr, y_pred_rf)

# GPR
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
gpr.fit(X_sc, y_arr)

gpr_pipe = Pipeline([('sc', StandardScaler()),
                     ('gpr', GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42))])
y_pred_gpr = cross_val_predict(gpr_pipe, X, y_arr, cv=loo)
r2_gpr = r2_score(y_arr, y_pred_gpr)

print(f"RF  LOOCV R²: {r2_rf:.3f}")
print(f"GPR LOOCV R²: {r2_gpr:.3f}")
```

デモデータでの実行結果：RF R²=0.896、GPR R²=0.967。両モデルとも良好な汎化性能が確認できた。

:::message
**RF は `max_depth` を制限する**

`max_depth` を指定しないと、各葉ノードに訓練サンプルが1件ずつ入る「丸暗記」状態になる。訓練精度は 1.0 になるが、LOOCV では大きく下がる。今回は `max_depth=4` に制限している。
:::

---

## 4. SHAP の計算

SHAP の計算は1回で済ませ、以降の可視化で共通して使い回す。

```python
import shap

explainer = shap.TreeExplainer(rf)
X_df = pd.DataFrame(X, columns=feature_cols)
shap_values = explainer(X_df, check_additivity=False)

print(f"Base value（平均予測値）: {float(explainer.expected_value):.1f} MPa")
print(f"SHAP値の形状: {shap_values.values.shape}（サンプル数 × 特徴量数）")
```

`check_additivity=False` は、RF のノイズ推定による微小な加法性のずれを警告なしで許容するオプションだ。厳密な検証が必要な場合は外してよいが、通常は不要な警告を抑制するために指定する。

---

## 5. 大域解釈：Beeswarm plot から始める

SHAP を使う場合、**Beeswarm plot から始めることを推奨する**。

Beeswarm plot は全サンプルの SHAP 値を特徴量ごとに散布したもので、モデル全体の傾向を一枚で把握できる。

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
shap.plots.beeswarm(shap_values, max_display=10, show=False)
axes[0].set_title("Beeswarm plot（全サンプル）")

plt.sca(axes[1])
shap.plots.bar(shap_values, max_display=10, show=False)
axes[1].set_title("Bar plot（平均|SHAP|）")

plt.tight_layout()
plt.show()
```

**Beeswarm plot の読み方：**
- 横軸：SHAP 値（正 ＝ 予測値を上げる方向に寄与）
- 色：その特徴量の実際の値（赤 ＝ 高、青 ＝ 低）
- 縦軸の順序：SHAP 絶対値の平均が大きい順（＝ Feature Importance 順）

読み方の例：`Ni` の点が「赤が右側、青が左側」に分布していれば、Ni が多いほど引張強度が向上することをモデルが学習している。

**Beeswarm と Bar plot の使い分け：**

Beeswarm があれば、特徴量の重要度順序も、値が高い/低いときの方向性も、分布の広がりも、すべて一枚で読める。Bar plot は「数値で重要度を確認したいとき」の補足だ。Beeswarm を見せた後で数値を求められることはあるが、その場合に追加すればよい。

**赤青が混在している特徴量は要注意：**

`temp` の点が赤（高温）でも青（低温）でも SHAP 値が正負に混在する場合、単純な単調性ではなく非線形性や交互作用が疑われる。その特徴量を Dependence plot で深掘りする。

---

## 6. Dependence plot：非線形・交互作用を深掘りする

Beeswarm で赤青が混在していた特徴量について、Dependence plot で確認する。

- 横軸：特徴量の実際の値
- 縦軸：その特徴量の SHAP 値（予測への純粋な貢献度）
- 色：別の特徴量（交互作用の相手候補）

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ni_idx  = feature_cols.index('Ni')
tmp_idx = feature_cols.index('temp')

# Ni の dependence plot（色：temp）
sc = axes[0].scatter(X[:, ni_idx], shap_values.values[:, ni_idx],
                     c=X[:, tmp_idx], cmap='RdBu_r', alpha=0.7,
                     edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=axes[0], label='temp (°C)')
axes[0].set_xlabel('Ni (%)')
axes[0].set_ylabel('SHAP value for Ni')
axes[0].set_title('Dependence plot：Ni（色：temp）')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

# temp の dependence plot（色：Ni）
sc2 = axes[1].scatter(X[:, tmp_idx], shap_values.values[:, tmp_idx],
                      c=X[:, ni_idx], cmap='RdBu_r', alpha=0.7,
                      edgecolors='k', linewidths=0.3)
plt.colorbar(sc2, ax=axes[1], label='Ni (%)')
axes[1].set_xlabel('temp (°C)')
axes[1].set_ylabel('SHAP value for temp')
axes[1].set_title('Dependence plot：temp（色：Ni）')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.show()
```

今回のデモデータには `temp × Ni` の交互作用項を組み込んでいるため、「Ni が多い（赤）かつ temp が高い点ほど SHAP 値が大きい」というパターンが見えるはずだ。

**Dependence plot はいつ使うか：** Beeswarm で赤青が混在している特徴量が複数あるときに、交互作用の相手候補を絞るために使う。Beeswarm で綺麗に赤右・青左と分かれていれば、Dependence plot で掘り下げる優先度は下がる。

---

## 7. 局所解釈：Waterfall plot（オプション）

Waterfall plot は、**特定の1サンプルについて「なぜこの予測値になったか」を説明したいとき**に使う局所解釈ツールだ。

全サンプルを毎回確認するものではない。「この実験だけ予測が大きく外れた理由を確認したい」「この高性能サンプルの要因を実験者に説明したい」という場面で使う。

:::message
**Waterfall と Beeswarm の関係**

「Beeswarm は全サンプルの Waterfall を一枚にまとめたもの」という理解は正しい。Waterfall は1サンプルの縦棒グラフ、Beeswarm はその全サンプル版の散布図だ。全体傾向の把握には Beeswarm で十分であり、Waterfall を最初に見せてから Beeswarm に移るという順番には合理的な理由はない。コードや報告書で Waterfall → Beeswarm → Dependence の順で書かれているケースが多いが、「まず Beeswarm → 気になる特徴量を Dependence → 必要に応じて Waterfall」の順が実務的には自然だ。
:::

```python
idx_high = np.argmax(y_arr)
print(f"サンプル#{idx_high}：実測値={y_arr[idx_high]:.1f} MPa, "
      f"予測値={rf.predict(X[idx_high:idx_high+1])[0]:.1f} MPa")

shap.plots.waterfall(shap_values[idx_high], max_display=10, show=False)
plt.tight_layout()
plt.show()
```

---

## 8. 落とし穴：多重共線性で SHAP 値が分散する

Ni と強く相関する特徴量 `Ni_ratio`（= Ni / Fe）を追加してモデルを再訓練すると、SHAP の重要度がどう変わるかを確認する。

```python
df['Ni_ratio'] = df['Ni'] / df['Fe']

feature_cols_mc = ['Fe', 'Ni', 'Cr', 'temp', 'time', 'cool_r', 'Ni_ratio']
X_mc = df[feature_cols_mc].values

rf_mc = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
rf_mc.fit(X_mc, y_arr)

X_mc_df = pd.DataFrame(X_mc, columns=feature_cols_mc)
shap_values_mc = shap.TreeExplainer(rf_mc)(X_mc_df, check_additivity=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plt.sca(axes[0])
shap.plots.bar(shap_values, max_display=8, show=False)
axes[0].set_title("元の6変数")

plt.sca(axes[1])
shap.plots.bar(shap_values_mc, max_display=8, show=False)
axes[1].set_title("Ni_ratio 追加後（多重共線性あり）")

plt.tight_layout()
plt.show()

print(f"Ni と Ni_ratio の相関係数: {np.corrcoef(df['Ni'], df['Ni_ratio'])[0,1]:.3f}")
```

**結果の読み方：**

Ni_ratio を追加すると、Ni の SHAP 値が Ni_ratio との間で分散し、「どちらも中程度の重要度」として見えるようになる。物理的には Ni そのものが支配因子であるはずだが、モデルは「Ni で分岐しても Ni_ratio で分岐しても同じ」として重要度を均等に割り当ててしまう。

SHAP は多重共線性を自動で解決するツールではない。事前の相関分析（Section 2）で多重共線性を確認し、特徴量を整理してからモデルを訓練することが前提となる。

**多重共線性の主な原因：**

| 要因 | 例 |
|:----|:--|
| 物理的相関 | イオン半径と電気陰性度、密度と原子量 |
| 組成の閉包制約 | Fe + Ni + Cr = 100% による構造的共線性 |
| プロセス制約 | 温度を上げると必然的に圧力が上がる装置仕様 |
| 派生変数の追加 | Ni_ratio（= Ni / Fe）など元の変数から計算した特徴量 |

多重共線性が疑われる場合の対処は以下の通り。VIF が高い変数を除く（今回なら Fe をドロップ）か、相関の強い変数群を PCA で合成変数に置き換えるか、CLR 変換で組成制約を処理してから特徴量として使うかを検討する。

---

## 9. 逆解析①：scipy.optimize による単目標最適化

SHAP でモデルの判断を確認した後、次は「目的特性を最大化する組成・プロセス条件」を求める逆解析に移る。

`scipy.optimize.minimize` を使った勾配ベースの最適化は、連続変数の探索空間で高速に収束し、組成制約（Fe + Ni + Cr = 100）のような等式制約も組み込みやすい。

```python
from scipy.optimize import minimize, Bounds

def objective(x):
    return -rf.predict(x.reshape(1, -1))[0]  # 最小化なのでマイナス

bounds = Bounds(
    lb=[50.0, 5.0,  2.0, 800.0,  1.0,  1.0],
    ub=[90.0, 40.0, 20.0, 1200.0, 10.0, 50.0]
)

constraints = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 100.0}]

# マルチスタート（10回）：局所解に陥るリスクを軽減
results = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    ni0 = rng.uniform(5, 35)
    cr0 = rng.uniform(2, 18)
    fe0 = 100 - ni0 - cr0
    x0  = np.array([fe0, ni0, cr0,
                    rng.uniform(800, 1200),
                    rng.uniform(1, 10),
                    rng.uniform(1, 50)])
    res = minimize(objective, x0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 500, 'ftol': 1e-9})
    if res.success:
        results.append((res.fun, res.x))

results.sort(key=lambda r: r[0])
best_val, best_x = results[0]
print(f"予測最大引張強度: {-best_val:.1f} MPa")
```

**最適解が訓練データ範囲内かを必ず確認する：**

```python
train_min = df[feature_cols].min().values
train_max = df[feature_cols].max().values
in_range = (best_x >= train_min) & (best_x <= train_max)

for feat, val, lo, hi, ok in zip(feature_cols, best_x, train_min, train_max, in_range):
    flag = "✓" if ok else "⚠️ 範囲外"
    print(f"  {feat:8s}: {val:7.2f}  [{lo:.2f}, {hi:.2f}]  {flag}")
```

デモでの実行結果では、`temp=1190.50` が訓練データ最大値 1189.20 をわずかに超え、「⚠️ 範囲外」と表示された。アルゴリズムが上限ぎりぎりの外挿領域を最適解として提案した典型例だ。

:::message
**RF 逆解析の限界**

ランダムフォレストは学習データ範囲内の内挿には強いが、訓練データ範囲の外では不確実な予測をする。上限・下限の `bounds` を厳しく制限していても、最適化アルゴリズムは「モデルが最大値を返す点」を見つけるだけで、それが実際に高性能かどうかは保証しない。RF 逆解析で得た最適条件は「出発点の仮説」として扱い、GPR の不確実性（次節）と合わせて判断することを推奨する。
:::

---

## 10. 逆解析②：GPR の σ で外挿リスクを制御する

GPR（ガウス過程回帰）は予測値（μ）と不確実性（σ）を同時に出力できる。訓練データから遠い領域ほど σ が大きくなる性質を利用して、「不確実性の高い外挿領域を避けながら最適化する」ことができる。

**ペナルティ方式：** 目的関数に `κ × σ` を追加し、不確実性の高い領域に対してコストをかける。

```python
def gpr_objective(x, kappa=2.0):
    x_sc = scaler.transform(x.reshape(1, -1))
    mu_pred, sigma_pred = gpr.predict(x_sc, return_std=True)
    return -(mu_pred[0] - kappa * sigma_pred[0])  # μ - κσ を最大化

results_gpr = {}
for kappa in [0.0, 1.0, 2.0]:
    best_runs = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        ni0 = rng.uniform(5, 35)
        cr0 = rng.uniform(2, 18)
        fe0 = 100 - ni0 - cr0
        x0  = np.array([fe0, ni0, cr0,
                        rng.uniform(800, 1200),
                        rng.uniform(1, 10),
                        rng.uniform(1, 50)])
        res = minimize(lambda x: gpr_objective(x, kappa), x0,
                       method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 500})
        if res.success:
            x_sc = scaler.transform(res.x.reshape(1, -1))
            mu, sig = gpr.predict(x_sc, return_std=True)
            best_runs.append((mu[0], sig[0], res.x))
    if best_runs:
        best_runs.sort(key=lambda r: -(r[0] - kappa * r[1]))
        results_gpr[kappa] = best_runs[0]

print(f"{'κ':>5} | {'予測μ (MPa)':>12} | {'不確実性σ':>10} | {'Ni (%)':>8} | {'temp (°C)':>10}")
for kappa, (mu_val, sig_val, x_opt) in results_gpr.items():
    print(f"{kappa:>5.1f} | {mu_val:>12.1f} | {sig_val:>10.3f} | {x_opt[1]:>8.1f} | {x_opt[3]:>10.1f}")
```

**κ パラメータの解釈：**

| κ | 意味 | 用途 |
|:--|:----|:----|
| 0.0 | 予測μのみ最大化（大胆） | モデルの性能上限を確認する探索的用途 |
| 1.0 | μ − σ を最大化 | 中程度のリスク許容 |
| 2.0 | μ − 2σ を最大化（保守的） | 信頼できる領域内での最適化 |

:::message
**κ を変えても最適解が変わらない場合の解釈**

今回のデモでは、κ を 0.0 → 2.0 に変えても最適解が Ni 上限・temp 上限に張り付く結果になった。これは訓練データ範囲の境界域が最高性能域であるためで、GPR σ 制御だけでは外挿を防げていない。実務では `bounds` をより保守的に設定する（訓練データ最大値の 90% 程度を上限にするなど）か、適用領域チェック（AD チェック）を別途実装することを検討する。
:::

---

## まとめ：SHAP = Why、逆解析 = Where to next

| 手法 | 役割 | 主な用途 |
|:----|:----|:-------|
| Beeswarm plot | 大域解釈：モデル全体でどの特徴量が重要か | まず全体傾向を把握する起点 |
| Dependence plot | 非線形・交互作用の可視化 | Beeswarm で赤青混在の特徴量を深掘り |
| Waterfall plot | 局所解釈：特定サンプルの要因分解 | 実験者に個別サンプルを説明する場面で使用 |
| VIF + 相関分析 | 多重共線性の事前チェック | モデル訓練前・SHAP 解釈前の前処理として実施 |
| scipy.optimize | 単目標逆解析（組成制約あり） | 最適条件の初期候補探索 |
| GPR + σ制御 | 外挿リスクを抑えた逆解析 | 信頼できる領域内での実験候補提案 |

「モデルの精度が出た」の次に「モデルが何を学んでいるか」を確認するステップとして、Beeswarm と VIF はほぼ常に実施する。逆解析は GPR の不確実性と組み合わせることで、外挿領域への過信を抑えながら実験候補を提案できる。

SHAP で示せる非線形性や交互作用の発見、逆解析での最適条件候補——これらはあくまで「仮説」だ。ドメイン知識と第一原理計算・過去文献との照合を経て、初めて「次の実験でやること」として信頼に足るものになる。

**次回：** 今回は単一目的（引張強度の最大化）だったが、「強度と延性のトレードオフを同時に最適化する」多目標最適化（pymoo / NSGA-II）と、ベイズ最適化フレームワーク（PHYSBO）での実験候補提案を扱う予定だ。

---

[^shap]: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
[^shap_tree]: Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.
[^multicollinearity]: 多重共線性と SHAP 値の分散については VIF（分散拡大係数）によるモニタリングが有効。VIF > 10 が一般的な除外目安：Montgomery, D. C., et al. *Introduction to Linear Regression Analysis* (5th ed.).
[^gpr]: ガウス過程回帰の理論と実装：Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
[^scipy_opt]: scipy.optimize.minimize の SLSQP 法と等式制約：https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
[^coda]: 組成データ分析（CoDA）と CLR 変換：Aitchison, J. (1982). The Statistical Analysis of Compositional Data. *Journal of the Royal Statistical Society*, Series B, 44(2), 139-177.
[^inverse]: 材料科学における逆解析の実践的レビュー：Zunger, A. (2018). Inverse design in search of materials with target functionalities. *Nature Chemistry*, 10(6), 579-582.
