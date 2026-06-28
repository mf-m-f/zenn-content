---
title: "材料・化学の実験データをMLに使う前にやること——欠損・外れ値・単位不統一の実践的対処法"
emoji: "🧪"
type: "tech"
topics: ["python", "機械学習", "材料"]
published: false
---

## はじめに：研究データが「そのまま使えない」理由

マテリアルズ・インフォマティクス（MI）に取り組む人が最初につまずく場所は、モデルの選定でも精度チューニングでもなく、たいてい「データの前処理」である。

材料・化学分野の実験データには、他の分野のデータにはない難しさがある。

- **小データ**：実験コストの制約から、データが数十〜数百行しかない
- **高次元**：組成・構造・合成条件など、影響因子が多い
- **ノイズが多い**：測定機器の誤差、サンプルの不均一性、複数の実験室データの統合

この環境では、欠損値・外れ値・単位の不統一といった「データの汚れ」がモデルの予測精度に直結する。本記事では、それぞれの問題に対して実務でどう判断し、どう対処するかを整理する。
記事内の Python コードは説明用の断片です。デモデータ生成を含む実行可能版は GitHub の [code.ipynb](https://github.com/sas625efta/zenn-content/blob/main/notebooks/materials-data-cleansing/code.ipynb) を参照。（[Colab で開く](https://colab.research.google.com/github/sas625efta/zenn-content/blob/main/notebooks/materials-data-cleansing/code.ipynb)）


---

## 1. 欠損値の扱い

### なぜ生じるか

研究データの欠損は、単なる入力漏れではない。代表的な原因を整理すると以下のようになる。

- **実験中断**：サンプルの破損、機器のエラー
- **BDL（Below Detection Limit）**：測定機器の検出限界以下のため値が取れない
- **測定項目の不一致**：複数の論文やデータベースを統合した際に、測定されていない特徴量がある

統計学的には、欠損の発生メカニズムは3つに分類される。

- **MCAR**（Missing Completely At Random）：欠損が他の変数と無関係に生じる
- **MAR**（Missing At Random）：欠損の発生が、観測された他の変数に依存する
- **MNAR**（Missing Not At Random）：欠損値そのもの、または未観測の変数に依存する

材料データの欠損はMARかMNARに該当することが多い。測定限界以下の値（BDL）は典型的なMNARである。欠損のメカニズムを誤って仮定すると、補完後に擬似相関が生まれる。

### 欠損の種類：測定値か、条件変数か

補完手法を選ぶ前に、もう一つ確認すべき区分がある。**欠損しているのが「測定値」か「条件変数」か**である。

**測定値の欠損**は、硬度・導電率・引張強度などの物性が取れなかったケース。BDLや実験中断が主な原因で、補完アルゴリズムの出番になる。

**条件変数の欠損**は、合成温度・焼成時間・圧力などの実験条件が記録されていないケース。原因は異なる。「Aのデータは温度を変数として制御していたが、Bのデータは温度固定で実験しており記録がない」という状況が典型で、複数の実験群やデータベースを統合するときに頻繁に発生する。

この2種類は性格が異なるため、対処も変わってくる。

| | 測定値の欠損 | 条件変数の欠損 |
|--|------------|--------------|
| 主な原因 | BDL・実験中断・機器エラー | 実験設計の違い・条件を変数として扱っていなかった |
| 統計的分類 | MAR・MNARが多い | MCARまたはMAR（設計上の欠損）が多い |
| 補完アルゴリズム | KNN/MICE/MissForest/MatImpute | ドメイン知識による埋め戻しが有効な場合もある |
| 欠損フラグの意味 | 欠損自体にパターンがある可能性 | 「この条件を制御していない実験」という情報 |

条件変数の欠損への対処は以下の3パターンが多い。

**① ドメイン知識で埋める**  
記録がないだけで実際は標準条件が使われていた（例：特に記載がなければ室温・大気圧）場合、その値で補完する。ただし「記録しなかった」のか「その条件が存在しなかった」のかを混同しないよう注意が必要。

**② 欠損フラグを特徴量として追加する**  
「この実験ではその条件を制御変数として扱っていなかった」という事実がモデルにとって有意義な情報になりうる。欠損フラグ列（0/1）を特徴量に加えることで、実験群の違いをモデルが学習できる。

**③ 列ごと除外する、またはデータを分割する**  
特定の実験群でしか記録されていない条件変数は、全体モデルの特徴量から除外するか、その条件が揃っているデータだけを対象にサブモデルを作る判断もある。統計的な補完で埋めた数値は、実際には「測定していない」という物理的事実を消してしまう点に注意したい。

### 構造的ゼロと真の欠損を混同しない

配合・処方データ特有の問題として、**構造的ゼロ（Structural Zero）**がある。

例えば、材料A・材料B・材料Cの使用量をそれぞれ列として持つ場合を考える。

| 実験 | 材料A_量 | 材料B_量 | 材料C_量 |
|-----|---------|---------|---------|
| 実験1 | 50 | - | 30 |
| 実験2 | - | 40 | 20 |
| 実験3 | 60 | - | - |

実験1で材料Bのセルが空欄になっているのは、「材料Bを使ったが量を記録し忘れた」のではなく、「材料Bをそもそも使っていないので量は0」という意味である。これが構造的ゼロだ。

この区別が重要な理由は、補完アルゴリズムに与える影響にある。KNNやMICEをそのまま適用すると、「近い実験では材料Bを少量使っている」という相関から微小な非ゼロ値が埋め込まれ、「実際には使っていない材料がわずかに入っている」という物理的に誤ったデータが生成されてしまう。

```python
import pandas as pd
import numpy as np

# 構造的ゼロと真の欠損を分けて管理するアプローチ
# 1. どのセルが「使っていない（=0）」かをドメイン知識で定義する
#    例：各実験の配合記録から使用材料リストを取得
used_materials = {
    '実験1': ['材料A', '材料C'],
    '実験2': ['材料B', '材料C'],
    '実験3': ['材料A'],
}

# 2. 構造的ゼロを先に埋める（補完アルゴリズムの前に実施）
for exp, materials in used_materials.items():
    all_material_cols = ['材料A_量', '材料B_量', '材料C_量']
    used_cols = [m + '_量' for m in materials]
    unused_cols = [c for c in all_material_cols if c not in used_cols]
    df.loc[exp, unused_cols] = 0  # 使っていない材料は0で確定

# 3. 残った欠損（真の欠損）のみに補完アルゴリズムを適用
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

なお、材料の使用有無を「材料A を使ったか否か」という0/1のカテゴリ列として持つ設計にすれば、使用量の列に欠損は生じない。どちらの設計を選ぶかはモデルの目的次第だが、使用量の絶対値や比率がモデルに重要であれば数値列として持ち、上記の前処理を経て使うことになる。また、使用量の合計が100%（重量分率や体積分率）になる場合は、後述するCLR変換も合わせて適用する必要がある。

### 削除か補完か

数十〜数百行しかないデータで、欠損を含む行を丸ごと削除するリストワイズ削除は基本的に避ける。数パーセントの欠損であっても、学習に必要なサンプルの多様性が失われる。

**原則として補完を選ぶ**。ただし、欠損率が70%を超えており推測が不可能な特徴量は、その列ごと除外するという判断もある。

### 補完手法の選び方

| 手法 | 特徴 | 研究データへの適用基準 |
|------|------|----------------------|
| 平均値・中央値補完 | 最も単純。実装が楽 | 分布を歪め、分散を過小評価する。材料データには推奨しない |
| **KNN Imputation** | 特徴量空間で距離が近いサンプルの平均で補完 | 「組成が近い材料は物性も近い」という局所的類似性を活かせる |
| **MICE** | 各特徴量を他の全特徴量を使った回帰モデルで反復補完 | 変数間の相関構造を保持できる。材料データで高い評価を得ている |
| **MissForest** | ランダムフォレストで反復的に欠損を予測 | 非線形な関係に強い。外れ値の影響を受けにくく、スケーリング不要 |
| **MatImpute** | Extra TreesとKNNのハイブリッド（材料科学向け） | RMSE・Wasserstein距離を最小化し、データセットの相関構造の収束性（DCC）を最も高く保持することが実証されている[^matimpute]。なぜ材料データに有効かというと、局所的類似性と非線形関係の両方を捉えられるためである |
| **BDL特化型**（QRILC等） | 検出限界以下データに対し、分布の裾野をベイズ推定 | BDL由来の欠損に対して、単純な0や「検出限界の半分」で置換する歪みを防ぐ |

```python
# df はあらかじめ実験データが格納されている DataFrame とします
from sklearn.impute import KNNImputer
import pandas as pd

# 欠損フラグを特徴量として追加（欠損していること自体が有意味な情報の場合）
missing_flag = df.isnull().astype(int).add_suffix('_missing')
df_with_flag = pd.concat([df, missing_flag], axis=1)

# KNNで補完
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

補完の前後で分布（ヒストグラム）と変数間の相関係数を比較して、補完が合理的かを確認することも重要である。

### モデルごとの耐性の違い

採用するアルゴリズム自体が欠損をどう扱うかも、前処理の判断に影響する。

- **XGBoost・LightGBM**：内部で「欠損サンプルを右のノードへ送るか左へ送るか」を最適化する。欠損していること自体にパターンがある（MNAR）場合、明示的に補完しなくてもモデルが学習できる
- **通常のランダムフォレスト（scikit-learn実装）**：欠損値を許容しない。事前補完が必須
- **SVM・線形回帰・ニューラルネットワーク**：欠損があると距離計算や内積が成立しない。補完は必須

---

## 2. 外れ値の検出と対処

### 最も重要な問い：実験ミスか、新発見か

材料科学において外れ値は2種類ある。

一つは「実験ミスやノイズ」。測定機器のキャリブレーション不良、コンタミネーション、転記ミスなどが原因。これはモデルの予測限界を下げるため、検出して除外または修正する必要がある。

もう一つは「真の特異挙動」。特定の組成で突如として臨界温度が上がる超伝導体、予想外の高反応性を示す触媒など、材料探索の最大の目標がここにある。統計アルゴリズムが「異常」と判定したからといって機械的に削除すると、科学的発見を捨てることになる。

**外れ値の検出は自動化できても、その解釈はドメイン知識が不可欠。**

### 検出手法の使い分け

| 手法 | 特徴 | 適用場面 |
|------|------|---------|
| **Z-score / IQR法** | 1変数ずつ、分布から逸脱した値を検出 | 特徴量が少ない。個別の物性値の明らかな入力ミスを弾く場合 |
| **Isolation Forest** | ランダム分割を繰り返し、少ない分割で孤立するサンプルを外れ値とする | 高次元データ（化学記述子が数十〜数百）に有効。計算コストが低い |
| **LOF**（Local Outlier Factor） | 局所的な密度を比較して外れ値を判定 | データ内に複数の材料グループが混在している場合に有効 |

```python
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Step 1: まず分布を目視確認
df[['hardness', 'conductivity', 'tensile_strength']].boxplot(figsize=(10, 4))
plt.title('分布確認：箱ひげ図')
plt.show()

# Step 2: Isolation Forestで多変量の異常検知
clf = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = clf.fit_predict(X_scaled)
# -1 が外れ値、1 が正常値
outliers = df[outlier_labels == -1]
print(f"検出された外れ値: {len(outliers)} 件")
```

### 対処の選択肢

検出した外れ値への対処は4つある。

**① 除外（Trimming）**  
物理的法則に反する値（絶対温度がマイナス、組成の合計が100%を大きく超えるなど）は削除する。ただし小データでは貴重なサンプルを失うため、慎重に判断する。

**② Winsorization**  
外れ値を削除せず、分布の上下数パーセンタイル（例：1%と99%）の値でクリッピングする手法。データ数を減らさずに、極端な値による平均・分散の歪みを防げる。小データ環境において特に有効[^winsorization]。

```python
# df はあらかじめ実験データが格納されている DataFrame とします
from feature_engine.outliers import Winsorizer

winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
df_winsorized = winsorizer.fit_transform(df[feature_cols])
```

**③ ロバスト回帰 / RANSAC**  
前処理で外れ値を除去するのではなく、モデル自体に耐性を持たせる。太陽電池材料のQSPRモデル構築においてRANSACが有効であることが示されている[^ransac]。

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor

# Huber回帰：外れ値の影響を線形損失で緩和
huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train)

# RANSAC：インライア（正常値）のコンセンサスを最大化
ransac = RANSACRegressor(random_state=42)
ransac.fit(X_train, y_train)
```

**④ そのまま使う**  
ランダムフォレスト・XGBoostなどのツリー系モデルは特徴量空間を閾値で分割するため、特徴量側の外れ値への影響が限定的。目的変数に外れ値がなければ、あえて処理しないという判断も合理的。

### プロセスに専門家レビューを組み込む

検出した外れ値のリストを材料科学者・実験担当者に見せて「この値は物理的にあり得るか」を確認する工程（Expert-augmented approach）を必ずプロセスに入れる。この判断がなければ、上記の選択肢はどれも機能しない。

---

## 3. 単位・表記の不統一

### よくあるパターン

複数の論文・データベースからデータを統合する際に必ず起きる問題。

- **単位系の混在**：温度が℃とKで混在、圧力がPaとatmとTorrで入り乱れる
- **SMILESの表記ゆれ**：同じ分子でも複数の正しい表記がある（芳香環、互変異性体、プロトン化状態）
- **組成文字列のパース**：「Ni0.8Co0.15Al0.05O2」が1セルに入っており、特徴量として展開されていない

### データ辞書を最初に設計する

プロジェクト開始時に「データ辞書（Data Dictionary）」を作成し、単位と表記のルールを明文化する。材料データベースのAPI標準化コンソーシアムである **OPTIMADE**（Open Databases Integration for Materials Design）の仕様[^optimade]は、このルール設計の良い参考になる。

例として決める内容：
- エネルギー：eV
- 距離：Å（オングストローム）
- 温度：K（ケルビン）
- 化学構造：IUPAC名または正規化SMILES

### Pintで単位変換を自動化する

**Pint**ライブラリ[^pint]は数値と単位をセットで扱い、単位間の変換を自動で行う。換算係数の入力ミスを根絶できる。さらに **pint-pandas** を使うと、PandasのDataFrameの列単位で一括単位変換ができ、機械学習パイプラインへの組み込みが容易になる。例えば「pressure列全体をPaからatmに変換する」という操作が1行で書け、前処理スクリプトの可読性と保守性が上がる。

```python
# df はあらかじめ実験データが格納されている DataFrame とします
from pint import UnitRegistry
import pint_pandas

ureg = UnitRegistry()

# 摂氏をケルビンに変換（オフセット単位は Quantity で指定）
temp_C = ureg.Quantity(25, 'degC')
temp_K = temp_C.to('kelvin')
print(f"{temp_C} = {temp_K}")  # 25 °C = 298.15 K

# PandasのDataFrameに単位を適用（pint-pandas）
pint_pandas.PintType.ureg = ureg

df['pressure_atm'] = pd.array(df['pressure_Pa'], dtype="pint[Pa]").pint.to('atm').pint.m
```

### 化学構造の標準化（RDKit）

ケモインフォマティクスでは分子構造の標準化が必須。**RDKit**[^rdkit]を使って以下を自動化する。

```python
# df はあらかじめ実験データが格納されている DataFrame とします
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def standardize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 水素の整理・塩の除去・電荷の正規化
    cleanup = rdMolStandardize.Cleanup(mol)
    # 互変異性体を最安定構造に統一
    enumerator = rdMolStandardize.TautomerEnumerator()
    canonical = enumerator.Canonicalize(cleanup)
    return Chem.MolToSmiles(canonical)

df['canonical_smiles'] = df['smiles'].apply(standardize_smiles)
```

### スケーリングの選び方

特徴量の単位を揃えた後、SVM・ニューラルネットワーク・KNNなどの距離ベースモデルを使う場合はスケーリングが必要。

| スケーラー | 特徴 | 研究データへの適用基準 |
|-----------|------|----------------------|
| **StandardScaler** | 平均0、標準偏差1に変換 | 外れ値に弱い。外れ値が存在すると正常値のスケールが極端に狭くなる |
| **MinMaxScaler** | 0〜1の範囲に線形変換 | 外れ値に最も弱い。外れ値が1つあると残りが0付近に圧縮される |
| **RobustScaler** | 中央値と四分位範囲（IQR）でスケーリング | **小データで外れ値が存在するケースに最適**。外れ値の影響を受けにくい |

RobustScalerが外れ値に強い理由は、スケーリングの基準として**外れ値の影響を受けない統計量**を使うからだ。StandardScalerは平均と標準偏差を使うため、外れ値1点が両方を大きく歪める。RobustScalerは代わりに**中央値**（全データを並べたときの真ん中の値）と**四分位範囲IQR**（上位25%と下位25%の差）を使う。これらは外れ値がどれだけ極端な値を取っても変化しないため、正常値のスケールが保たれる。

```python
# X_train はあらかじめスケーリング前の特徴量行列とします
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 組成データのCLR変換（重要）

合金の組成比率や混合物のwt%など、「合計が常に100%になるデータ（組成データ）」には特殊な処理が必要。

通常のスケーリングをそのまま適用すると、「ある成分が増えれば必然的に他の成分が減る」という制約から**閉包問題（Closure Problem）**が生じ、偽の相関が発生する[^compositional]。

これを避けるために、スケーリングの前に**中心対数比変換（CLR: Centered Log-Ratio）**を施す。

```python
# df はあらかじめ実験データが格納されている DataFrame とします
import numpy as np

def clr_transform(X):
    """
    組成データ（各行の合計が1または100）にCLR変換を適用
    X: shape (n_samples, n_components)
    """
    # ゼロ値がある場合は微小値で置換（対数計算のため）
    X = np.where(X == 0, 1e-10, X)
    log_X = np.log(X)
    # 各サンプルの対数の幾何平均を引く
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean

# 例：合金の組成データ（Al, Ni, Co, Crの重量分率）
composition = df[['Al_wt%', 'Ni_wt%', 'Co_wt%', 'Cr_wt%']].values / 100
X_clr = clr_transform(composition)
```

### phr データの場合：CLR が必要か

ポリマー配合でよく使われる **phr（parts per hundred rubber/resin）** は、wt% とは性格が異なる。

wt% は「全成分の合計 = 100%」という制約を持つが、phr はゴム・樹脂を 100 として他の成分をその相対量で表すため、合計が 100 に縛られない。カーボンブラック 50 phr、可塑剤 30 phr、架橋剤 2 phr はそれぞれ独立して変化できる。**phr のまま使う限り、閉包問題は基本的に起きない。**

#### wt% → phr 変換で閉包を開放する

wt% データを持っている場合、基準成分を決めて phr に変換することで閉包問題を解消できる。

```python
# df_wt はあらかじめ wt% データが格納されている DataFrame とします
import pandas as pd

# wt% データのサンプル（合計 = 100%）
df_wt = pd.DataFrame({
    'ゴム_wt%':         [60.0, 55.0, 65.0],
    'カーボンブラック_wt%': [25.0, 27.0, 22.0],
    '可塑剤_wt%':       [12.0, 15.0, 10.0],
    '架橋剤_wt%':       [3.0,  3.0,  3.0],
})

# ゴムを基準（= 100 phr）として変換
def wt_to_phr(df, base_col):
    """
    wt% データを phr に変換する。
    base_col を 100 phr として他成分を正規化する。
    合計の制約が外れ、閉包問題が解消される。
    """
    base = df[base_col] / 100  # 割合に変換
    phr = df.div(df[base_col], axis=0) * 100
    return phr

df_phr = wt_to_phr(df_wt, 'ゴム_wt%')
print("phr 変換後（合計の制約が外れる）:")
print(df_phr)
print(f"\n行合計: {df_phr.sum(axis=1).values}")  # 100 より大きくなる
```

#### CLR との使い分け

| | wt% → phr 変換 | CLR 変換 |
|--|--------------|---------|
| 基準 | 特定成分を選ぶ（恣意的） | 幾何平均（自動・対称） |
| 直感的な解釈 | 配合技術者に馴染みやすい | 統計的には整合的だが非直感的 |
| 基準成分が変わる場合（異なるポリマー系の比較など） | 比較が困難 | 問題なく扱える |
| ポリマーブレンド比（SBR/NR = 70/30 など） | ブレンド比自体は組成データ → 別途 CLR | CLR が有効 |
| 基準成分がゼロに近い場合 | 発散する | log(0) 問題はあるが微小値補正で対応可能 |

実務的な選択基準はシンプルで、**ベースポリマーが常に同じ系統のデータ**であれば phr 変換の方が現場の感覚と合っていて解釈しやすい。一方、**異なるポリマー系を横断して比較する場合**や**ポリマーブレンド比も特徴量に含む場合**は CLR の方が無難だ。

なお、phr に変換した後にさらに対数を取ること（log-phr）も有効で、これは統計学で **ALR（Additive Log-Ratio）変換** と呼ばれる手法に相当する。線形モデルや距離ベースのアルゴリズムとの相性が改善される。


---

## まとめ：ドメイン知識なしに完結しない

3つの問題を振り返ると、共通の構造がある。

| 問題 | 自動化できること | 人間の判断が必要なこと |
|------|----------------|---------------------|
| 欠損値（測定値） | KNN/MissForest/MatImputeによる補完 | BDLか実験失敗か。欠損自体に意味があるか |
| 欠損値（条件変数） | 欠損フラグの生成 | 標準条件の推定。列を残すか除外するか |
| 外れ値 | Isolation Forest/LOFによる異常スコア算出 | 実験ミスか真の特異挙動か |
| 単位不統一 | Pintによる自動変換、RDKitによる構造標準化 | データ辞書の設計（何を正とするか） |

機械的に処理できる工程は自動化して、判断が必要な工程に時間を使う。材料科学では「データを知っている人が前処理に関わること」がモデルの品質を決める。

---

## 参考文献

[^matimpute]: Imputation of Missing Data in Materials Science through Nearest Neighbors and Iterative Predictions. *Journal of Chemical Theory and Computation*, ACS Publications. https://pubs.acs.org/doi/10.1021/acs.jctc.4c01237

[^winsorization]: Winsorization for Robust Bayesian Neural Networks. *Entropy*, MDPI, 2021. https://www.mdpi.com/1099-4300/23/11/1546

[^ransac]: RANdom SAmple Consensus (RANSAC) algorithm for material-informatics: application to photovoltaic solar cells. ResearchGate. https://www.researchgate.net/publication/317498531

[^optimade]: OPTIMADE, an API for exchanging materials data. https://www.optimade.org/

[^pint]: Pint: makes units easy. Read the Docs. https://pint.readthedocs.io/

[^rdkit]: Getting Started with the RDKit in Python. https://www.rdkit.org/docs/GettingStartedInPython.html

[^compositional]: Statistical Analysis and Interpolation of Compositional Data in Materials Science. *ACS Combinatorial Science*, 2015. https://pubs.acs.org/doi/abs/10.1021/co5001458