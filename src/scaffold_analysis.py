import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None

# 데이터 로드
wt_df  = pd.read_csv("results/ga_candidates.csv")
mt_df  = pd.read_csv("results/ga_candidates_t790m.csv")
mo_df  = pd.read_csv("results/mo_ga_candidates.csv")

wt_df["origin"]  = "WT"
mt_df["origin"]  = "T790M"
mo_df["origin"]  = "MO-GA"

# SMILES 컬럼 통일
if "SMILES" not in mo_df.columns:
    mo_df = mo_df.rename(columns={mo_df.columns[0]: "SMILES"})

all_df = pd.concat([
    wt_df[["SMILES","predicted_pchembl","origin"]],
    mt_df[["SMILES","predicted_pchembl","origin"]],
    mo_df[["SMILES","predicted_pchembl","origin"]],
], ignore_index=True)

# Scaffold 추출
all_df["scaffold"] = all_df["SMILES"].apply(get_scaffold)
all_df = all_df.dropna(subset=["scaffold"])

print("=== Scaffold 분석 ===")
print(f"전체 후보: {len(all_df)}개")
print(f"고유 scaffold: {all_df['scaffold'].nunique()}개")
print()

# 그룹별 scaffold 분포
for origin in ["WT", "T790M", "MO-GA"]:
    sub = all_df[all_df["origin"] == origin]
    print(f"{origin}: {len(sub)}개 후보, {sub['scaffold'].nunique()}개 고유 scaffold")

print()

# 가장 많이 나온 scaffold Top 5
print("=== 전체 Top 5 Scaffold ===")
top_scaffolds = Counter(all_df["scaffold"]).most_common(5)
for i, (sca, cnt) in enumerate(top_scaffolds):
    origins = all_df[all_df["scaffold"]==sca]["origin"].value_counts().to_dict()
    print(f"#{i+1} (n={cnt}): {origins} | {sca[:60]}")

# 시각화 — Top 6 scaffold 구조
top6 = [s for s, _ in Counter(all_df["scaffold"]).most_common(6)]
mols = [Chem.MolFromSmiles(s) for s in top6 if Chem.MolFromSmiles(s)]
counts = [Counter(all_df["scaffold"])[s] for s in top6]
origins_list = []
for s in top6:
    o = all_df[all_df["scaffold"]==s]["origin"].value_counts().to_dict()
    origins_list.append(str(o).replace("{'","").replace("'}","").replace("', '",", "))

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=3,
    subImgSize=(320, 260),
    legends=[f"n={c} | {o}" for c, o in zip(counts, origins_list)]
)
img.save("paper/figures/fig5_scaffolds.png")
print()
print("저장 완료: paper/figures/fig5_scaffolds.png")

# scaffold 다양성 지수 (Tanimoto 기반)
from rdkit.Chem import AllChem
from rdkit import DataStructs

def diversity(smiles_list):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
    if len(fps) < 2:
        return 0.0
    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1 - np.mean(sims)  # diversity = 1 - mean similarity

print()
print("=== 화합물 다양성 지수 (1=완전 다양, 0=완전 동일) ===")
for origin in ["WT", "T790M", "MO-GA"]:
    sub = all_df[all_df["origin"]==origin]
    div = diversity(sub["SMILES"].tolist())
    print(f"  {origin}: {div:.3f}")
