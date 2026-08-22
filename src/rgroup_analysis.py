import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.rdRGroupDecomposition import RGroupDecomposition, RGroupDecompositionParameters

def get_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return round(QED.qed(mol), 3) if mol else None

# 데이터 로드
wt_df = pd.read_csv("results/ga_candidates.csv")
mo_df = pd.read_csv("results/mo_ga_candidates.csv")

wt_df["origin"] = "WT"
mo_df["origin"] = "MO-GA"

for df in [wt_df, mo_df]:
    if "SMILES" not in df.columns:
        df.rename(columns={df.columns[0]: "SMILES"}, inplace=True)

common_cols = ["SMILES", "predicted_pchembl", "origin"]
wt_sub = wt_df[[c for c in common_cols if c in wt_df.columns]].copy()
mo_sub = mo_df[[c for c in common_cols if c in mo_df.columns]].copy()

df = pd.concat([wt_sub, mo_sub], ignore_index=True)
df["QED"] = df["SMILES"].apply(get_qed)

# Top 1 Scaffold Core
core_smiles = "O=C1OC2(CNC2)CN1c1ccc2ncnc(Nc3ccccc3)c2c1"
core_mol = Chem.MolFromSmiles(core_smiles)

matched_mols = []
valid_indices = []

for idx, row in df.iterrows():
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol and mol.HasSubstructMatch(core_mol):
        matched_mols.append(mol)
        valid_indices.append(idx)

matched_df = df.loc[valid_indices].copy().reset_index(drop=True)
print(f"Top 1 Scaffold 공유 화합물 개수: 총 {len(matched_df)}개 (WT: {sum(matched_df['origin']=='WT')}개, MO-GA: {sum(matched_df['origin']=='MO-GA')}개)")

# RGroupDecomposition 객체 생성
params = RGroupDecompositionParameters()
rg = RGroupDecomposition(core_mol, params)

added_indices = []
for idx, mol in enumerate(matched_mols):
    res = rg.Add(mol)
    if res != -1:
        added_indices.append(idx)

rg.Process()
rgroups = rg.GetRGroupsAsColumns(asSmiles=True)

r_df = pd.DataFrame(rgroups)
if "Core" in r_df.columns:
    r_df = r_df.drop(columns=["Core"])

full_r_df = pd.concat([matched_df.loc[added_indices].reset_index(drop=True), r_df], axis=1)

os.makedirs("results", exist_ok=True)
full_r_df.to_csv("results/rgroup_analysis_results.csv", index=False)

full_r_df.rename(columns={"predicted_pchembl": "pChEMBL"}, inplace=True)
r_cols = [c for c in r_df.columns]

print("\n=== R-Group 변이별 물성 요약 ===")
print("\n[WT Candidates - Top 3]")
print(full_r_df[full_r_df["origin"]=="WT"][["pChEMBL", "QED"] + r_cols].head(3).to_string(index=False))

print("\n[MO-GA Candidates - Top 3]")
print(full_r_df[full_r_df["origin"]=="MO-GA"][["pChEMBL", "QED"] + r_cols].head(3).to_string(index=False))
print("\n저장 완료: results/rgroup_analysis_results.csv")
