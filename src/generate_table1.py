import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None, None, None
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 2)
    qed_val = round(QED.qed(mol), 2)
    return mw, logp, qed_val

# 데이터 로드
wt_df = pd.read_csv("results/ga_candidates.csv")
mt_df = pd.read_csv("results/ga_candidates_t790m.csv")
mo_df = pd.read_csv("results/mo_ga_candidates.csv")

wt_df["Group"] = "WT"
mt_df["Group"] = "T790M"
mo_df["Group"] = "MO-GA"

# SMILES 컬럼명 통일
for df in [wt_df, mt_df, mo_df]:
    if "SMILES" not in df.columns:
        df.rename(columns={df.columns[0]: "SMILES"}, inplace=True)

# 그룹별 Top 2 선별 (predicted_pchembl 기준)
top_candidates = []
for df in [wt_df, mt_df, mo_df]:
    sub = df.sort_values(by="predicted_pchembl", ascending=False).head(2)
    top_candidates.append(sub)

final_df = pd.concat(top_candidates, ignore_index=True)

# RDKit 물성 계산 추가
mws, logps, qeds = [], [], []
for smi in final_df["SMILES"]:
    mw, logp, qed_val = compute_properties(smi)
    mws.append(mw)
    logps.append(logp)
    qeds.append(qed_val)

final_df["MW"] = mws
final_df["LogP"] = logps
final_df["QED"] = qeds
final_df["pChEMBL"] = final_df["predicted_pchembl"].round(2)

# 출력용 테이블 정리
cols = ["Group", "SMILES", "pChEMBL", "QED", "MW", "LogP"]
out_df = final_df[cols]

# 디렉토리 생성 및 저장
os.makedirs("paper/tables", exist_ok=True)
out_df.to_csv("paper/tables/table1_top_candidates.csv", index=False)

print("\n=== Main Table 1: Top Representative Candidates ===")
print(out_df.to_string(index=False))
print("\n저장 완료: paper/tables/table1_top_candidates.csv")
