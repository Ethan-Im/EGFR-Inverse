import sys
sys.path.insert(0, "src")
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer

df = pd.read_csv("data/processed/t790m_train.csv")
df = df.sort_values("pchembl_value", ascending=False).head(200)

print("=== T790M 시드 화합물 200개 진단 ===\n")

pass_sa, pass_qed, pass_lip, pass_all = 0, 0, 0, 0
sa_list, qed_list = [], []

for smi in df["canonical_smiles"]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    try:
        sa = sascorer.calculateScore(mol)
        qed = QED.qed(mol)
    except:
        continue
    mw = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)
    lipinski_ok = mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5

    sa_list.append(sa)
    qed_list.append(qed)
    if sa <= 4.0: pass_sa += 1
    if qed >= 0.4: pass_qed += 1
    if lipinski_ok: pass_lip += 1
    if sa <= 4.0 and qed >= 0.4 and lipinski_ok: pass_all += 1

import numpy as np
print("SA Score: mean=", round(np.mean(sa_list),2), "min=", round(min(sa_list),2), "max=", round(max(sa_list),2))
print("QED:      mean=", round(np.mean(qed_list),2), "min=", round(min(qed_list),2), "max=", round(max(qed_list),2))
print()
print("SA<=4.0 통과:", pass_sa, "/200")
print("QED>=0.4 통과:", pass_qed, "/200")
print("Lipinski 통과:", pass_lip, "/200")
print("전체 통과:", pass_all, "/200")
