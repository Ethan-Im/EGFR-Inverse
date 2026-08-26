import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# 데이터 로드
df = pd.read_csv("paper/tables/table1_top_candidates.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plt.rcParams.update({'font.size': 10})

groups = ["WT", "T790M", "MO-GA"]
pdb_map = {"WT": "1M17", "T790M": "2JIT", "MO-GA": "1M17"}
hinge_residues = {"WT": "Met793", "T790M": "Met790 / Thr790", "MO-GA": "Met793 (Key H-Bond)"}

for idx, group in enumerate(groups):
    ax = axes[idx]
    row = df[df["Group"] == group].iloc[0]
    smi = row["SMILES"]
    pchembl = row["pChEMBL"]
    
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol, size=(400, 300))
    
    ax.imshow(img)
    ax.axis('off')
    
    title_text = f"[{group}] Top 1 Candidate\nReceptor: {pdb_map[group]}\npChEMBL: {pchembl} | Hinge: {hinge_residues[group]}"
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
os.makedirs("paper/figures", exist_ok=True)
out_path = "paper/figures/fig7_binding_modes.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"저장 완료: {out_path}")
