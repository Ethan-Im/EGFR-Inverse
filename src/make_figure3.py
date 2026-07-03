import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

wt_df = pd.read_csv("results/ga_candidates.csv").head(6)
mt_df = pd.read_csv("results/ga_candidates_t790m.csv").head(6)

def smiles_to_img(smiles, size=(280, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.patch.set_facecolor("white")

# WT 후보 (왼쪽 2열)
for i, (_, row) in enumerate(wt_df.iterrows()):
    ax = axes[i // 2][i % 2]
    img = smiles_to_img(row["SMILES"])
    if img:
        ax.imshow(img)
    ax.set_title(f"WT #{i+1}\npChEMBL: {row['predicted_pchembl']:.2f}",
                 fontsize=9, fontweight="bold", color="#2563eb")
    ax.axis("off")

# T790M 후보 (오른쪽 2열)
for i, (_, row) in enumerate(mt_df.iterrows()):
    ax = axes[i // 2][2 + i % 2]
    img = smiles_to_img(row["SMILES"])
    if img:
        ax.imshow(img)
    ax.set_title(f"T790M #{i+1}\npChEMBL: {row['predicted_pchembl']:.2f}",
                 fontsize=9, fontweight="bold", color="#dc2626")
    ax.axis("off")

# 구분선
fig.text(0.5, 0.98, "GA-Generated EGFR Inhibitor Candidates",
         ha="center", fontsize=14, fontweight="bold")
fig.text(0.26, 0.93, "Wild-type EGFR Candidates",
         ha="center", fontsize=12, color="#2563eb", fontweight="bold")
fig.text(0.74, 0.93, "T790M-targeted Candidates",
         ha="center", fontsize=12, color="#dc2626", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("paper/figures/fig3_candidates.pdf", bbox_inches="tight", dpi=200)
plt.savefig("paper/figures/fig3_candidates.png", bbox_inches="tight", dpi=200)
print("저장 완료: paper/figures/fig3_candidates.pdf/png")
