import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# 데이터 로드
wt_df = pd.read_csv("results/ga_candidates.csv")
mt_df = pd.read_csv("results/ga_candidates_t790m.csv")
mo_df = pd.read_csv("results/mo_ga_candidates.csv")

wt_df["Group"] = "WT"
mt_df["Group"] = "T790M"
mo_df["Group"] = "MO-GA"

# SMILES 통일 및 데이터 병합
for df in [wt_df, mt_df, mo_df]:
    if "SMILES" not in df.columns:
        df.rename(columns={df.columns[0]: "SMILES"}, inplace=True)

all_df = pd.concat([wt_df, mt_df, mo_df], ignore_index=True)

# 박스플롯 시각화 설정
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.rcParams.update({'font.size': 11})

metrics = [
    ("predicted_pchembl", "Predicted pChEMBL", "Higher is Better"),
    ("qed", "QED (Drug-likeness)", "Higher is Better"),
    ("dili_risk", "DILI Risk Score", "Lower is Better")
]

colors = ["#4C72B0", "#DD8452", "#55A868"]

for idx, (col, title, note) in enumerate(metrics):
    ax = axes[idx]
    if col in all_df.columns:
        data_to_plot = [all_df[all_df["Group"] == g][col].dropna() for g in ["WT", "T790M", "MO-GA"]]
        ax.boxplot(data_to_plot, labels=["WT", "T790M", "MO-GA"], patch_artist=True,
                   boxprops=dict(facecolor="#EAEAF2", color="#4C4C4C"),
                   medianprops=dict(color="#C44E52", linewidth=2))
        ax.set_title(f"{title}\n({note})", fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
os.makedirs("paper/figures", exist_ok=True)
output_path = "paper/figures/fig6_admet_profile.png"
plt.savefig(output_path, dpi=300)
print(f"저장 완료: {output_path}")
