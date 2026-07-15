import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

single = pd.read_csv("results/ga_candidates.csv")
multi  = pd.read_csv("results/mo_ga_candidates.csv")

# ADMET 스크리닝 결과 로드
admet = pd.read_csv("results/admet_screening.csv")
single_admet = admet[admet["origin"] == "WT"][["SMILES","hERG","DILI","Caco2"]].copy()
single_admet = single_admet.rename(columns={"SMILES":"SMILES"})

print("=== 단일목표 vs 멀티오브젝티브 GA 비교 ===")
print()
print(f"{'':30s} {'Single-obj GA':>15} {'Multi-obj GA':>15}")
print("-"*62)
print(f"{'후보 수':30s} {len(single):>15} {len(multi):>15}")
print(f"{'평균 pChEMBL':30s} {single['predicted_pchembl'].mean():>15.3f} {multi['predicted_pchembl'].mean():>15.3f}")
print(f"{'최고 pChEMBL':30s} {single['predicted_pchembl'].max():>15.3f} {multi['predicted_pchembl'].max():>15.3f}")
print(f"{'평균 hERG':30s} {single_admet['hERG'].mean():>15.3f} {multi['hERG'].mean():>15.3f}")
print(f"{'평균 DILI':30s} {single_admet['DILI'].mean():>15.3f} {multi['DILI'].mean():>15.3f}")
print(f"{'평균 Caco2':30s} {single_admet['Caco2'].mean():>15.3f} {multi['Caco2'].mean():>15.3f}")
print(f"{'ADMET 통과(hERG<0.5&DILI<0.5)':30s} {0:>15} {len(multi[(multi.hERG<0.5)&(multi.DILI<0.5)]):>15}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor("white")

# pChEMBL vs hERG scatter
axes[0].scatter(single_admet["hERG"], single["predicted_pchembl"],
                alpha=0.7, s=60, color="#2563eb", label="Single-obj GA", zorder=3)
axes[0].scatter(multi["hERG"], multi["predicted_pchembl"],
                alpha=0.7, s=60, color="#16a34a", marker="^", label="Multi-obj GA", zorder=3)
axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="hERG threshold")
axes[0].set_xlabel("hERG probability (lower = safer)", fontsize=11)
axes[0].set_ylabel("Predicted pChEMBL", fontsize=11)
axes[0].set_title("Affinity vs hERG Risk", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# pChEMBL vs DILI scatter
axes[1].scatter(single_admet["DILI"], single["predicted_pchembl"],
                alpha=0.7, s=60, color="#2563eb", label="Single-obj GA", zorder=3)
axes[1].scatter(multi["DILI"], multi["predicted_pchembl"],
                alpha=0.7, s=60, color="#16a34a", marker="^", label="Multi-obj GA", zorder=3)
axes[1].axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="DILI threshold")
axes[1].set_xlabel("DILI probability (lower = safer)", fontsize=11)
axes[1].set_ylabel("Predicted pChEMBL", fontsize=11)
axes[1].set_title("Affinity vs DILI Risk", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# hERG 분포 비교
axes[2].hist(single_admet["hERG"], bins=10, alpha=0.6, color="#2563eb", label="Single-obj GA")
axes[2].hist(multi["hERG"], bins=10, alpha=0.6, color="#16a34a", label="Multi-obj GA")
axes[2].axvline(x=0.5, color="red", linestyle="--", alpha=0.8, label="Safety threshold")
axes[2].set_xlabel("hERG probability", fontsize=11)
axes[2].set_ylabel("Count", fontsize=11)
axes[2].set_title("hERG Distribution Comparison", fontsize=11, fontweight="bold")
axes[2].legend(fontsize=9)
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

plt.suptitle("Single-objective vs Multi-objective GA: Affinity-Safety Trade-off",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("paper/figures/fig4_mo_ga_comparison.png", bbox_inches="tight", dpi=200)
plt.savefig("paper/figures/fig4_mo_ga_comparison.pdf", bbox_inches="tight", dpi=200)
print()
print("저장 완료: paper/figures/fig4_mo_ga_comparison.png/pdf")
