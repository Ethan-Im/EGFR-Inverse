import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load covalent reactivity analysis results
csv_path = "results/covalent/cys797_covalent_analysis.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Result file not found: {csv_path}")

df = pd.read_csv(csv_path)

# Set figure style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
plt.rcParams.update({'font.size': 10})

# 1. Scatter Plot: Cys797 Distance vs. Attack Angle
colors = ["#2ca02c" if v == "Bioisostere Modelable" or v == "Covalent Viable (Cys797 Matched)" else "#7f7f7f" for v in df["Covalent_Viability"]]
sizes = [100 if v == "Bioisostere Modelable" or v == "Covalent Viable (Cys797 Matched)" else 40 for v in df["Covalent_Viability"]]

scatter = ax1.scatter(
    df["Cys797_SG_Distance (Å)"], 
    df["Attack_Angle (deg)"], 
    c=colors, 
    s=sizes, 
    alpha=0.8, 
    edgecolors="k"
)

# Highlight active reaction window (Distance <= 3.5 Å, Angle: 100-120 deg)
ax1.axvspan(2.5, 3.5, color="green", alpha=0.15, label="Reactive Window (Dist <= 3.5 Å)")
ax1.axhspan(100, 120, color="blue", alpha=0.10, label="Bürgi-Dunitz Angle (100-120°)")

ax1.set_title("A) Near-Attack Conformation (NAC) to Cys797", fontweight="bold")
ax1.set_xlabel("Cys797(SG) to Warhead Carbon Distance (Å)")
ax1.set_ylabel("Nucleophilic Attack Angle (°)")
ax1.set_xlim(2.5, 9.0)
ax1.set_ylim(60, 130)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", fontsize=8)

# 2. Bar Plot: Electrophilicity Index (omega) of Top Candidates
top5_df = df.head(5)
ax2.barh(top5_df["ID"], top5_df["Electrophilicity_Index (omega)"], color="#1f77b4", edgecolor="black", alpha=0.8)
ax2.set_title("B) Electrophilicity Index (ω) of Top 5 Candidates", fontweight="bold")
ax2.set_xlabel("Electrophilicity Index (ω)")
ax2.invert_yaxis()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Figure 9: Covalent Geometric Viability and Reactivity Analysis for EGFR T790M", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()

os.makedirs("paper/figures", exist_ok=True)
out_fig = "paper/figures/fig9_covalent_viability.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight")

print(f"Saved successfully: {out_fig}")
