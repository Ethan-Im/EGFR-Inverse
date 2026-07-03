import sys
sys.path.insert(0, "src")
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from dataset import EGFRDataset
from model import EGFRAffinityModel

device = torch.device("cpu")

def get_preds(model_path, csv_path):
    model = EGFRAffinityModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    ds = EGFRDataset(csv_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(batch.y.cpu().view(-1).numpy().tolist())
    return np.array(preds), np.array(targets)

models_info = [
    ("models/best_model.pt",            "data/processed/test.csv",       "Wild-type EGFR",          "#2563eb"),
    ("models/best_model_t790m.pt",      "data/processed/t790m_test.csv", "T790M (real only)",        "#dc2626"),
    ("models/best_model_t790m_pseudo.pt","data/processed/t790m_test.csv","T790M (+pseudo-labeling)", "#16a34a"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.patch.set_facecolor("white")

for ax, (mpath, cpath, label, color) in zip(axes, models_info):
    preds, targets = get_preds(mpath, cpath)
    r, _ = pearsonr(preds, targets)
    r2 = float(1 - np.sum((targets-preds)**2) / np.sum((targets-targets.mean())**2))
    rmse = float(np.sqrt(np.mean((preds-targets)**2)))

    ax.scatter(targets, preds, alpha=0.35, s=12, color=color, linewidths=0)
    lims = [min(targets.min(), preds.min())-0.2, max(targets.max(), preds.max())+0.2]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual pChEMBL", fontsize=11)
    ax.set_ylabel("Predicted pChEMBL", fontsize=11)
    ax.set_title(label, fontsize=11, fontweight="bold", color=color)
    ax.text(0.05, 0.95, f"R = {r:.3f}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("AttentiveFP GNN: Predicted vs Actual pChEMBL (Test Set)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("paper/figures/fig2_scatter.pdf", bbox_inches="tight", dpi=200)
plt.savefig("paper/figures/fig2_scatter.png", bbox_inches="tight", dpi=200)
print("저장 완료: paper/figures/fig2_scatter.pdf/png")
