import sys
sys.path.insert(0, "src")
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from dataset import EGFRDataset
from model import EGFRAffinityModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cpu")

def evaluate_model(model_path, test_csv, label):
    model = EGFRAffinityModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_ds = EGFRDataset(test_csv)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(batch.y.cpu().view(-1).numpy().tolist())

    preds, targets = np.array(preds), np.array(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r, _ = pearsonr(preds, targets)
    r2 = float(1 - np.sum((targets - preds)**2) / np.sum((targets - targets.mean())**2))

    print(label, "| n=", len(targets), "| RMSE:", round(rmse,4), "| R:", round(r,4), "| R2:", round(r2,4))
    return preds, targets, rmse, r, r2

print("=== Test Set Evaluation ===")
wt_preds, wt_targets, wt_rmse, wt_r, wt_r2 = evaluate_model(
    "models/best_model.pt", "data/processed/test.csv", "Wild-type "
)
mt_preds, mt_targets, mt_rmse, mt_r, mt_r2 = evaluate_model(
    "models/best_model_t790m.pt", "data/processed/t790m_test.csv", "T790M     "
)

# 비교 표 저장
summary = pd.DataFrame({
    "Model": ["Wild-type EGFR", "L858R/T790M"],
    "Train_size": [14098, 1552],
    "Test_size": [len(wt_targets), len(mt_targets)],
    "Test_RMSE": [wt_rmse, mt_rmse],
    "Test_R": [wt_r, mt_r],
    "Test_R2": [wt_r2, mt_r2],
})
summary.to_csv("results/wt_vs_t790m_comparison.csv", index=False)
print()
print(summary.to_string(index=False))

# 산점도 그림 (나란히)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

axes[0].scatter(wt_targets, wt_preds, alpha=0.4, s=15, color="#2b6cb0")
axes[0].plot([wt_targets.min(), wt_targets.max()], [wt_targets.min(), wt_targets.max()], "k--", lw=1)
axes[0].set_xlabel("Actual pChEMBL")
axes[0].set_ylabel("Predicted pChEMBL")
axes[0].set_title("Wild-type EGFR\nR={:.3f}, RMSE={:.3f}, n={}".format(wt_r, wt_rmse, len(wt_targets)))

axes[1].scatter(mt_targets, mt_preds, alpha=0.4, s=15, color="#c53030")
axes[1].plot([mt_targets.min(), mt_targets.max()], [mt_targets.min(), mt_targets.max()], "k--", lw=1)
axes[1].set_xlabel("Actual pChEMBL")
axes[1].set_ylabel("Predicted pChEMBL")
axes[1].set_title("L858R/T790M Mutant\nR={:.3f}, RMSE={:.3f}, n={}".format(mt_r, mt_rmse, len(mt_targets)))

plt.tight_layout()
plt.savefig("figures/wt_vs_t790m_scatter.png", dpi=150)
print()
print("저장 완료: figures/wt_vs_t790m_scatter.png")
print("저장 완료: results/wt_vs_t790m_comparison.csv")
