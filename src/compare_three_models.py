import sys
sys.path.insert(0, "src")
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader
from dataset import EGFRDataset
from model import EGFRAffinityModel

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
    return rmse, r, r2

print("=== Final Test Set Comparison ===")
wt = evaluate_model("models/best_model.pt", "data/processed/test.csv", "Wild-type        ")
mt = evaluate_model("models/best_model_t790m.pt", "data/processed/t790m_test.csv", "T790M (real only) ")
mtp = evaluate_model("models/best_model_t790m_pseudo.pt", "data/processed/t790m_test.csv", "T790M (+pseudo)   ")

summary = pd.DataFrame({
    "Model": ["Wild-type EGFR", "T790M (real only)", "T790M (+pseudo-labeling)"],
    "Train_size": [14098, 1552, 13900],
    "Test_RMSE": [wt[0], mt[0], mtp[0]],
    "Test_R": [wt[1], mt[1], mtp[1]],
    "Test_R2": [wt[2], mt[2], mtp[2]],
})
summary.to_csv("results/wt_vs_t790m_vs_pseudo_comparison.csv", index=False)
print()
print(summary.to_string(index=False))
print()
print("저장 완료: results/wt_vs_t790m_vs_pseudo_comparison.csv")
