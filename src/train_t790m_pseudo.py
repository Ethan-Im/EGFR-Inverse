import sys
sys.path.insert(0, "src")
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from scipy.stats import pearsonr

class WeightedEGFRDataset(Dataset):
    def __init__(self, real_csv, pseudo_csv=None, pseudo_weight=0.3):
        super().__init__()
        self.graphs = []

        real_df = pd.read_csv(real_csv).dropna(subset=["canonical_smiles", "pchembl_value"])
        for _, row in real_df.iterrows():
            g = smiles_to_graph(row["canonical_smiles"], float(row["pchembl_value"]))
            if g:
                g.weight = torch.tensor([1.0])
                self.graphs.append(g)
        n_real = len(self.graphs)

        if pseudo_csv:
            pseudo_df = pd.read_csv(pseudo_csv).dropna(subset=["canonical_smiles", "pchembl_value"])
            for _, row in pseudo_df.iterrows():
                g = smiles_to_graph(row["canonical_smiles"], float(row["pchembl_value"]))
                if g:
                    g.weight = torch.tensor([pseudo_weight])
                    self.graphs.append(g)

        print("실측:", n_real, "| 의사 라벨:", len(self.graphs) - n_real, "| 총합:", len(self.graphs))

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


def weighted_mse(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(batch.y.cpu().view(-1).numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r, _ = pearsonr(preds, targets)
    r2 = float(1 - np.sum((targets - preds)**2) / np.sum((targets - targets.mean())**2))
    return rmse, float(r), r2


def main():
    device = torch.device("cpu")
    print("Device:", device)

    train_ds = WeightedEGFRDataset(
        "data/processed/t790m_train.csv",
        "data/processed/t790m_pseudo_labels.csv",
        pseudo_weight=0.3
    )
    val_ds = WeightedEGFRDataset("data/processed/t790m_val.csv")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = EGFRAffinityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_rmse = float("inf")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            loss = weighted_mse(out, batch.y.view(-1), batch.weight.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_rmse, val_r, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "models/best_model_t790m_pseudo.pt")
            tag = " << best"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1:
            print("Epoch", str(epoch).zfill(3), "| Loss:", round(total_loss/len(train_loader),4),
                  "| Val RMSE:", round(val_rmse,4), "| R:", round(val_r,4), "| R2:", round(val_r2,4), tag)

    print()
    print("학습 완료. Best Val RMSE:", round(best_val_rmse,4))

if __name__ == "__main__":
    main()
