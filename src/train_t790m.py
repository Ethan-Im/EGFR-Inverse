import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from dataset import EGFRDataset
from model import EGFRAffinityModel
from scipy.stats import pearsonr
import numpy as np

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            preds.extend(out.cpu().detach().numpy().tolist())
            targets.extend(batch.y.cpu().view(-1).numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    r, _ = pearsonr(preds, targets)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)
    return rmse, float(r), r2

def main():
    device = torch.device("cpu")
    print("Device:", device)

    train_ds = EGFRDataset("data/processed/t790m_train.csv")
    val_ds   = EGFRDataset("data/processed/t790m_val.csv")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    model = EGFRAffinityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_rmse, val_r, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "models/best_model_t790m.pt")
            tag = " << best"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1:
            print("Epoch", str(epoch).zfill(3), "| Loss:", round(total_loss/len(train_loader), 4),
                  "| Val RMSE:", round(val_rmse, 4), "| R:", round(val_r, 4), "| R2:", round(val_r2, 4), tag)

    print("학습 완료. Best Val RMSE:", round(best_val_rmse, 4))

if __name__ == "__main__":
    main()
