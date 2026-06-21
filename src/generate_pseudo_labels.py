import sys
sys.path.insert(0, "src")
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from torch_geometric.data import Batch

device = torch.device("cpu")
model = EGFRAffinityModel().to(device)
model.load_state_dict(torch.load("models/best_model_t790m.pt", map_location=device))
model.eval()

wt = pd.read_csv("data/processed/egfr_processed.csv")
mt = pd.read_csv("data/processed/t790m_processed.csv")

wt_only = wt[~wt["canonical_smiles"].isin(set(mt["canonical_smiles"]))].copy()
print("의사 라벨링 대상:", len(wt_only))

# 배치로 예측
smiles_list = wt_only["canonical_smiles"].tolist()
batch_size = 64
preds = []
valid_smiles = []

for i in range(0, len(smiles_list), batch_size):
    chunk = smiles_list[i:i+batch_size]
    graphs, idx_map = [], []
    for j, smi in enumerate(chunk):
        g = smiles_to_graph(smi, 0.0)
        if g:
            graphs.append(g)
            idx_map.append(j)
    if not graphs:
        continue
    batch = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1).cpu().numpy()
    for k, j in enumerate(idx_map):
        valid_smiles.append(chunk[j])
        preds.append(float(out[k]))
    if (i // batch_size) % 20 == 0:
        print("  진행:", i, "/", len(smiles_list))

pseudo_df = pd.DataFrame({
    "canonical_smiles": valid_smiles,
    "pchembl_value": preds,
    "is_pseudo": True
})
pseudo_df.to_csv("data/processed/t790m_pseudo_labels.csv", index=False)
print()
print("의사 라벨 생성 완료:", len(pseudo_df))
print("의사 라벨 pChEMBL 분포: min=", round(pseudo_df["pchembl_value"].min(),2),
      "max=", round(pseudo_df["pchembl_value"].max(),2),
      "mean=", round(pseudo_df["pchembl_value"].mean(),2))
print("저장 완료: data/processed/t790m_pseudo_labels.csv")
