import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
from rdkit import Chem

def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        int(atom.IsInRing()),
    ]

def bond_features(bond):
    bt = bond.GetBondTypeAsDouble()
    return [
        bt,
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

def smiles_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [feat, feat]
    if len(edge_index) == 0:
        return None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class EGFRDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        df = pd.read_csv(csv_path).dropna(subset=["canonical_smiles", "pchembl_value"])
        self.graphs = []
        skipped = 0
        for _, row in df.iterrows():
            g = smiles_to_graph(row["canonical_smiles"], float(row["pchembl_value"]))
            if g:
                self.graphs.append(g)
            else:
                skipped += 1
        print(f"그래프 변환 완료: {len(self.graphs)}개 (스킵: {skipped}개)")

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]
