import pandas as pd

wt = pd.read_csv("data/processed/egfr_processed.csv")
mt = pd.read_csv("data/processed/t790m_processed.csv")

wt_smiles = set(wt["canonical_smiles"])
mt_smiles = set(mt["canonical_smiles"])

overlap = wt_smiles & mt_smiles
wt_only = wt_smiles - mt_smiles

print("Wild-type 화합물 수:", len(wt_smiles))
print("T790M 화합물 수:", len(mt_smiles))
print("겹치는 화합물 수:", len(overlap))
print("Wild-type에만 있는 화합물 (의사 라벨링 후보 풀):", len(wt_only))
