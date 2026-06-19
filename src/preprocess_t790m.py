import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/egfr_t790m_classified.csv")
df = df[df["mutation_category"] == "Double_T790M_L858R"].copy()

print("L858R/T790M 원본:", len(df))

df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
df = df[df["pchembl_value"] > 0]

df_grouped = df.groupby("canonical_smiles", as_index=False).agg(
    pchembl_value=("pchembl_value", "mean"),
    molecule_chembl_id=("molecule_chembl_id", "first"),
    n_assays=("pchembl_value", "count")
)

print("중복 제거 후 (고유 화합물):", len(df_grouped))
pmin = df_grouped["pchembl_value"].min()
pmax = df_grouped["pchembl_value"].max()
pmean = df_grouped["pchembl_value"].mean()
print("pChEMBL 분포: min=", round(pmin,2), "max=", round(pmax,2), "mean=", round(pmean,2))

train, temp = train_test_split(df_grouped, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print("Train:", len(train), "Val:", len(val), "Test:", len(test))

train.to_csv("data/processed/t790m_train.csv", index=False)
val.to_csv("data/processed/t790m_val.csv", index=False)
test.to_csv("data/processed/t790m_test.csv", index=False)
df_grouped.to_csv("data/processed/t790m_processed.csv", index=False)

print("저장 완료: data/processed/t790m_train/val/test/processed.csv")
