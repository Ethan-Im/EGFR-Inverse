import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

print("EGFR 데이터 전처리 시작")

# load
df = pd.read_csv(
    "data/raw/egfr_bioactivity_raw.csv"
)

print("원본:", df.shape)

# 필요한 컬럼만 유지
df = df[
    [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "pchembl_value"
    ]
]

# 결측 제거
df = df.dropna()

print("결측 제거 후:", df.shape)


# SMILES validation
def valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


df["valid"] = df["canonical_smiles"].apply(valid_smiles)

df = df[df["valid"]]

print("SMILES 정제 후:", df.shape)


# 중복 제거
df = df.drop_duplicates(
    subset=["canonical_smiles", "pchembl_value"]
)

print("중복 제거 후:", df.shape)


# pChEMBL 숫자 변환
df["pchembl_value"] = (
    df["pchembl_value"]
    .astype(float)
)


# 저장
df.to_csv(
    "data/processed/egfr_processed.csv",
    index=False
)

print(
    "저장 완료:",
    "data/processed/egfr_processed.csv"
)

