import warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tdc.single_pred import ADME, Tox

os.makedirs("models/admet", exist_ok=True)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return list(fp)

def get_xy(df):
    X, y = [], []
    for _, row in df.iterrows():
        fp = smiles_to_fp(row["Drug"])
        if fp:
            X.append(fp)
            y.append(row["Y"])
    return np.array(X), np.array(y)

DATASETS = {
    "Caco2":  ("ADME", "Caco2_Wang",                   "regression"),
    "hERG":   ("Tox",  "hERG",                          "classification"),
    "DILI":   ("Tox",  "DILI",                          "classification"),
    "CYP3A4": ("ADME", "CYP3A4_Substrate_CarbonMangels","classification"),
    "BBB":    ("ADME", "BBB_Martini",                   "classification"),
}

print("=== ADMET 모델 학습 및 저장 ===")
for name, (cat, dset, task) in DATASETS.items():
    data = ADME(name=dset) if cat == "ADME" else Tox(name=dset)
    split = data.get_split()
    X_tr, y_tr = get_xy(split["train"])

    if task == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    model.fit(X_tr, y_tr)
    joblib.dump(model, f"models/admet/{name}_rf.pkl")
    print(f"  저장 완료: models/admet/{name}_rf.pkl")

print("완료!")
