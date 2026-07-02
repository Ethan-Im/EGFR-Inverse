import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
from tdc.single_pred import ADME, Tox

def smiles_to_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return list(fp)

def get_xy(df):
    X, y, valid_idx = [], [], []
    for i, row in df.iterrows():
        fp = smiles_to_fp(row["Drug"])
        if fp:
            X.append(fp)
            y.append(row["Y"])
            valid_idx.append(i)
    return np.array(X), np.array(y)

DATASETS = {
    "Caco2":  ("ADME", "Caco2_Wang",                  "regression",     "log cm/s",   "high = better absorption"),
    "hERG":   ("Tox",  "hERG",                         "classification", "probability","high = cardiac toxicity risk"),
    "DILI":   ("Tox",  "DILI",                         "classification", "probability","high = liver toxicity risk"),
    "CYP3A4": ("ADME", "CYP3A4_Substrate_CarbonMangels","classification","probability","high = metabolized by CYP3A4"),
    "BBB":    ("ADME", "BBB_Martini",                  "classification", "probability","high = crosses blood-brain barrier"),
}

models = {}
print("=== 모델 학습 ===")
for name, (cat, dset, task, unit, note) in DATASETS.items():
    if cat == "ADME":
        data = ADME(name=dset)
    else:
        data = Tox(name=dset)
    split = data.get_split()
    X_tr, y_tr = get_xy(split["train"])
    X_va, y_va = get_xy(split["valid"])

    if task == "regression":
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = mean_absolute_error(y_va, preds)
        print(f"  {name}: MAE={score:.3f} {unit}")
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_va)[:,1]
        score = roc_auc_score(y_va, preds)
        print(f"  {name}: AUC={score:.3f}")
    models[name] = (model, task, unit, note)

print()
print("=== GA 후보 ADMET 스크리닝 ===")

wt_df = pd.read_csv("results/ga_candidates.csv")
mt_df = pd.read_csv("results/ga_candidates_t790m.csv")
wt_df["origin"] = "WT"
mt_df["origin"] = "T790M"
all_df = pd.concat([wt_df, mt_df], ignore_index=True)
all_df["rank"] = list(range(1, len(wt_df)+1)) + list(range(1, len(mt_df)+1))

results = []
for _, row in all_df.iterrows():
    smi = row["SMILES"]
    origin = row["origin"]
    rank = row["rank"]
    fp = smiles_to_fp(smi)
    if fp is None:
        continue
    fp_arr = np.array(fp).reshape(1, -1)
    res = {"SMILES": smi, "origin": origin, "rank": rank,
           "predicted_pchembl": row["predicted_pchembl"]}
    for name, (model, task, unit, note) in models.items():
        if task == "regression":
            res[name] = round(float(model.predict(fp_arr)[0]), 3)
        else:
            res[name] = round(float(model.predict_proba(fp_arr)[0,1]), 3)
    results.append(res)

result_df = pd.DataFrame(results)

# 필터 기준
# Caco2 > -5.15 (moderate absorption)
# hERG < 0.5 (low cardiac toxicity)
# DILI < 0.5 (low liver toxicity)
result_df["Caco2_pass"] = result_df["Caco2"] > -5.15
result_df["hERG_pass"]  = result_df["hERG"]  < 0.5
result_df["DILI_pass"]  = result_df["DILI"]  < 0.5
result_df["ADMET_pass"] = result_df["Caco2_pass"] & result_df["hERG_pass"] & result_df["DILI_pass"]

result_df.to_csv("results/admet_screening.csv", index=False)

print()
print("=== 결과 요약 ===")
for origin in ["WT", "T790M"]:
    sub = result_df[result_df["origin"] == origin]
    passed = sub["ADMET_pass"].sum()
    print(f"{origin} 후보: {len(sub)}개 중 {passed}개 ADMET 통과")

print()
print("=== ADMET 통과 후보 ===")
passed_df = result_df[result_df["ADMET_pass"]]
for _, row in passed_df.iterrows():
    print(f"  [{row['origin']}#{row['rank']:02d}] pChEMBL={row['predicted_pchembl']:.3f} | Caco2={row['Caco2']} | hERG={row['hERG']} | DILI={row['DILI']} | BBB={row['BBB']}")

print()
print("저장 완료: results/admet_screening.csv")
