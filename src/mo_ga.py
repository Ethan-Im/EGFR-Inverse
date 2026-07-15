import sys
sys.path.insert(0, "src")
import torch
import numpy as np
import random
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem
from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import Batch
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

ATOMS = ["C", "N", "O", "F", "S", "Cl", "Br"]
device = torch.device("cpu")

# 모델 로드
def load_gnn(path):
    m = EGFRAffinityModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

gnn_model = load_gnn("models/best_model.pt")

admet_models = {
    "Caco2":  joblib.load("models/admet/Caco2_rf.pkl"),
    "hERG":   joblib.load("models/admet/hERG_rf.pkl"),
    "DILI":   joblib.load("models/admet/DILI_rf.pkl"),
}

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(list(fp)).reshape(1, -1)

def predict_affinity(smiles):
    g = smiles_to_graph(smiles, 0.0)
    if g is None: return 0.0
    batch = Batch.from_data_list([g]).to(device)
    with torch.no_grad():
        return float(gnn_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).item())

def predict_admet(smiles):
    fp = smiles_to_fp(smiles)
    if fp is None:
        return {"Caco2": -6.0, "hERG": 1.0, "DILI": 1.0}
    return {
        "Caco2": float(admet_models["Caco2"].predict(fp)[0]),
        "hERG":  float(admet_models["hERG"].predict_proba(fp)[0, 1]),
        "DILI":  float(admet_models["DILI"].predict_proba(fp)[0, 1]),
    }

def is_valid(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    try:
        sa = sascorer.calculateScore(mol)
        qed = QED.qed(mol)
        mw = Descriptors.MolWt(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        logp = Descriptors.MolLogP(mol)
        return sa <= 4.0 and qed >= 0.4 and mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5
    except:
        return False

def mutate(smiles):
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    if mol is None: return smiles
    try:
        op = random.choice(["add_atom", "change_atom", "remove_atom"])
        if op == "change_atom" and mol.GetNumAtoms() > 1:
            idx = random.randint(0, mol.GetNumAtoms()-1)
            mol.GetAtomWithIdx(idx).SetAtomicNum(
                Chem.GetPeriodicTable().GetAtomicNumber(random.choice(ATOMS)))
        elif op == "add_atom" and mol.GetNumAtoms() < 40:
            new_idx = mol.AddAtom(Chem.Atom(random.choice(ATOMS)))
            mol.AddBond(random.randint(0, mol.GetNumAtoms()-2), new_idx, Chem.BondType.SINGLE)
        elif op == "remove_atom" and mol.GetNumAtoms() > 5:
            idx = random.randint(0, mol.GetNumAtoms()-1)
            if mol.GetAtomWithIdx(idx).GetDegree() == 1:
                mol.RemoveAtom(idx)
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi if Chem.MolFromSmiles(smi) else smiles
    except:
        return smiles

def run_mo_ga(seed_smiles, pop_size=100, n_gen=50):
    population = seed_smiles[:pop_size]
    pareto_front = []

    for gen in range(1, n_gen+1):
        # 평가
        scored = []
        for smi in population:
            if Chem.MolFromSmiles(smi) is None:
                continue
            aff = predict_affinity(smi)
            admet = predict_admet(smi)
            # objectives: maximize affinity, minimize hERG, minimize DILI, maximize Caco2
            # pymoo minimizes, so negate affinity and Caco2
            obj = [-aff, admet["hERG"], admet["DILI"], -admet["Caco2"]]
            scored.append((smi, obj, aff, admet))

        # 간단한 Pareto ranking (pymoo 없이 직접)
        def dominates(a, b):
            return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

        pareto = []
        for i, (smi_i, obj_i, aff_i, admet_i) in enumerate(scored):
            dominated = False
            for j, (smi_j, obj_j, _, _) in enumerate(scored):
                if i != j and dominates(obj_j, obj_i):
                    dominated = True
                    break
            if not dominated:
                pareto.append((smi_i, obj_i, aff_i, admet_i))

        # Pareto front에서 유효한 후보 수집
        for smi, obj, aff, admet in pareto:
            if is_valid(smi) and smi not in [p[0] for p in pareto_front]:
                pareto_front.append((smi, aff, admet))

        top5 = sorted(pareto, key=lambda x: x[1][0])[:5]
        top_aff = [-x[1][0] for x in top5[:3]]
        top_herg = [x[1][1] for x in top5[:3]]
        print(f"Gen {gen:03d} | Pareto: {len(pareto):3d} | Top aff: {[round(a,2) for a in top_aff]} | hERG: {[round(h,2) for h in top_herg]}")

        # 다음 세대
        survivors = [s for s, _, _, _ in pareto[:pop_size//2]]
        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p = random.choice(survivors)
            new_pop.append(mutate(p))
        population = new_pop

    return sorted(pareto_front, key=lambda x: (-x[1], x[2]["hERG"]))[:20]

if __name__ == "__main__":
    df = pd.read_csv("data/processed/train.csv")
    df = df.dropna(subset=["canonical_smiles","pchembl_value"])
    df = df.sort_values("pchembl_value", ascending=False)
    seed_smiles = df["canonical_smiles"].head(200).tolist()

    print("멀티오브젝티브 GA 시작...")
    results = run_mo_ga(seed_smiles, pop_size=100, n_gen=50)

    print()
    print("=== Pareto 최적 후보 ===")
    rows = []
    for smi, aff, admet in results:
        print(f"pChEMBL={aff:.3f} | hERG={admet['hERG']:.3f} | DILI={admet['DILI']:.3f} | Caco2={admet['Caco2']:.3f} | {smi[:50]}")
        rows.append({"SMILES": smi, "predicted_pchembl": aff,
                     "hERG": admet["hERG"], "DILI": admet["DILI"], "Caco2": admet["Caco2"]})

    out_df = pd.DataFrame(rows)
    out_df.to_csv("results/mo_ga_candidates.csv", index=False)
    print()
    print("저장 완료: results/mo_ga_candidates.csv")
    print("ADMET 필터(hERG<0.5, DILI<0.5) 통과:", len(out_df[(out_df.hERG<0.5)&(out_df.DILI<0.5)]))
