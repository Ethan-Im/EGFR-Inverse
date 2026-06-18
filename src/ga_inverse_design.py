import sys
sys.path.insert(0, "src")
import torch
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import RWMol, AllChem, Descriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from torch_geometric.loader import DataLoader
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from torch_geometric.data import Batch

# SA Score
from rdkit.Contrib.SA_Score import sascorer

ATOMS = ["C", "N", "O", "F", "S", "Cl", "Br"]

def lipinski(mol):
    mw  = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)
    return mw <= 500 and hbd <= 5 and hba <= 10 and logp <= 5

def drug_filter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        sa = sascorer.calculateScore(mol)
        qed = QED.qed(mol)
    except:
        return False
    return sa <= 4.0 and qed >= 0.4 and lipinski(mol)

def predict_affinity(model, device, smiles_list):
    graphs = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi, 0.0)
        if g:
            graphs.append(g)
            valid_idx.append(i)
    if not graphs:
        return {i: 0.0 for i in range(len(smiles_list))}
    batch = Batch.from_data_list(graphs).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1).cpu().numpy()
    result = {i: 0.0 for i in range(len(smiles_list))}
    for idx, pred in zip(valid_idx, preds):
        result[idx] = float(pred)
    return result

def mutate(smiles):
    mol = Chem.RWMol(Chem.MolFromSmiles(smiles))
    if mol is None:
        return smiles
    try:
        op = random.choice(["add_atom", "change_atom", "remove_atom"])
        if op == "change_atom" and mol.GetNumAtoms() > 1:
            idx = random.randint(0, mol.GetNumAtoms() - 1)
            atom = mol.GetAtomWithIdx(idx)
            new_sym = random.choice(ATOMS)
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_sym))
        elif op == "add_atom" and mol.GetNumAtoms() < 40:
            new_idx = mol.AddAtom(Chem.Atom(random.choice(ATOMS)))
            anchor = random.randint(0, mol.GetNumAtoms() - 2)
            mol.AddBond(anchor, new_idx, Chem.BondType.SINGLE)
        elif op == "remove_atom" and mol.GetNumAtoms() > 5:
            idx = random.randint(0, mol.GetNumAtoms() - 1)
            if mol.GetAtomWithIdx(idx).GetDegree() == 1:
                mol.RemoveAtom(idx)
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi if Chem.MolFromSmiles(smi) else smiles
    except:
        return smiles

def crossover(smi1, smi2):
    # fragment 교환 (단순 버전: 둘 중 하나 반환)
    return random.choice([smi1, smi2])

def run_ga(seed_smiles, model, device, pop_size=100, n_gen=50, top_k=16):
    population = seed_smiles[:pop_size]
    best_results = []

    for gen in range(1, n_gen + 1):
        # fitness 평가
        scores = predict_affinity(model, device, population)
        scored = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 상위 50% 선택
        top_n = max(10, pop_size // 2)
        survivors = [population[i] for i, _ in scored[:top_n]]

        # 로그
        top_scores = [s for _, s in scored[:5]]
        print(f"Gen {gen:03d} | Top5 pChEMBL: {[round(s,3) for s in top_scores]}")

        # 다음 세대 생성
        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.choices(survivors, k=2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

        # 필터링해서 best 후보 저장
        for smi in survivors[:20]:
            if drug_filter(smi) and smi not in [r[0] for r in best_results]:
                sc = scores[[k for k, v in scores.items() if population[k] == smi][0]] if False else predict_affinity(model, device, [smi])[0]
                best_results.append((smi, sc))

    # 최종 top_k
    best_results = sorted(best_results, key=lambda x: x[1], reverse=True)[:top_k]
    return best_results

if __name__ == "__main__":
    import pandas as pd
    device = torch.device("cpu")
    model = EGFRAffinityModel().to(device)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))

    # 시드: train 데이터에서 상위 활성 화합물
    df = pd.read_csv("data/processed/train.csv")
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    df = df.sort_values("pchembl_value", ascending=False)
    seed_smiles = df["canonical_smiles"].head(200).tolist()

    print("GA 역설계 시작...")
    results = run_ga(seed_smiles, model, device, pop_size=100, n_gen=50, top_k=16)

    print("\n=== 최종 후보 화합물 ===")
    for i, (smi, score) in enumerate(results):
        print(f"{i+1:02d} | pChEMBL: {score:.3f} | {smi}")

    out_df = pd.DataFrame(results, columns=["SMILES", "predicted_pchembl"])
    out_df.to_csv("results/ga_candidates.csv", index=False)
    print("\n저장 완료: results/ga_candidates.csv")
