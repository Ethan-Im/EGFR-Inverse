import sys
sys.path.insert(0, "src")
import torch
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from torch_geometric.loader import DataLoader
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from torch_geometric.data import Batch
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

def predict_affinity_batch(model, device, smiles_list):
    graphs, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi, 0.0)
        if g:
            graphs.append(g)
            valid_idx.append(i)
    result = {i: None for i in range(len(smiles_list))}
    if not graphs:
        return result
    batch = Batch.from_data_list(graphs).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1).cpu().numpy()
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
    return random.choice([smi1, smi2])

def run_ga(seed_smiles, model, device, pop_size=100, n_gen=50, top_k=16):
    population = seed_smiles[:pop_size]
    best_dict = {}  # smiles -> score, dedup으로 관리

    for gen in range(1, n_gen + 1):
        scores = predict_affinity_batch(model, device, population)
        valid_scored = [(population[i], scores[i]) for i in range(len(population)) if scores[i] is not None]
        valid_scored.sort(key=lambda x: x[1], reverse=True)

        top_n = max(10, pop_size // 2)
        survivors = [s for s, _ in valid_scored[:top_n]]

        top_scores = [s for _, s in valid_scored[:5]]
        n_pass = sum(1 for smi, _ in valid_scored if drug_filter(smi))
        print("Gen", str(gen).zfill(3), "| Top5:", [round(s,3) for s in top_scores], "| Filter pass this gen:", n_pass)

        # 전체 population에서 필터 통과 후보 수집 (survivors로 제한하지 않음)
        for smi, sc in valid_scored:
            if smi not in best_dict and drug_filter(smi):
                best_dict[smi] = sc

        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.choices(survivors, k=2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    best_results = sorted(best_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return best_results

if __name__ == "__main__":
    import pandas as pd
    device = torch.device("cpu")
    model = EGFRAffinityModel().to(device)
    model.load_state_dict(torch.load("models/best_model_t790m.pt", map_location=device))

    df = pd.read_csv("data/processed/t790m_train.csv")
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    df = df.sort_values("pchembl_value", ascending=False)
    seed_smiles = df["canonical_smiles"].head(200).tolist()

    print("T790M GA 역설계 시작... (시드:", len(seed_smiles), "개)")
    results = run_ga(seed_smiles, model, device, pop_size=100, n_gen=50, top_k=16)

    print()
    print("=== T790M 최종 후보 화합물 ===")
    for i, (smi, score) in enumerate(results):
        print(str(i+1).zfill(2), "| pChEMBL:", round(score,3), "|", smi)

    out_df = pd.DataFrame(results, columns=["SMILES", "predicted_pchembl"])
    out_df.to_csv("results/ga_candidates_t790m.csv", index=False)
    print()
    print("저장 완료: results/ga_candidates_t790m.csv (", len(results), "개 )")
