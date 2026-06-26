import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation

os.makedirs("docking/ligands", exist_ok=True)

def smiles_to_pdbqt(smiles, name, out_dir):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  유효하지 않은 SMILES: {name}")
            return None
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            result = AllChem.EmbedMolecule(mol)
        if result != 0:
            print(f"  3D 생성 실패: {name}")
            return None
        AllChem.MMFFOptimizeMolecule(mol)

        preparator = MoleculePreparation()
        preparator.prepare(mol)
        pdbqt_string = preparator.write_pdbqt_string()

        pdbqt_path = f"{out_dir}/{name}.pdbqt"
        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string)
        print(f"  완료: {name}")
        return pdbqt_path
    except Exception as e:
        print(f"  에러 {name}: {e}")
        return None

print("=== Wild-type GA 후보 ===")
wt_df = pd.read_csv("results/ga_candidates.csv").head(5)
for i, row in wt_df.iterrows():
    smiles_to_pdbqt(row["SMILES"], f"wt_cand_{i+1:02d}", "docking/ligands")

print("=== T790M GA 후보 ===")
mt_df = pd.read_csv("results/ga_candidates_t790m.csv").head(5)
for i, row in mt_df.iterrows():
    smiles_to_pdbqt(row["SMILES"], f"t790m_cand_{i+1:02d}", "docking/ligands")

print("완료!")
