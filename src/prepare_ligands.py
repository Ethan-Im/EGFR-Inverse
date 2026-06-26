import subprocess
import os
import pandas as pd

os.makedirs("docking/ligands", exist_ok=True)

def smiles_to_pdbqt(smiles, name, out_dir):
    smi_path = f"{out_dir}/{name}.smi"
    pdb_path = f"{out_dir}/{name}.pdb"
    pdbqt_path = f"{out_dir}/{name}.pdbqt"
    
    with open(smi_path, "w") as f:
        f.write(f"{smiles}\t{name}\n")
    
    # SMILES → 3D PDB (수소 추가 + 3D 좌표 생성)
    cmd1 = ["obabel", smi_path, "-O", pdb_path, "--gen3d", "--best", "-h"]
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    if not os.path.exists(pdb_path):
        print(f"  PDB 변환 실패: {name}")
        return None
    
    # PDB → PDBQT
    cmd2 = ["obabel", pdb_path, "-O", pdbqt_path, "--partialcharge", "gasteiger"]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if os.path.exists(pdbqt_path):
        print(f"  완료: {name}")
        return pdbqt_path
    else:
        print(f"  PDBQT 변환 실패: {name}")
        return None

# Wild-type 후보 상위 5개
print("=== Wild-type GA 후보 ===")
wt_df = pd.read_csv("results/ga_candidates.csv").head(5)
for i, row in wt_df.iterrows():
    smiles_to_pdbqt(row["SMILES"], f"wt_cand_{i+1:02d}", "docking/ligands")

# T790M 후보 상위 5개
print("=== T790M GA 후보 ===")
mt_df = pd.read_csv("results/ga_candidates_t790m.csv").head(5)
for i, row in mt_df.iterrows():
    smiles_to_pdbqt(row["SMILES"], f"t790m_cand_{i+1:02d}", "docking/ligands")

print()
print("완료:", os.listdir("docking/ligands"))
