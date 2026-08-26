import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem

# 1. 데이터 로드 (Table 1)
df = pd.read_csv("paper/tables/table1_top_candidates.csv")

# 2. 도킹 결과 저장 폴더 생성
os.makedirs("results/docking", exist_ok=True)

print("=== Top Candidates 3D 구조 생성 및 최적화 ===")
groups = ["WT", "T790M", "MO-GA"]

for group in groups:
    # 그룹별 1위 SMILES 추출
    top1_smi = df[df["Group"] == group].iloc[0]["SMILES"]
    
    # RDKit 분자 객체 생성 및 수소 원자(Hs) 추가 (3D 구조 최적화 필수 조건)
    mol = Chem.MolFromSmiles(top1_smi)
    mol = Chem.AddHs(mol)
    
    # 3D 좌표 생성 (랜덤 시드 고정으로 재현성 확보)
    print(f"[{group}] 3D Conformer 임베딩 및 MMFF94 에너지 최적화 진행 중...")
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # MMFF94 포스필드 기반 기하구조 최적화
    AllChem.MMFFOptimizeMolecule(mol)
    
    # 파일 저장 (도킹용 SDF 및 PyMOL 시각화용 PDB)
    sdf_path = f"results/docking/{group}_Top1.sdf"
    pdb_path = f"results/docking/{group}_Top1.pdb"
    
    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()
    
    Chem.MolToPDBFile(mol, pdb_path)
    print(f"  -> 저장 완료: {sdf_path}, {pdb_path}\n")

print("=== 3D 리간드 파일 준비 완료 ===")
