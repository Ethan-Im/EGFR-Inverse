import subprocess
import os

def prepare_receptor(pdb_path, out_path):
    # 물(HOH), 리간드 제거하고 단백질만 남기기
    with open(pdb_path) as f:
        lines = f.readlines()
    
    clean_lines = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            if line.startswith("END"):
                clean_lines.append(line)
            continue
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname in ["HOH", "WAT", "SO4", "GOL", "EDO", "PEG"]:
                continue
            # 리간드는 제거 (EGFR 결합 리간드 포함)
            continue
        clean_lines.append(line)
    
    tmp_path = pdb_path.replace(".pdb", "_clean.pdb")
    with open(tmp_path, "w") as f:
        f.writelines(clean_lines)
    print(f"정제 완료: {tmp_path} ({len(clean_lines)} lines)")
    
    # obabel로 PDBQT 변환
    cmd = ["obabel", tmp_path, "-O", out_path, "--partialcharge", "gasteiger"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(out_path):
        print(f"PDBQT 변환 완료: {out_path}")
    else:
        print(f"변환 실패: {result.stderr}")
    return tmp_path

prepare_receptor("docking/structures/egfr_wt.pdb", "docking/structures/egfr_wt.pdbqt")
prepare_receptor("docking/structures/egfr_t790m.pdb", "docking/structures/egfr_t790m.pdbqt")
