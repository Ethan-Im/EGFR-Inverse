import subprocess
import os
import glob
import csv

os.makedirs("docking/results", exist_ok=True)

RECEPTORS = {
    "wt":    ("docking/structures/egfr_wt.pdbqt",     1.5, 194.2, 20.4),
    "t790m": ("docking/structures/egfr_t790m.pdbqt", -0.2,  49.4, 20.0),
}

BOX_SIZE = 25
results = []

# 모든 리간드를 모든 receptor에 도킹 (cross-docking)
ligands = sorted(glob.glob("docking/ligands/wt_cand_*.pdbqt")) + sorted(glob.glob("docking/ligands/t790m_cand_*.pdbqt"))

print(f"Cross-docking: {len(ligands)} ligands x {len(RECEPTORS)} receptors = {len(ligands)*len(RECEPTORS)} runs\n")

for lig_path in ligands:
    lig_name = os.path.basename(lig_path).replace(".pdbqt", "")
    lig_origin = "wt" if lig_name.startswith("wt_cand") else "t790m"

    for rec_name, (rec_path, cx, cy, cz) in RECEPTORS.items():
        out_path = f"docking/results/cross_{lig_name}_on_{rec_name}_out.pdbqt"

        cmd = [
            "vina",
            "--receptor", rec_path,
            "--ligand", lig_path,
            "--center_x", str(cx),
            "--center_y", str(cy),
            "--center_z", str(cz),
            "--size_x", str(BOX_SIZE),
            "--size_y", str(BOX_SIZE),
            "--size_z", str(BOX_SIZE),
            "--out", out_path,
            "--num_modes", "5",
            "--exhaustiveness", "32",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        best_score = None
        for line in output.split("\n"):
            stripped = line.strip()
            if stripped.startswith("1 ") or stripped.startswith("1\t"):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        best_score = float(parts[1])
                        break
                    except:
                        pass

        score_str = f"{best_score:.3f}" if best_score else "N/A"
        match = "native" if lig_origin == rec_name else "cross"
        print(f"  [{match:6s}] {lig_name} -> {rec_name}: {score_str} kcal/mol")
        results.append((lig_name, lig_origin, rec_name, best_score, match))

with open("docking/results/cross_docking_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ligand", "ligand_origin", "receptor", "vina_score_kcal_mol", "match_type"])
    for row in results:
        writer.writerow(row)

print("\n저장 완료: docking/results/cross_docking_scores.csv")
