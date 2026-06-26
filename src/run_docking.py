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

for rec_name, (rec_path, cx, cy, cz) in RECEPTORS.items():
    ligands = sorted(glob.glob(f"docking/ligands/{rec_name}_cand_*.pdbqt"))
    print(f"\n=== {rec_name.upper()} receptor: {len(ligands)} ligands ===")

    for lig_path in ligands:
        lig_name = os.path.basename(lig_path).replace(".pdbqt", "")
        out_path = f"docking/results/{lig_name}_out.pdbqt"

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
            "--exhaustiveness", "8",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        best_score = None
        for line in output.split("\n"):
            stripped = line.strip()
            if stripped.startswith("1 ") or stripped.startswith("1	"):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        best_score = float(parts[1])
                        break
                    except:
                        pass

        score_str = f"{best_score:.3f}" if best_score else "N/A"
        print(f"  {lig_name}: {score_str} kcal/mol")
        results.append((lig_name, rec_name, best_score))

with open("docking/results/docking_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ligand", "receptor", "vina_score_kcal_mol"])
    for row in results:
        writer.writerow(row)

print("\n=== 최종 결과 ===")
results_valid = [(l, r, s) for l, r, s in results if s is not None]
results_valid.sort(key=lambda x: x[2])
for l, r, s in results_valid:
    print(f"  {l} ({r}): {s:.3f} kcal/mol")

print("\n저장 완료: docking/results/docking_scores.csv")
