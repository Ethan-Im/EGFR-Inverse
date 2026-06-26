# Research Note 04 — AutoDock Vina Docking Validation

**Date:** 2026-06-26
**Author:** Ethan Im

## Objective

Preliminary docking validation of GA-generated candidates using AutoDock Vina,
to assess whether AI-predicted high-affinity compounds actually dock into the
EGFR binding pocket with reasonable binding energies.

## Setup

- Wild-type receptor: PDB 4WKQ (EGFR + IRE ligand, 2.1Å)
- T790M receptor: PDB 3UG2 (EGFR L858R/T790M + IRE ligand, 2.5Å)
- Ligand preparation: RDKit ETKDGv3 3D conformer + Meeko PDBQT conversion
- Box center: derived from co-crystallized IRE ligand centroid
  - WT:   X=1.5,  Y=194.2, Z=20.4
  - T790M: X=-0.2, Y=49.4,  Z=20.0
- Box size: 25 x 25 x 25 Å
- Exhaustiveness: 8, num_modes: 5
- Top 5 candidates from each GA run tested (10 total)

## Results

| Ligand | Receptor | Vina Score (kcal/mol) |
|--------|----------|----------------------|
| t790m_cand_04 | T790M | -8.310 |
| t790m_cand_02 | T790M | -8.155 |
| wt_cand_04 | Wild-type | -8.124 |
| wt_cand_05 | Wild-type | -8.098 |
| wt_cand_01 | Wild-type | -8.078 |
| wt_cand_03 | Wild-type | -7.929 |
| t790m_cand_05 | T790M | -7.907 |
| t790m_cand_03 | T790M | -7.899 |
| t790m_cand_01 | T790M | -7.601 |
| wt_cand_02 | Wild-type | -7.593 |

## Observations

1. All 10 candidates showed negative binding energies (-7.6 to -8.3 kcal/mol),
   confirming that AI-generated candidates physically dock into the EGFR binding
   pocket. This is a meaningful sanity check: GA candidates are not random
   molecules but structurally coherent EGFR-binding candidates.

2. T790M-targeted candidates showed slightly stronger binding to the T790M
   receptor (best: -8.310) vs wild-type candidates on the WT receptor
   (best: -8.124). This suggests the GA successfully generated mutant-selective
   molecules, consistent with the covalent warhead finding in Research Note 02.

3. For reference, known EGFR inhibitors dock at approximately:
   - Gefitinib: ~-9.0 kcal/mol
   - Erlotinib: ~-8.5 kcal/mol
   - Osimertinib: ~-9.5 kcal/mol
   Our candidates (-7.6 to -8.3) are in a reasonable range for preliminary
   candidates, though below clinical-grade inhibitors.

## Caveats

- Docking scores are a rough proxy for binding affinity; they do not account
  for entropic effects, solvation, or induced fit.
- Exhaustiveness=8 is a quick scan; higher exhaustiveness (32+) would give
  more reliable results.
- Cross-docking (WT candidates into T790M receptor and vice versa) was not
  performed — this would be a valuable next experiment to quantify
  mutant-selectivity more rigorously.

## Next Steps

- [ ] Cross-docking: dock WT candidates into T790M receptor and vice versa
      to quantify selectivity index
- [ ] ADMET screening of top docking candidates
- [ ] arXiv technical report draft

## Files

- Docking scripts: src/prepare_receptors.py, src/prepare_ligands_meeko.py, src/run_docking.py
- Results: docking/results/docking_scores.csv
- Receptor structures: docking/structures/egfr_wt.pdbqt, egfr_t790m.pdbqt
