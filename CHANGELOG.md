# Changelog

## v1.3 — Multi-Objective Inverse Design (2026-07)
- Added multi-objective GA (Pareto optimization) incorporating hERG, DILI, Caco-2 alongside affinity
- Compared single-objective vs multi-objective GA: affinity-safety trade-off analysis
- Added Figure 4: affinity vs ADMET risk scatter plots
- Submitted all three preprints (EGFR-Inverse, Polyinverse, Battery-AI) to ChemRxiv

## v1.2 — ADMET & Docking Validation (2026-06)
- ADMET profiling: 5 endpoints (Caco-2, hERG, DILI, CYP3A4, BBB) via TDC + Random Forest
- AutoDock Vina docking validation (PDB: 4WKQ, 3UG2)
- Cross-docking 2x2 matrix: revealed non-covalent scoring limitation for covalent-mechanism candidates
- Research Notes 04, 05 added
- arXiv draft v1 completed (10 pages, 3 figures, 4 tables)

## v1.1 — T790M Mutant Extension (2026-06)
- Added L858R/T790M dataset (1,941 compounds) from ChemRxiv bioactivity scan
- Quantified data scarcity effect: R degraded from 0.74 to 0.40
- Pseudo-labeling (PI1M-style, w=0.3) recovered T790M model to R=0.70
- T790M GA inverse design: 13 candidates with spontaneous acrylamide warhead emergence
- Research Notes 01, 02, 03 added

## v1.0 — Wild-type EGFR Pipeline (2026-05)
- ChEMBL data collection: 17,623 wild-type EGFR compounds
- AttentiveFP GNN: Test R=0.736, RMSE=0.890
- Genetic Algorithm inverse design: 16 novel candidates (pChEMBL > 10.0)
- Drug-likeness filters: SA Score ≤ 4.0, QED ≥ 0.4, Lipinski Rule of Five
- Streamlit + FastAPI demo deployed to HuggingFace Spaces
