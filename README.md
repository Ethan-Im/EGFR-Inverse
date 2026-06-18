# EGFR-Inverse

> AI-driven inverse drug design pipeline for EGFR inhibitors

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023-green)](https://www.rdkit.org/)
[![ChEMBL](https://img.shields.io/badge/ChEMBL-CHEMBL203-purple)](https://www.ebi.ac.uk/chembl/)

---

## Overview

EGFR-Inverse is an end-to-end AI pipeline for discovering novel EGFR inhibitor candidates.  
Starting from ChEMBL bioactivity data, the pipeline trains an **AttentiveFP Graph Neural Network** to predict binding affinity, then runs a **Genetic Algorithm** to inverse-design new molecules with high predicted potency and drug-likeness.

This project is a direct extension of [Polyinverse](https://github.com/Ethan-Im/polyinverse) — applying the same inverse design philosophy from polymer materials to drug discovery.
ChEMBL Data → Preprocessing → AttentiveFP GNN → Genetic Algorithm → Novel Candidates

---

## Results

### Phase 2 — Affinity Prediction Model

| Metric | Score |
|--------|-------|
| Test RMSE | 0.8896 |
| Test R | 0.7362 |
| Test R2 | 0.5335 |

- Model: AttentiveFP GNN
- Training set: 14,098 compounds
- Validation set: 1,761 compounds
- Test set: 1,763 compounds

### Phase 3 — Inverse Molecular Design

16 novel EGFR inhibitor candidates generated via Genetic Algorithm optimization.

| Rank | Predicted pChEMBL | SMILES |
|------|-------------------|--------|
| 1 | 11.123 | CN1CC2(C1)CN(c1cc3c(Nc4cccc(Cl)c4F)ncnc3cc1CCCl)C(=O)O2 |
| 2 | 10.934 | CN1CC2(C1)CN(c1cc3c(Nc4cccc(Cl)c4F)ncnc3cc1SCO)C(=O)O2 |
| 3 | 10.797 | CSc1cc2ncnc(Nc3cccc(Cl)c3F)c2cc1N1CC2(CN(C)C2)OC1=O |
| 4 | 10.737 | CN1CC2(CC(c3cc4c(Nc5cccc(Cl)c5F)ncnc4cc3O)C(=O)O2)C1 |
| 5 | 10.682 | CN1CC2(C1)CN(c1cc3c(Nc4cccc(Cl)c4F)ncnc3cc1Cl)C(=O)O2 |

> All 16 candidates: SA Score <= 4.0 · QED >= 0.4 · Lipinski PASS
> Full results: [results/ga_candidates.csv](results/ga_candidates.csv)

---

## Pipeline

### Phase 1 — Data Collection & Processing ✅

- Source: ChEMBL (Target ID: CHEMBL203)
- Activity types: IC50, Ki
- Standardization: pChEMBL values, SMILES validation, duplicate removal
- Raw records: 20,039 → Processed: 17,623

### Phase 2 — Affinity Prediction Model ✅

- Architecture: AttentiveFP GNN (PyTorch Geometric)
- Node features: atomic number, degree, formal charge, hybridization, aromaticity, H count, ring membership (7-dim)
- Edge features: bond type, conjugation, ring membership (3-dim)
- Training: 100 epochs, Adam optimizer, ReduceLROnPlateau scheduler

### Phase 3 — Inverse Molecular Design ✅

- Algorithm: Genetic Algorithm (population=100, generations=50)
- Mutation operators: atom substitution, atom addition, atom removal
- Fitness function: predicted pChEMBL (AttentiveFP model)
- Filters: SA Score <= 4.0, QED >= 0.4, Lipinski Rule of Five
- Output: 16 novel candidates, all pChEMBL > 10.0

### Phase 4 — Demo ✅

- Streamlit app: SMILES input → affinity prediction + drug-likeness evaluation
- GA candidates visualization with molecular structures

---

## Project Structure
EGFR-Inverse/

├── data/

│   ├── raw/                  # ChEMBL raw bioactivity data

│   └── processed/            # train / val / test splits

├── src/

│   ├── data_collection.py    # ChEMBL API query

│   ├── dataset.py            # SMILES to PyG graph conversion

│   ├── model.py              # AttentiveFP model

│   ├── train.py              # Training loop

│   └── ga_inverse_design.py  # Genetic Algorithm inverse design

├── models/

│   └── best_model.pt         # Best checkpoint

├── results/

│   └── ga_candidates.csv     # 16 novel EGFR inhibitor candidates

├── app.py                    # Streamlit demo

└── README.md

---

## Quick Start

```bash
git clone https://github.com/Ethan-Im/EGFR-Inverse.git
cd EGFR-Inverse
conda create -n egfr-inverse python=3.10 -y
conda activate egfr-inverse
pip install torch torch-geometric rdkit streamlit pandas scipy
streamlit run app.py
```

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Polyinverse](https://github.com/Ethan-Im/polyinverse) | GNN-based polymer property prediction and inverse design |
| [Battery-AI](https://github.com/Ethan-Im/Battery-Ai) | Ionic conductivity prediction for solid-state electrolytes |

---

## Author

**Ethan Im** — Independent researcher at the intersection of chemistry, materials science, and deep learning.

[![GitHub](https://img.shields.io/badge/GitHub-Ethan--Im-black)](https://github.com/Ethan-Im)
