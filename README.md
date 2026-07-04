# EGFR-Inverse

> **Data Scarcity in Mutant-Specific Drug Design:**
> Graph Neural Networks, Pseudo-Labeling, and Inverse Design for Drug-Resistant EGFR T790M

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-red)](https://pyg.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023-green)](https://www.rdkit.org/)
[![ChEMBL](https://img.shields.io/badge/ChEMBL-CHEMBL203-purple)](https://www.ebi.ac.uk/chembl/)
[![Demo](https://img.shields.io/badge/HF_Live_Demo-EGFR--Inverse-blue)](https://huggingface.co/spaces/Ethan-Im/EGFR-Inverse)

---

## Overview

EGFR-Inverse is an end-to-end AI pipeline for inverse molecular design targeting **wild-type EGFR** and the clinically prevalent **L858R/T790M drug-resistant mutant** — the most common mechanism of acquired resistance to first- and second-generation EGFR tyrosine kinase inhibitors.

**Central research question:**
> *Does data scarcity in mutant-specific bioactivity datasets degrade not just affinity prediction performance, but the reliability of AI-generated drug candidates — and can pseudo-labeling close this gap?*

This project is a direct extension of [Polyinverse](https://github.com/Ethan-Im/polyinverse), applying the same inverse design methodology from polymer materials science to drug discovery.

ChEMBL Data → AttentiveFP GNN → Pseudo-Labeling (T790M) → Genetic Algorithm → Novel Candidates
                                                                 → ADMET Screening
                                                → AutoDock Vina

---

## Key Results

### Model Performance (Test Set)

| Model | Train n | RMSE | R | R² |
|-------|---------|------|---|-----|
| Wild-type EGFR | 14,098 | 0.890 | **0.736** | 0.534 |
| T790M (real only) | 1,552 | 0.920 | 0.400 | 0.155 |
| **T790M (+pseudo-labeling)** | 13,900 | **0.717** | **0.698** | 0.487 |

**Finding:** Data scarcity (9× fewer T790M samples) degraded model reliability from R=0.74 to R=0.40. PI1M-style pseudo-labeling with 12,348 wild-type compounds (loss weight=0.3) recovered performance to R=0.70.

### Inverse Design

| Candidate set | n | Predicted pChEMBL | Drug-likeness |
|---|---|---|---|
| Wild-type GA candidates | 16 | > 10.0 | SA ≤ 4.0, QED ≥ 0.4, Lipinski ✅ |
| T790M GA candidates | 13 | 6.75–7.45 | SA ≤ 4.0, QED ≥ 0.4, Lipinski ✅ |

**Notable finding:** Acrylamide warhead motifs (`C=CC(=O)N-`) emerged spontaneously in T790M candidates — consistent with the known covalent binding mechanism of 3rd-generation EGFR-TKIs (e.g. osimertinib).

### Docking Validation (AutoDock Vina)

| Candidate origin | Native receptor | Cross receptor | Note |
|---|---|---|---|
| WT candidates (n=5) | -7.99 kcal/mol (WT) | -7.90 kcal/mol (T790M) | Weakly WT-selective |
| T790M candidates (n=5) | -8.08 kcal/mol (T790M) | -8.78 kcal/mol (WT) | Not T790M-selective† |

†Cross-docking revealed T790M candidates do not show non-covalent selectivity — attributed to AutoDock Vina's inability to model covalent warhead binding (Cys797). This highlights a validation gap for AI-driven covalent inhibitor design.

### ADMET Profiling (TDC Benchmarks)

All 29 candidates screened across 5 endpoints (Caco-2, hERG, DILI, CYP3A4, BBB) using Random Forest models (hERG AUC=0.805, DILI AUC=0.869, BBB AUC=0.867). No candidate passed all primary safety filters — consistent with known kinase inhibitor scaffold hERG/DILI liability. Results motivate multi-objective GA design incorporating ADMET constraints.

---

## Research Notes

Detailed reasoning, hypotheses, and negative results documented in [`research_notes/`](research_notes/):

| Note | Topic |
|------|-------|
| [01](research_notes/01_wt_vs_t790m_comparison.md) | Wild-type vs T790M model comparison |
| [02](research_notes/02_ga_wt_vs_t790m.md) | GA inverse design: covalent warhead emergence |
| [03](research_notes/03_pseudo_labeling_t790m.md) | Pseudo-labeling recovers T790M performance |
| [04](research_notes/04_docking_validation.md) | AutoDock Vina docking validation |
| [05](research_notes/05_cross_docking_covalent_limitation.md) | Cross-docking reveals non-covalent scoring limitations |

---

## Pipeline

### Phase 1 — Data Collection ✅
- Source: ChEMBL (CHEMBL203), IC50/Ki, pChEMBL values
- Wild-type: 20,039 raw → 17,623 processed
- T790M: 4,991 records → 1,941 unique (L858R/T790M double mutant)

### Phase 2 — AttentiveFP GNN ✅
- 7-dim node features, 3-dim edge features
- 100 epochs, Adam + ReduceLROnPlateau
- Wild-type: Test R=0.736 | T790M (+pseudo): Test R=0.698

### Phase 3 — Pseudo-Labeling ✅
- 12,348 wild-type compounds pseudo-labeled by T790M model
- Weighted MSE loss (real=1.0, pseudo=0.3)
- R recovered from 0.40 → 0.70

### Phase 4 — GA Inverse Design ✅
- Population=100, Generations=50
- Filters: SA Score ≤ 4.0, QED ≥ 0.4, Lipinski Rule of Five
- 16 WT + 13 T790M candidates generated

### Phase 5 — Validation ✅
- ADMET: TDC benchmarks, Random Forest (5 endpoints)
- Docking: AutoDock Vina v1.2.7 (PDB: 4WKQ, 3UG2)
- Cross-docking: full 2×2 matrix (exhaustiveness=8, 32)

### Phase 6 — Demo & Paper ✅
- Live demo: [HF Spaces](https://huggingface.co/spaces/Ethan-Im/EGFR-Inverse)
- Technical report: [`paper/egfr_inverse_draft_v1.pdf`](paper/egfr_inverse_draft_v1.pdf)

---

## Project Structure

EGFR-Inverse/
├── data/
│   ├── raw/                        # ChEMBL raw data
│   └── processed/                  # train/val/test splits (WT + T790M)
├── src/
│   ├── data_collection.py          # ChEMBL API query
│   ├── dataset.py                  # SMILES to PyG graph
│   ├── model.py                    # AttentiveFP GNN
│   ├── train.py                    # Wild-type training
│   ├── train_t790m.py              # T790M training
│   ├── train_t790m_pseudo.py       # Pseudo-labeling training
│   ├── ga_inverse_design.py        # GA (wild-type)
│   ├── ga_inverse_design_t790m.py  # GA (T790M)
│   ├── admet_screening.py          # TDC ADMET profiling
│   ├── run_docking.py              # AutoDock Vina
│   └── run_cross_docking.py        # Cross-docking
├── docking/                        # Receptor/ligand structures + results
├── models/                         # Model checkpoints
├── results/                        # GA candidates, docking scores, ADMET
├── research_notes/                 # Versioned research notes (01-05)
├── paper/                          # LaTeX manuscript + figures
├── app_api.py                      # FastAPI web demo
└── README.md

---

## Quick Start

```bash
git clone https://github.com/Ethan-Im/EGFR-Inverse.git
cd EGFR-Inverse
conda create -n egfr-inverse python=3.10 -y
conda activate egfr-inverse
pip install torch torch-geometric rdkit pandas scipy fastapi uvicorn meeko

# Run demo locally
uvicorn app_api:app --host 0.0.0.0 --port 8080
```

---

## Related Projects

| Project | Description |
|---------|-------------|
| [Polyinverse](https://github.com/Ethan-Im/polyinverse) | GNN-based polymer property prediction & inverse design (PI1M pseudo-labeling) |
| [Battery-AI](https://github.com/Ethan-Im/Battery-Ai) | Ionic conductivity prediction for solid-state electrolytes |

---

## Citation
Im, E. (2026). Data Scarcity in Mutant-Specific Drug Design:
Graph Neural Networks, Pseudo-Labeling, and Inverse Design
for Drug-Resistant EGFR T790M. Technical Report.

---

## Author

**Ethan Im** — Independent AI researcher, computational drug discovery & molecular machine learning.

[![GitHub](https://img.shields.io/badge/GitHub-Ethan--Im-black)](https://github.com/Ethan-Im)
[![HuggingFace](https://img.shields.io/badge/HF_Demo-EGFR--Inverse-blue)](https://huggingface.co/spaces/Ethan-Im/EGFR-Inverse)
