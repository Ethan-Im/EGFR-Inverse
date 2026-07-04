<div align="center">

# EGFR-Inverse

### Data Scarcity in Mutant-Specific Drug Design
### Graph Neural Networks, Pseudo-Labeling, and Inverse Design for Drug-Resistant EGFR T790M

End-to-end molecular AI pipeline for mutant-specific drug discovery using
Graph Neural Networks, semi-supervised learning, inverse molecular design,
ADMET prediction, and molecular docking.

<br>

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.x-red)](https://pytorch-geometric.readthedocs.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2023-green)](https://www.rdkit.org/)
[![ChEMBL](https://img.shields.io/badge/ChEMBL-CHEMBL203-purple)](https://www.ebi.ac.uk/chembl/)
[![Demo](https://img.shields.io/badge/HuggingFace-Live_Demo-blue)](https://huggingface.co/spaces/Ethan-Im/EGFR-Inverse)

</div>

---

# Overview

**EGFR-Inverse** is an end-to-end AI framework for inverse molecular design targeting both

- Wild-type EGFR
- Drug-resistant EGFR L858R/T790M

using graph neural networks, pseudo-labeling, genetic algorithms, ADMET prediction, and molecular docking.

The project investigates an important question in molecular machine learning:

> **Can pseudo-labeling compensate for severe data scarcity in mutant-specific drug discovery?**

This work extends the inverse design methodology developed in **Polyinverse**, adapting it from polymer materials science to computational drug discovery.

---

# Pipeline

```text
                ChEMBL Bioactivity Data
                         │
                         ▼
               Molecular Graph Generation
                         │
                         ▼
               AttentiveFP Graph Neural Network
                │                     │
                │                     │
         Wild-type EGFR         T790M EGFR
                │                     │
                └─────Pseudo Labeling─┘
                         │
                         ▼
                 Affinity Prediction
                         │
                         ▼
                Genetic Algorithm Search
                         │
                         ▼
              Novel Molecule Generation
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
      ADMET Screening          AutoDock Vina
            │                         │
            └────────────┬────────────┘
                         ▼
               Final Drug Candidates
```

---

# Research Question

The EGFR T790M mutation is the dominant mechanism of acquired resistance against first- and second-generation EGFR inhibitors.

However, mutant-specific experimental datasets are dramatically smaller than wild-type datasets.

This project studies:

> **Does data scarcity reduce not only predictive performance, but also the reliability of AI-generated drug candidates?**

and

> **Can semi-supervised pseudo-labeling recover that performance?**

---

# Key Results

## Prediction Performance

| Model | Training Samples | RMSE | Pearson R | R² |
|:------|-----------------:|------:|---------:|------:|
| Wild-type EGFR | 14,098 | 0.890 | **0.736** | 0.534 |
| T790M (Real only) | 1,552 | 0.920 | 0.400 | 0.155 |
| **T790M + Pseudo-labeling** | **13,900** | **0.717** | **0.698** | **0.487** |

### Main Finding

Using only mutant-specific experimental data reduced predictive correlation

**0.736 → 0.400**

After pseudo-labeling,

**0.400 → 0.698**

recovering nearly all lost performance.

---

# Inverse Molecular Design

Genetic Algorithm optimization generated

| Target | Candidates |
|---------|-----------|
| Wild-type EGFR | 16 |
| EGFR T790M | 13 |

All generated molecules satisfied

- Lipinski Rule of Five
- SA Score ≤ 4
- QED ≥ 0.4

Interestingly, several T790M molecules spontaneously evolved the classical

```
C=CC(=O)N
```

acrylamide warhead,

which is the same covalent motif employed by third-generation EGFR inhibitors such as **Osimertinib**.

---

# Docking Validation

AutoDock Vina validation was performed using

| Structure | PDB |
|-----------|------|
| Wild-type EGFR | 4WKQ |
| EGFR T790M | 3UG2 |

Cross-docking revealed

- WT candidates favored WT receptor
- T790M candidates were **not selectively favored** by T790M

This is expected because AutoDock Vina models only non-covalent interactions and cannot evaluate covalent bond formation with **Cys797**.

---

# ADMET Screening

29 generated molecules were evaluated on five pharmacological endpoints.

Benchmarks

- Caco-2
- BBB
- CYP3A4
- hERG
- DILI

Random Forest models achieved

| Endpoint | Performance |
|----------|-------------|
| hERG | AUC = 0.805 |
| BBB | AUC = 0.867 |
| DILI | AUC = 0.869 |

No molecule passed every safety filter,

highlighting the importance of future **multi-objective inverse design** incorporating ADMET optimization.

---

# Project Structure

```text
EGFR-Inverse
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── data_collection.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── train_t790m.py
│   ├── train_t790m_pseudo.py
│   ├── ga_inverse_design.py
│   ├── ga_inverse_design_t790m.py
│   ├── admet_screening.py
│   ├── run_docking.py
│   └── run_cross_docking.py
│
├── models
├── docking
├── results
├── research_notes
├── paper
├── app_api.py
└── README.md
```

---

# Quick Start

```bash
git clone https://github.com/Ethan-Im/EGFR-Inverse.git

cd EGFR-Inverse

conda create -n egfr-inverse python=3.10 -y

conda activate egfr-inverse

pip install \
torch \
torch-geometric \
rdkit \
pandas \
scipy \
fastapi \
uvicorn \
meeko
```

Run the API

```bash
uvicorn app_api:app --host 0.0.0.0 --port 8080
```

---

# Research Notes

| Note | Description |
|------|-------------|
| 01 | Wild-type vs T790M comparison |
| 02 | Genetic Algorithm inverse design |
| 03 | Pseudo-labeling experiments |
| 04 | Docking validation |
| 05 | Cross-docking limitations |

---

# Related Projects

| Project | Description |
|---------|-------------|
| Polyinverse | Polymer inverse design with PI1M pseudo-labeling |
| Battery-AI | Solid-state electrolyte property prediction |

---

# Citation

```text
Im, E. (2026).

Data Scarcity in Mutant-Specific Drug Design:
Graph Neural Networks,
Pseudo-Labeling,
and Inverse Design
for Drug-Resistant EGFR T790M.

Technical Report.
```

---

# Author

**Ethan Im**

Independent AI Researcher

Computational Drug Discovery

Molecular Machine Learning

- GitHub: https://github.com/Ethan-Im
- HuggingFace: https://huggingface.co/spaces/Ethan-Im/EGFR-Inverse

---
