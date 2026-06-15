# EGFR-Inverse

AI-driven inverse drug design pipeline for EGFR inhibitors.

## Overview

EGFR-Inverse is a research project exploring an AI-based inverse design workflow for discovering potential EGFR inhibitor candidates.

The project combines:

- ChEMBL bioactivity data
- RDKit molecular processing
- Graph Neural Network based affinity prediction
- Genetic Algorithm based molecular optimization

The long-term goal is to build an end-to-end pipeline:

Data → Prediction Model → Inverse Molecular Design → Candidate Evaluation

---

## Project Roadmap

### Phase 1 — Data Collection & Processing ✅

- Target: EGFR
- ChEMBL Target ID: CHEMBL203
- Activity types:
  - IC50
  - Ki
- Standardization:
  - pChEMBL values
  - SMILES validation
  - Duplicate removal

Current dataset:

- Raw records: 20,039
- Processed samples: 17,623

---

### Phase 2 — Affinity Prediction Model

Coming next:

- AttentiveFP Graph Neural Network
- Molecular graph representation
- EGFR binding affinity prediction

---

### Phase 3 — Inverse Molecular Design

Planned:

- Genetic Algorithm optimization
- RDKit molecular mutation/crossover
- SA Score filtering
- Lipinski rule filtering
- QED evaluation

---

### Phase 4 — Validation & Demo

Planned:

- Molecular docking
- Visualization
- Streamlit demo

---

## Project Structure


---

## Current Status

Phase 1 completed.

Next milestone:
Build EGFR affinity prediction model using graph neural networks.

---

## Author

Ethan Im

