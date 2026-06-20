# Research Note 01 — Wild-type vs T790M Model Comparison

**Date:** 2026-06-(today)
**Author:** Ethan Im

## Research Question

Does a model trained on wild-type EGFR data generalize to the
clinically important L858R/T790M resistance mutant, or does the
mutant require dedicated modeling? And how does the ~9x smaller
T790M dataset affect model reliability compared to wild-type?

## Setup

- Wild-type EGFR: AttentiveFP GNN, trained on 14,098 ChEMBL CHEMBL203
  bioactivity records (IC50/Ki, pChEMBL).
- L858R/T790M mutant: same architecture, trained on 1,552 records
  filtered from ChEMBL via assay_description keyword matching
  ("T790M" + "L858R", excluding C797S triple mutants).
- Same train/val/test split strategy (80/10/10), same hyperparameters.

## Results

| Model           | Train n | Test n | RMSE  | R     | R²    |
|-----------------|---------|--------|-------|-------|-------|
| Wild-type EGFR  | 14,098  | 1,763  | 0.890 | 0.736 | 0.534 |
| L858R/T790M     | 1,552   | 195    | 0.920 | 0.400 | 0.155 |

## Observations

1. RMSE is similar between models (0.89 vs 0.92), but R and R² drop
   sharply for the T790M model. This suggests the T790M model is
   defaulting toward the mean pChEMBL value rather than learning
   meaningful structure-activity relationships — RMSE alone would
   have hidden this; R²/R exposed it.
2. The ~9x reduction in training data (14,098 -> 1,552) appears to
   be the dominant factor, consistent with known GNN data-efficiency
   limitations on molecular property prediction.
3. This raises the central question for the next phase: does this
   affinity-prediction degradation propagate into the *inverse
   design* stage? I.e., are GA-generated T790M candidates less
   trustworthy than wild-type candidates, and can this be quantified?

## Next Steps

- [ ] Run GA inverse design using the T790M model; compare candidate
      diversity / fitness distribution against the wild-type GA run.
- [ ] Investigate data augmentation strategies for the T790M model
      (pseudo-labeling on the wild-type compound pool, following the
      PI1M approach used in Polyinverse).
- [ ] Consider whether triple mutant (T790M/L858R/C797S, n=1,119)
      should be added as a third comparison arm later.

## Files

- Model checkpoints: `models/best_model.pt`, `models/best_model_t790m.pt`
- Comparison table: `results/wt_vs_t790m_comparison.csv`
- Scatter plots: `figures/wt_vs_t790m_scatter.png`
- Data prep scripts: `src/preprocess_t790m.py`, `src/classify_mutations.py`
