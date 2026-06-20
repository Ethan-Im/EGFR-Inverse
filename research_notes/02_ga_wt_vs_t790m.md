# Research Note 02 — GA Inverse Design: Wild-type vs T790M

**Date:** 2026-06-(today)
**Author:** Ethan Im

## Bug Found & Fixed

Initial T790M GA run produced 0 candidates. Diagnosis showed the seed
pool itself had a reasonable pass rate (25/200, 12.5%) against the
drug-likeness filter (SA<=4.0, QED>=0.4, Lipinski), so the issue was
in the GA loop: the original implementation only checked the top-20
survivors each generation for filter-passing candidates, and used a
fragile `population[k] == smi` index lookup. With a low pass rate,
top-20-only sampling frequently missed all passing candidates across
50 generations.

Fix: check the *entire* scored population each generation for filter
passes (not just top-20), and deduplicate via a dict keyed by SMILES.
This resolved the issue immediately (13 candidates recovered).

## Results: Wild-type GA vs T790M GA

| | Wild-type | T790M |
|---|---|---|
| Candidates generated | 16 | 13 |
| Predicted pChEMBL range | 10.08 - 11.12 | 6.75 - 7.45 |
| Mean predicted pChEMBL | ~10.6 | ~7.2 |

## Key Observation: Covalent Warhead Motif

Most top T790M candidates contain an acrylamide warhead
(`C=CC(=O)N-`), which is the same reactive group used by real
3rd-generation EGFR-TKIs (osimertinib class) to form a covalent bond
with Cys797 near the T790M gatekeeper pocket. This was not hand-coded
into the GA — it emerged from the GNN's learned structure-activity
relationship on the T790M training data. This is a meaningful
qualitative signal that the model captured a real mechanistic feature
of mutant-selective inhibitors, despite its lower R-squared.

## Important Caveat

Absolute pChEMBL values are NOT directly comparable between the two
models. The T790M model has substantially lower test R (0.40 vs
0.74), meaning its predictions carry more uncertainty. The gap in
predicted potency (10.6 vs 7.2) likely reflects a mix of:
(a) genuine differences in chemical space difficulty for the mutant,
(b) the T790M model's lower confidence / regression-to-mean behavior
    noted in Research Note 01.

This is the central methodological finding so far: data scarcity in
mutant-specific affinity models propagates into inverse design,
producing candidates whose fitness scores should be trusted less than
the wild-type run's. Any claim of "candidate quality" must be
qualified by model confidence, not just predicted pChEMBL.

## Next Steps

- [ ] Quantify candidate quality more rigorously (e.g. ensemble
      uncertainty, ensemble of T790M models) rather than relying on
      single-model point estimates.
- [ ] Explore pseudo-labeling / data augmentation for T790M model
      (PI1M-style strategy from Polyinverse) to test whether R
      improves and whether GA candidate scores shift accordingly.
- [ ] Preliminary docking (AutoDock Vina) on top candidates from both
      runs to sanity-check the covalent warhead hypothesis.

## Files

- GA script: `src/ga_inverse_design_t790m.py`
- Candidates: `results/ga_candidates_t790m.csv` (T790M),
  `results/ga_candidates.csv` (wild-type)
