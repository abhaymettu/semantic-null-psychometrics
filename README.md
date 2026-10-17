# Semantic null for self-report psychometrics: code

Code for testing whether the wording of questionnaire items, represented by sentence-embedding similarity, predicts
the covariance of item responses beyond scale membership, and for studying what such a model can and cannot identify.
Wording enters a likelihood model of the response covariance as a fixed kernel, and the test is an item-label
permutation null that reruns the whole fit, so the question is whether the kernel's item correspondence matters, not
only whether two similarity matrices correlate.

Paper: Mettu, A. (2026). *Item wording and the covariance of self-report items: a shape-alignment test, held-out items and an identification limit*. Working paper, version 1. Zenodo (DOI added on release).

- `model/`: covariance model with a fixed text-similarity kernel, a bifactor residual, a nuisance kernel and a free
  equal-loading term (`model/m1.py`, `model/kernels.py`).
- `fit/`: polychoric correlations and the fitting interface.
- `embed/`: embedding pipeline (three encoders, all-but-the-top reference correction, nuisance text features); contract in
  `embed/CONTRACT.md`. `embed/out/` holds embeddings of public-domain item text and of synthetic fixtures.
- `integration/`: analyses: the item-heldout test
  (`r3_heldout.py`, `split.py`), the level ridge (`level_sweep.py`, `ridge_invariance_bge.py`), effective n (`neff.py`),
  descriptive baseline (`m0.py`), DASS-42 driver (`dass42.py`).
- `scripts/`: fetch public sources by hash, build item tables and item-level aggregates locally.
- `tests/`: unit tests, including exact identification counterexamples (`test_c3_exchange.py`, `test_id_redteam.py`).

## What is not here, and why
The study's results are computed from the DASS-42 response file posted by the Open-Source Psychometrics Project. That
file carries no licence, so this repository contains no response data and no statistic computed from it (a SHA-256 of the host file and a git object id (SHA-1) of the rebuilt correlation matrix, are kept so a rebuild can be checked): no aggregates,
fitted outputs, figures or manuscript. The code above rebuilds the aggregates and the real-data analyses from a copy you download yourself
(`data/README.md`). Simulation and sampler-calibration code tied to that sample, and tests that check the private frozen
outputs (including the permutation-null checker, which records statistics of the real fits), are also left out.
`embed/MANIFEST.json` describes the full embedding run of the study, so it also names files not distributed here.

## What a fresh clone reproduces

- The model, the fitting code and the unit tests, including the exact identification counterexamples.
- The embedding inputs: every kernel in `embed/out/` is recomputed from the item text by `embed/pipeline.py`.
- The DASS-42 aggregates and effective n, once you download the response file yourself (`data/README.md`).

The analysis drivers keep the integrity checks of the original frozen runs. `integration/r3_heldout.py check-inputs`
verifies the correlation matrix against its git history in the original repository (`smoothing_history`), and a clone
of this repository has no such history, so the held-out run stops at that check until you remove it. The fitted
outputs, permutation nulls and figures behind the paper are not distributed (see above).

Code comments cite design decisions by number (D1 to D14b). D14 and D14b are the statistics the paper calls pi_shape
and pi_shape_groups. `integration/r3_heldout.py` also hashes the original run's protocol files, which are not in this
repository.

## Run
```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-model.txt
.venv/bin/python -m pytest -q --ignore=tests/test_embed_pipeline.py
```
`tests/test_embed_pipeline.py` needs the embedding environment (`embed/requirements.lock.txt`).

## Licence
Code: MIT (`LICENSE`). Item text, the VADER lexicon and the embedding models keep their own terms (`NOTICE.md`).
