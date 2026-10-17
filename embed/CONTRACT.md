# embed/ contract

Implements the embedding interface (contract C2 v1), plus extra sensitivity variants listed below.

## Run

```
.venv/bin/python -m embed.pipeline --itemset dass42 --items <data>/items/dass42_items.csv \
    --ref <data>/items/ipipneo300_items.csv [--instrument <instrument.json>]
```

- `--items`: C1 items CSV. `item_id` and `text` are required. `text` is the exact administered wording, and row order is item order. Other columns pass through unused.
- `--ref`: the reference sentence set for `abtt-ref` (design decision D4), in the same CSV shape. It uses IPIP-NEO-300 for DASS and DASS for IPIP. The generic sentence set D4 also names for IPIP is not built yet (open).
- `--instrument`: optional `{"instruction": "...", "choices": [...]}` for the wording variants. Empty or missing parts mean those variants are skipped. The DASS-42 online wording file used in the study is not distributed here. It holds the on-screen text of the online administration that produced the response data, with the sha256 of its sources inside. IPIP-NEO-300 runs stem only, because the online-administration wording is not in hand and the hard-copy form's labels differ.
- The environment is pinned in `embed/requirements.lock.txt` (Python 3.12.13, torch 2.14.1, transformers 5.19.0, numpy 2.5.4).

Fixture: `embed/fixtures/items_fixture.csv` (12 items), `ref_fixture.csv` (30 sentences) and `instrument_fixture.json`. All are synthetic sentences written for tests. None is text from DASS, IPIP or any published instrument, and no DASS item IDs or text appear in `embed/`.

## Models (licenses, revisions, sizes verified on the Hugging Face API, 2026-10-15)

| family key | repo @ revision | license | architecture / lineage | pooling | instruction prefix | model.safetensors |
|---|---|---|---|---|---|---|
| bge-large-en-v1.5 | BAAI/bge-large-en-v1.5 @ d4aa6901d3a41ba39fb536a557fa166f842b0e09 | MIT | BERT-large encoder, BAAI contrastive (335M) | CLS | none (card: instruction only for query-to-passage retrieval) | 1,340,616,616 B, sha256 45e19549… |
| all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 @ e8c3b32edf5434bc2275fc9bab85f82640a19130 | Apache-2.0 | MPNet-base encoder, permuted-LM pretraining, SBERT 1B-pair contrastive (110M) | token mean (special tokens included, as sentence-transformers does) | none | 437,971,872 B, sha256 78c0197b… |
| qwen3-embedding-0.6b | Qwen/Qwen3-Embedding-0.6B @ 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 | Apache-2.0 | Qwen3 decoder-only LLM, causal attention (596M) | last token (`<|endoftext|>`, id 151643, appended by the pinned tokenizer and asserted) | `Instruct: Retrieve semantically similar text.\nQuery:` (format from the model card; the task sentence is our choice for a symmetric task) | 1,191,586,416 B, sha256 0437e45c… |

- Full weight hashes are in `embed/pipeline.py:MODELS` and are asserted at load. The sha256 of every snapshot file (tokenizer, config, pooling config) is recorded in `MANIFEST.json`.
- Pooling and normalization follow each repo's `1_Pooling/config.json` and `modules.json` (with its Normalize module).
- Rejected as second or third family: E5-large-v2, GTE-large and mxbai-embed-large. All three are BERT-large encoders like BGE, so they add little independent inductive bias.
- Sources: https://huggingface.co/BAAI/bge-large-en-v1.5, https://huggingface.co/sentence-transformers/all-mpnet-base-v2, https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

**Compute and resources**
- CPU, float32. Qwen3 ships bf16 and is upcast at load.
- Batch size 1, so there is no padding and pooling never sees pad tokens.
- `torch.set_num_threads(4)`, `use_deterministic_algorithms(True)`, seed 0.
- No truncation: a text over max_length is an error.
- Models load one at a time. Measured peak RSS on the fixture is 4.05 GB (`/usr/bin/time -l`).
- The HF cache at `~/.cache/huggingface` is 2.8 GB. Fetch once with `huggingface_hub.snapshot_download(repo, revision=rev, allow_patterns=["*.json","*.txt","model.safetensors"], ignore_patterns=["*/*"])`.

## Output (`embed/out/`, tracked) and `embed/MANIFEST.json`

C2 names:
- `<itemset>__<family>__raw.npy`: float32, J × d, pooled and L2-normalized, row order = items CSV.
- `<itemset>__<family>__ref.npy`: float32 embeddings of the reference set.
- `<itemset>__<family>__S_<variant>.npy`: float64 J × J cosine Gram, unit diagonal. The C2 variants are `raw`, `abtt-item` and `abtt-ref`.
- `<itemset>_nuisance.csv`: `item_id` plus the C2 columns, plus extras.

Extras:
- `<itemset>__sensitivity.json`
- more S variants (below)
- wording variants as itemset `<itemset>.<variant>`, which carry only `raw`, `abtt-ref`, `abtt-item` and `center`

`MANIFEST.json`:
- `families`: hf_repo, revision_sha, license, dim, pooling, normalize, instruction_prefix, seed, threads, weight and snapshot hashes.
- `itemsets.<itemset>`: item_order, per-item stem sha256, input file names and hashes, text-variant templates and per-text hashes, `abtt_D` per family, token counts, nuisance methods and sources, a dense PSD check per S, and `files` (sha256 and bytes for every output).
- `conventions` and `environment`.

### Variants

D for the primary `abtt-*` variants is round(d/100), the Mu & Viswanath (2018, arXiv:1702.01417) rule: 10 for BGE and Qwen3, 8 for MPNet. D = 1 and 3 are sensitivity.

| variant | definition | caveat |
|---|---|---|
| raw | model output as the card prescribes | the mean off-diagonal is high and mostly encoder offset (fixture stem: 0.24 MPNet, 0.55 BGE, 0.75 Qwen3) |
| abtt-ref (C2/D4 primary), -d1, -d3 | reference-set mean and top-D reference PCs removed | keeps content the items share relative to the reference (fixture off-diagonal mean 0.15–0.23). The reference set is stems only, so for wording variants it does not remove the shared instruction offset |
| abtt-item, -d1, -d3 | target-item mean and top-D item PCs removed; produced only if J > D + 2 | strips content the items share. The off-diagonal mean is about −1/(J−1) by construction. On the fixture, cross-model r swings from 0.35 to 0.92 with D, so it is unstable at small J |
| center | target-item mean removed | the off-diagonal mean is about −1/(J−1), so a general factor in S is fixed by convention |
| nuis | target mean removed, residualized on the C2 nuisance columns (constant columns dropped) | removes up to 8 of J−1 dimensions. For near-isotropic embeddings it imposes the projector I−H. On the fixture, r with the projector alone is 0.81–0.86, so most of its 0.97–0.98 cross-model agreement is mechanical (`projector_only_r`; test `test_linear_corrections_impose_projector_structure_on_isotropic_embeddings`) |
| nuis-nodesir | nuis without `desirability` | proxy-sensitivity check |

Conventions:
- Geometry is computed in float64.
- "Mean" is the reference-set mean for `abtt-ref` and the target-item mean for all other corrected variants, so those S matrices change when the item set changes.
- Every representation is re-normalized to unit rows.
- Each S is checked densely: symmetry, |diag−1|, eigvalsh minimum ≥ −1e-8·J, and numerical rank. On the fixture the minimum eigenvalue over all 58 S files is −2.3e-15.

Keying: cosine has no sign. On IPIP, opposite-keyed same-facet pairs have positive cosine, above different-facet pairs. Sign-align S with key_i·key_j, or model keying separately, before treating S as predicted correlation.

Downstream use: read S, or its low-rank eigendecomposition, as C2 and D6 do. Do not regress on the raw d-dimensional columns (d = 768–1024 ≫ J). D5 enters nuisance as its own kernel from `<itemset>_nuisance.csv`. That is preferable to relying only on the pre-residualized `S_nuis`, for the mechanical reason above.

### Nuisance columns (all proxies; none is ground truth)

| column | rule / source | known failure |
|---|---|---|
| n_words | count of `[a-z]+('[a-z]+)?` tokens in the lowercased stem | |
| n_chars | len(stem), Unicode code points | |
| negation | 0/1: a token in {not, no, never, nothing, nobody, none, neither, nor, nowhere, without, cannot, hardly, barely, scarcely} or ending in `n't` | misses "unable", "dislike", "rarely"; ignores scope |
| valence | mean VADER lexicon valence over matched tokens; cjhutto/vaderSentiment@0b8040fd, MIT, sha256 ec636f88…, vendored in `embed/vendor/` | raw lexicon mean without VADER's negation rules. Fixture "I do not like being around groups of strangers." scores +1.5 from "like" |
| first_person | 0/1: a token in {i, me, my, mine, myself, i'm, i've, i'd, i'll} | |
| has_example | 0/1: `( e.g. … )`, `(eg …)`, `(for example …)`, `(such as …)` | |
| desirability | per family, cos(item, desirable anchor) − cos(item, undesirable anchor), z-scored within family and averaged over the three; anchors in `nuisance.ANCHORS` | unvalidated, and circular (derived from the embeddings it corrects). No human desirability ratings are used. Published item ratings, if the data aggregates finds a verified source, should replace it |
| negation_count, valence_hits (extra) | negation token count; number of lexicon matches | informational, not used in residualization |

## Semantic null

A global orthogonal rotation E → EQ leaves the cosine Gram unchanged (`tests/test_embed_null.py::test_global_rotation_is_not_a_null`), so it is not a null.

`embed.geometry.null_gram(G, X, perm, orthogonalize=True)` and `null_distribution(E_rep, Z, n_perm, seed, statistic, blocks=None)` implement a nuisance-aware item-correspondence permutation in the Freedman–Lane (1983) form:
- Write E = HE + R, where H projects on [1, Z].
- The null is E* = HE + M P R, with M = I−H (default) or I. It is computed from the Gram as G* = A G Aᵀ, A = H + M P (I−H).
- The permutation is drawn with numpy PCG64(seed), optionally within blocks.
- `model/kernels.py` has the continuous analogue (an item-side Haar rotation of the nuisance-residual content) and a stratified permutation. This function is the discrete, exactly tested reference for cross-checking them.

**Pairing rule.** Pass the same representation the observed statistic was computed on, for example the `abtt-ref` embeddings for an `S_abtt-ref` statistic. The identity permutation then reproduces that S (tested for raw and nuis). Comparing a null built on raw embeddings with an observed `S_nuis` statistic is invalid.

Tested invariants (`tests/test_embed_null.py`):
- The Gram-only formula equals the explicit embedding construction, in both modes.
- The index convention is pinned by a loop: row i of E* = HE[i] + R[perm[i]]. The inverse permutation gives a different S.
- With `orthogonalize=True`, Xᵀ E* = Xᵀ E and the nuisance regression coefficients are identical. The null keeps the observed linear relation to every nuisance column exactly.
- With `orthogonalize=False`, every item keeps its own HE row and receives exactly one other item's residual.
- Every null S is PSD (checked in both modes). A non-trivial permutation changes S.
- Null draws stay closer to the observed S than plain label shuffling does (mean off-diagonal r higher by more than 0.1). This is a statement about retained nuisance structure, not a proven property of p-values.
- Block permutations stay within blocks, and a seed reproduces the null distribution exactly.

What it tests: whether the alignment between item semantics and item-level structure in response data exceeds what the same geometry gives under random item correspondence, with the nuisance relation held fixed. What it does not test: S's own spectrum or general factor in isolation. A plain row shuffle leaves the spectrum exactly unchanged. This null leaves it only approximately unchanged, because of the cross terms HG(I−H)Pᵀ: on a 30-item example the top eigenvalues moved by about 1–2%. Read "how much general factor language alone produces" through its alignment with response data.

## Determinism: exact vs tolerance

- **Exact (tested, `test_rerun_is_bit_identical`).** With the same machine, venv and thread count, every file listed in `itemsets.<itemset>.files` is reproduced byte for byte: raw and reference embeddings, all S files, nuisance and sensitivity. So are all PSD-check values. `MANIFEST.json` itself also records environment strings. The test is skipped when the pinned weights are not in the local cache. Without them only the pure-numpy tests and the pooling-index unit test run.
- **Tolerance (stated, not tested).** A different CPU, BLAS build, thread count or torch version changes the float32 accumulation order. Expect |ΔS| around 1e-6, and use atol = 1e-4 on S for cross-machine comparisons. The lexical nuisance columns are exact. `desirability` inherits the embedding tolerance.
