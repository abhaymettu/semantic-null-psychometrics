"""Aggregate-only artifacts (correlations, item stats, mapping checks) from the cached raw data.

Run:  uv run --with numpy==2.3.3 --with scipy==1.16.2 python -I scripts/build_aggregates.py [dass42|ipipneo300|fixtures ...]
Reads data/cache (never committed); writes data/aggregates/ and data/fixtures/. No row-level
value leaves this process: outputs are item x item matrices, per-item marginals and counts.

Coding contract (see data/README.md):
- every matrix is in ANSWERED direction: higher category = more agreement with the item text as
  worded. IPIP300.dat stores reverse-keyed items already reversed (DAT300.doc), so they are
  un-reversed here (x -> 6 - x). keyed[i, j] = key_i * key_j * answered[i, j] (exact for both
  Pearson and polychoric).
- DASS raw 1..4 is recoded to the published 0..3 scale (x - 1); IPIP stays 1..5.
"""
import csv, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtr, ndtri, owens_t

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SEMNULL_CACHE", REPO / "data" / "cache"))
ITEMS = REPO / "data" / "items"
OUT = Path(os.environ.get("SEMNULL_OUT", REPO / "data"))  # override only for rebuild checks
AGG = OUT / "aggregates"
FIX = OUT / "fixtures"


# ---------- polychoric (two-step: marginal thresholds, then ML rho per pair) ----------

def bvn_cdf(h, k, r):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation r (finite h, k).
    Owen (1956) via Owen's T; h or k exactly 0 is nudged to 1e-10 to avoid 0/0."""
    h = np.where(h == 0, 1e-10, h)
    k = np.where(k == 0, 1e-10, k)
    s = np.sqrt(1 - r * r)
    beta = np.where(h * k > 0, 0.0, 0.5)
    return 0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, (k - r * h) / (h * s)) - owens_t(k, (h - r * k) / (k * s)) - beta


def thresholds(marg):
    """marg: (..., K) category counts -> (..., K-1) normal thresholds from cumulative proportions."""
    c = np.cumsum(marg, axis=-1)[..., :-1] / marg.sum(axis=-1, keepdims=True)
    return ndtri(np.clip(c, 1e-12, 1 - 1e-12))


def cell_probs(ta, tb, r):
    """ta (P,K-1), tb (P,L-1), r (P,) -> (P,K,L) cell probabilities."""
    P, K1, L1 = ta.shape[0], ta.shape[1], tb.shape[1]
    C = np.zeros((P, K1 + 2, L1 + 2))
    C[:, 1:-1, 1:-1] = bvn_cdf(ta[:, :, None], tb[:, None, :], r[:, None, None])
    C[:, -1, 1:-1] = ndtr(tb)
    C[:, 1:-1, -1] = ndtr(ta)
    C[:, -1, -1] = 1.0
    return np.diff(np.diff(C, axis=1), axis=2)


def polychoric(tables, iters=60):
    """tables (P,K,L) counts -> rho (P,), by vectorised golden-section ML on (-0.9999, 0.9999)."""
    tables = np.asarray(tables, float)
    ta, tb = thresholds(tables.sum(2)), thresholds(tables.sum(1))

    def nll(r):
        p = np.maximum(cell_probs(ta, tb, r), 1e-300)
        return -(tables * np.log(p)).sum(axis=(1, 2))

    g = (np.sqrt(5) - 1) / 2
    lo, hi = np.full(len(tables), -0.9999), np.full(len(tables), 0.9999)
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = nll(x1), nll(x2)
    for _ in range(iters):
        left = f1 < f2  # minimum lies in [lo, x2]
        hi = np.where(left, x2, hi)
        lo = np.where(left, lo, x1)
        x1n, x2n = hi - g * (hi - lo), lo + g * (hi - lo)
        fx = nll(np.where(left, x1n, x2n))
        f2, f1 = np.where(left, f1, fx), np.where(left, fx, f2)
        x1, x2 = x1n, x2n
    return (lo + hi) / 2


# ---------- aggregation from category-index data ----------

def pair_tables(X, K, chunk=20000):
    """X (N,J) ints in 0..K-1, -1 = missing -> T (J,J,K,K) pairwise-complete bivariate counts
    (exact; float32 matmul per chunk)."""
    N, J = X.shape
    T = np.zeros((J * K, J * K))
    for s in range(0, N, chunk):
        x = X[s:s + chunk]
        O = np.zeros((len(x), J * K + 1), np.float32)  # last column absorbs missing (-1)
        O[np.arange(len(x))[:, None], np.where(x >= 0, np.arange(J) * K + x, J * K)] = 1
        O = O[:, :-1]
        T += (O.T @ O).astype(np.float64)
    return T.reshape(J, K, J, K).transpose(0, 2, 1, 3)


def correlations(T):
    J, K = T.shape[0], T.shape[2]
    iu = np.triu_indices(J, 1)
    tabs = T[iu]
    v = np.arange(K, dtype=float)
    n = tabs.sum((1, 2))
    ex, ey = (tabs.sum(2) @ v) / n, (tabs.sum(1) @ v) / n
    exx, eyy = (tabs.sum(2) @ v**2) / n, (tabs.sum(1) @ v**2) / n
    exy = np.einsum("pkl,k,l->p", tabs, v, v) / n
    pear = (exy - ex * ey) / np.sqrt((exx - ex**2) * (eyy - ey**2))
    poly = polychoric(tabs)
    out = {}
    for name, vals in (("pearson", pear), ("polychoric", poly)):
        M = np.eye(J)
        M[iu] = vals
        M.T[iu] = vals
        out[name] = M
    return out


def item_stats(X, K, ids, labels):
    rows = []
    for j, iid in enumerate(ids):
        xj = X[X[:, j] >= 0, j]
        c = np.bincount(xj, minlength=K)
        x = np.asarray(labels, float)[xj]
        row = {"item_id": iid, "n": len(x), "mean": f"{x.mean():.6f}", "sd": f"{x.std(ddof=1):.6f}"}
        row.update({f"p_{labels[a]}": f"{c[a] / len(x):.6f}" for a in range(K)})
        rows.append(row)
    return rows


def mapping_check(M, groups, keys):
    """Keyed-direction check that each item correlates most with its own scale/facet.
    Returns (n items whose own group has the highest mean r, failing indices,
    n negative keyed within-group pairs, mean keyed within-group r)."""
    Mk = M * np.outer(keys, keys)
    ug = sorted(set(groups))
    g = np.array(groups)
    own_best, fails, neg, own = 0, [], 0, []
    for j in range(len(g)):
        means = {u: Mk[j, (g == u) & (np.arange(len(g)) != j)].mean() for u in ug}
        if max(means, key=means.get) == g[j]:
            own_best += 1
        else:
            fails.append(j)
        neg += int((Mk[j, (g == g[j]) & (np.arange(len(g)) != j)] < 0).sum())
        own.append(means[g[j]])
    return own_best, fails, neg // 2, float(np.mean(own))


def offset_test(M, groups, keys, offsets=(1, 2, 5, 30, -1, -30)):
    """Falsification of the column<->item alignment: relabel item j with the group and key of item
    j+offset. A correct alignment should beat every offset (higher own-group r, fewer negatives)."""
    out = {}
    for o in offsets:
        _, _, neg, own = mapping_check(M, list(np.roll(groups, -o)), np.roll(keys, -o))
        out[str(o)] = {"mean_within_group_keyed_r": round(own, 4), "negative_within_group_keyed_pairs": neg}
    return out


# ---------- dataset loaders (row-level, in memory only) ----------

def load_dass():
    import zipfile
    with zipfile.ZipFile(CACHE / "raw" / "dass_raw_zip") as z:
        lines = z.read("DASS_data_21.02.19/data.csv").decode("utf8").splitlines()
    head = lines[0].split("\t")
    rows = np.array([l.split("\t") for l in lines[1:]])
    col = {h: i for i, h in enumerate(head)}
    A = rows[:, [col[f"Q{k}A"] for k in range(1, 43)]].astype(int)
    vcl = rows[:, [col["VCL6"], col["VCL9"], col["VCL12"]]].astype(int)
    engnat = rows[:, col["engnat"]].astype(int)
    vals = set(np.unique(A).tolist())
    assert vals <= {0, 1, 2, 3, 4}, vals
    missing = A == 0
    complete = ~missing.any(1)
    valid = (vcl == 0).all(1)
    variants = {
        "complete": complete,
        "primary": complete & valid,
        "primary_engnat": complete & valid & (engnat == 1),
    }
    counts = {
        "n_rows": int(len(A)),
        "n_incomplete_dass": int((~complete).sum()),
        "n_fake_word_checked_among_complete": int((complete & ~valid).sum()),
        "n_engnat_not1_among_primary": int((complete & valid & (engnat != 1)).sum()),
        "observed_raw_values": sorted(vals),
        "observed_vcl_fake_word_values": sorted(np.unique(vcl).tolist()),
        "observed_engnat_values": sorted(np.unique(engnat).tolist()),
        "columns_in_file": len(head),
    }
    return A - 1, missing.sum(0), variants, counts  # 0..3; -1 marks missing (never in a variant)


def load_ipip(keys):
    raw = (CACHE / "raw" / "ipip300_dat").read_bytes()
    lines = raw.split(b"\r\n")
    assert all(len(l) == 333 for l in lines), "fixed-width layout changed"
    X = np.frombuffer(b"".join(l[33:333] for l in lines), np.uint8).reshape(len(lines), 300) - ord("0")
    year = np.array([int(l[19:22]) for l in lines]) + 1900  # DAT300.doc: YEAR cols 20-22, years since 1900
    assert X.max() <= 5
    missing = X == 0
    X = np.where((keys < 0) & ~missing, 6 - X, X)  # un-reverse to answered direction
    complete = ~missing.any(1)
    per_row = np.bincount(missing.sum(1))
    counts = {"n_rows": int(len(X)), "n_rows_any_missing_item": int((~complete).sum()),
              "rows_by_n_missing_items": {str(i): int(c) for i, c in enumerate(per_row) if c},
              "observed_raw_values": sorted(np.unique(X).tolist()),
              "observed_completion_year_range": [int(year.min()), int(year.max())]}
    # complete: complete cases (primary). pairwise: every row, each pair uses rows observing both.
    return X.astype(np.int64) - 1, missing.sum(0), {"complete": complete, "pairwise": np.ones(len(X), bool)}, counts


# ---------- writing ----------

def write_matrix(M, ids, path):
    with open(path, "w", newline="") as f:
        f.write("item_id," + ",".join(ids) + "\n")
        for iid, row in zip(ids, M):
            f.write(iid + "," + ",".join(f"{v:.6f}" for v in row) + "\n")


def write_rows(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_items(name):
    return list(csv.DictReader(open(ITEMS / f"{name}_items.csv")))


def build(name, X, n_missing, variants, counts, items, K, labels, group_col, outdir, extra=None):
    outdir.mkdir(parents=True, exist_ok=True)
    ids = [r["item_id"] for r in items]
    keys = np.array([int(r["key"]) for r in items])
    groups = [r[group_col] for r in items]
    write_rows([{"item_id": i, "n_missing_all_rows": int(m)} for i, m in zip(ids, n_missing)],
               outdir / f"{name}_missingness.csv")
    man = {"dataset": name, "counts": counts, "variants": {}, **(extra or {})}
    for v, mask in variants.items():
        Xv = X[mask]
        write_rows(item_stats(Xv, K, ids, labels), outdir / f"{name}_{v}_item_stats.csv")
        T = pair_tables(Xv, K)
        npair = T.sum((2, 3))[np.triu_indices(len(ids), 1)]
        mats = correlations(T)
        vm = {"n": int(mask.sum()), "pair_n_min": int(npair.min()), "pair_n_max": int(npair.max())}
        for method, M in mats.items():
            p = outdir / f"{name}_{v}_{method}.csv"
            write_matrix(M, ids, p)
            own, fails, neg, mown = mapping_check(M, groups, keys)
            vm[method] = {"min_eigenvalue": round(float(np.linalg.eigvalsh(M)[0]), 6),
                          "mapping_check": {"group": group_col, "items_most_correlated_with_own_group": own,
                                            "items_failing": [ids[j] for j in fails],
                                            "negative_within_group_keyed_pairs": neg,
                                            "mean_within_group_keyed_r": round(mown, 4),
                                            "offset_falsification": offset_test(M, groups, keys)}}
        man["variants"][v] = vm
    files = sorted(p.name for p in outdir.glob(f"{name}_*.csv"))
    man["sha256"] = {f: sha(outdir / f) for f in files}
    (outdir / f"{name}_MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({k: man[k] for k in ("counts", "variants")}, indent=1))


def make_fixtures():
    """Synthetic ordinal data (NOT real responses) from a known 2-factor model, same schema as the
    real artifacts, so downstream code can test loaders without the cache."""
    rng = np.random.default_rng(20261015)
    J, K, N = 12, 4, 2000
    lam = np.zeros((J, 2))
    lam[:6, 0] = np.linspace(0.5, 0.8, 6)
    lam[6:, 1] = np.linspace(0.5, 0.8, 6)
    phi = np.array([[1, 0.4], [0.4, 1]])
    S = lam @ phi @ lam.T
    np.fill_diagonal(S, 1)
    Z = rng.multivariate_normal(np.zeros(J), S, size=N)
    cuts = np.array([-0.3, 0.6, 1.4])  # skewed like symptom items
    X = (Z[:, :, None] > cuts).sum(2)
    items = [{"item_id": f"SYN_Q{j + 1:02d}", "position": j + 1, "data_column": f"S{j + 1}",
              "scale": "F1" if j < 6 else "F2", "key": 1, "text": f"Synthetic item {j + 1} (no text)",
              "text_source": "synthetic", "secondary_match": True} for j in range(J)]
    FIX.mkdir(parents=True, exist_ok=True)
    write_rows(items, FIX / "synthetic_items.csv")
    write_matrix(S, [r["item_id"] for r in items], FIX / "synthetic_population_corr.csv")
    write_rows([{"row": i, **{r["item_id"]: int(x) for r, x in zip(items, xs)}} for i, xs in enumerate(X)],
               FIX / "synthetic_responses.csv")
    build("synthetic", X, np.zeros(J, int), {"primary": np.ones(N, bool)}, {"n_rows": N}, items, K,
          [0, 1, 2, 3], "scale", FIX,
          extra={"primary_variant": "primary", "generator": {"seed": 20261015, "loadings": lam.tolist(), "factor_corr": phi.tolist(),
                               "cuts": cuts.tolist()}})


def main(which):
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "python": sys.version.split()[0]}
    src = {f["id"]: {k: f[k] for k in ("sha256", "license_status", "public_output")}
           for f in json.load(open(REPO / "data" / "sources.json"))["files"]}
    if "dass42" in which:
        items = read_items("dass42")
        X, nm, var, cnt = load_dass()
        build("dass42", X, nm, var, cnt, items, 4, [0, 1, 2, 3], "scale", AGG,
              extra={"primary_variant": "primary",
                     "source_records": {k: src[k] for k in ("dass_raw_zip", "dass42_pdf", "dass_scoring_template_pdf")},
                     "software": versions})
    if "ipipneo300" in which:
        items = read_items("ipipneo300")
        X, nm, var, cnt = load_ipip(np.array([int(r["key"]) for r in items]))
        build("ipipneo300", X, nm, var, cnt, items, 5, [1, 2, 3, 4, 5], "facet_code", AGG,
              extra={"primary_variant": "complete",
                     "source_records": {k: src[k] for k in ("ipip300_dat", "ipipneo_itemkey_xls", "ipipneo300_questionnaire_docx")},
                     "software": versions})
    if "fixtures" in which:
        make_fixtures()


if __name__ == "__main__":
    main(sys.argv[1:] or ["dass42", "ipipneo300", "fixtures"])
