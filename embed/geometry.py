"""Representation corrections, Gram checks and the item-correspondence null.

Conventions (see embed/CONTRACT.md):
- All geometry runs in float64 on row-per-item matrices E (n_items x dim).
- "Mean" is the mean over the analysed item set (no external corpus). It is
  therefore dataset-dependent: adding or removing items changes every
  corrected representation.
- "Scale": every representation ends with rows re-normalised to unit L2 norm,
  so each S is a cosine Gram with unit diagonal.
"""
import numpy as np


def unit_rows(E):
    E = np.asarray(E, dtype=np.float64)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm row: representation collapsed an item to the origin")
    return E / n


def gram(E):
    G = E @ E.T
    return (G + G.T) / 2


def cosine_from_gram(G):
    d = np.sqrt(np.diag(G))
    if np.any(d == 0):
        raise ValueError("zero diagonal in Gram")
    S = G / np.outer(d, d)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    return S


def center(E):
    return E - E.mean(axis=0, keepdims=True)


def abtt(E, D, ref=None):
    """All-but-the-top (Mu & Viswanath 2018, arXiv:1702.01417): subtract the mean, then
    remove the projection on the top-D principal directions. Mean and directions come
    from `ref` (a reference sentence set, design decision D4) or, if ref is None, from
    the target items themselves (which also strips content the items share)."""
    B = E if ref is None else np.asarray(ref, dtype=np.float64)
    mu = B.mean(axis=0, keepdims=True)
    C = E - mu
    if D == 0:
        return C
    _, _, Vt = np.linalg.svd(B - mu, full_matrices=False)
    U = Vt[:D].T
    return C - C @ U @ U.T


def abtt_default_D(dim):
    """Mu & Viswanath's rule of thumb, D ~ d/100."""
    return max(1, round(dim / 100))


def design(Z):
    """Intercept + column-standardised nuisance covariates. Constant columns are
    dropped (returned in `dropped`) because they carry no information here."""
    Z = np.asarray(Z, dtype=np.float64)
    keep = [j for j in range(Z.shape[1]) if Z[:, j].std() > 1e-9 * max(1.0, np.abs(Z[:, j]).max())]
    Zk = Z[:, keep]
    Zs = (Zk - Zk.mean(axis=0)) / Zk.std(axis=0)
    X = np.column_stack([np.ones(len(Z)), Zs])
    return X, [j for j in range(Z.shape[1]) if j not in keep]


def hat(X):
    """Orthogonal projector on col(X) (pinv handles collinear nuisance columns)."""
    H = X @ np.linalg.pinv(X)
    return (H + H.T) / 2


def residualize(E, X):
    """Remove the part of every embedding dimension that is linear in X (incl. intercept)."""
    return E - hat(X) @ E


def representations(E_unit, Z, Z_names, ref=None):
    """All named representations of one model x text-variant, each unit-row normalised.
    Names follow interface C2 (embed/CONTRACT.md) (raw, abtt-item, abtt-ref); the rest are extra sensitivity.

    raw              pooled model output, L2-normalised (what the model card prescribes)
    abtt-ref         reference-set mean and top-D PCs removed, D = d/100 (C2/D4 primary)
    abtt-ref-d1/-d3  same with D = 1, 3
    abtt-item        target-item mean and top-D PCs removed, D = d/100 (only if n > D + 2)
    abtt-item-d1/-d3 same with D = 1, 3
    center           target-item mean removed
    nuis             centred and residualised on all nuisance columns
    nuis-nodesir     as nuis without the desirability proxy (proxy-sensitivity check)
    """
    E = np.asarray(E_unit, dtype=np.float64)
    n, dim = E.shape
    D0 = abtt_default_D(dim)
    reps = {"raw": unit_rows(E)}
    if ref is not None:
        reps["abtt-ref"] = unit_rows(abtt(E, D0, ref))
        for D in (1, 3):
            reps[f"abtt-ref-d{D}"] = unit_rows(abtt(E, D, ref))
    if n > D0 + 2:
        reps["abtt-item"] = unit_rows(abtt(E, D0))
    for D in (1, 3):
        if n > D + 2:
            reps[f"abtt-item-d{D}"] = unit_rows(abtt(E, D))
    reps["center"] = unit_rows(center(E))
    X, _ = design(Z)
    reps["nuis"] = unit_rows(residualize(E, X))
    keep = [j for j, nm in enumerate(Z_names) if not nm.startswith("desirability")]
    Xn, _ = design(np.asarray(Z)[:, keep])
    reps["nuis-nodesir"] = unit_rows(residualize(E, Xn))
    return reps


def projector_baselines(Z, Z_names):
    """Cosine Gram of the projector each linear correction applies, I - H.
    For near-isotropic embeddings (E E' ~ c I) the corrected Gram is ~ I - H whatever
    the model, so agreement with this baseline is mechanical, not semantic. Read
    cross-model agreement of center/nuis/nuis-nodesir against it."""
    n = len(Z)
    keep = [j for j, nm in enumerate(Z_names) if not nm.startswith("desirability")]
    Xs = {"center": np.ones((n, 1)), "nuis": design(Z)[0], "nuis-nodesir": design(np.asarray(Z)[:, keep])[0]}
    return {rep: cosine_from_gram(np.eye(n) - hat(X)) for rep, X in Xs.items()}


def psd_check(S, tol=1e-8):
    """Dense checks on a cosine Gram. tol is relative to n (eigvalsh error ~ n*eps*||S||)."""
    S = np.asarray(S, dtype=np.float64)
    n = len(S)
    w = np.linalg.eigvalsh(S)
    return {
        "n": n,
        "symmetric_max_abs": float(np.max(np.abs(S - S.T))),
        "diag_max_abs_dev": float(np.max(np.abs(np.diag(S) - 1))),
        "min_eig": float(w[0]),
        "max_eig": float(w[-1]),
        "numerical_rank": int(np.sum(w > 1e-10 * w[-1])),
        "psd": bool(w[0] >= -tol * n),
        "offdiag_mean": float((S.sum() - n) / (n * (n - 1))),
    }


def offdiag(S):
    i, j = np.triu_indices(len(S), 1)
    return np.asarray(S)[i, j]


def permutation(n, rng, blocks=None):
    """Uniform permutation, optionally restricted to permute only within blocks."""
    if blocks is None:
        return rng.permutation(n)
    blocks = np.asarray(blocks)
    p = np.arange(n)
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        p[idx] = idx[rng.permutation(len(idx))]
    return p


def null_gram(G, X, perm, orthogonalize=True):
    """Nuisance-aware item-correspondence null, computed from the Gram alone.

    Decompose E = H E + R with H the projector on [1, Z] (Freedman & Lane 1983).
    Pass as E (and so G) the SAME representation the observed statistic uses
    (e.g. the S_nuis or S_abtt-ref embeddings): the identity permutation then
    reproduces that observed S, and the null differs only in item correspondence.

    The null keeps every item's nuisance-predicted part H E and hands it the
    residual of another item: E* = H E + P R. With orthogonalize=True the
    permuted residual is re-projected, E* = H E + (I - H) P R, so that X' E* = X' E
    exactly: the null geometry has the same linear relation to every nuisance
    covariate as the observed one. Only item-to-residual correspondence changes.

    A global orthogonal rotation E -> E Q is NOT used: it leaves G = E E' unchanged
    (tests/embed/test_null.py demonstrates this), so it is no null at all.

    Works on G = E E' so it is dimension-free:
      E* = A E with A = H + M P (I - H),  M = I - H or I,   G* = A G A'.
    Returns the cosine Gram of E*.
    """
    n = len(G)
    H = hat(X)
    I = np.eye(n)
    P = I[perm]
    M = (I - H) if orthogonalize else I
    A = H + M @ P @ (I - H)
    return cosine_from_gram(gram_from(A, G))


def gram_from(A, G):
    Gs = A @ G @ A.T
    return (Gs + Gs.T) / 2


def null_distribution(E_rep, Z, n_perm, seed, statistic, blocks=None, orthogonalize=True):
    """statistic(S) -> float over n_perm null cosine Grams. Seeded with numpy PCG64.
    E_rep: rows of the representation the observed statistic was computed on."""
    E = np.asarray(E_rep, dtype=np.float64)
    G = gram(E)
    X, _ = design(Z)
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.array([
        statistic(null_gram(G, X, permutation(len(E), rng, blocks), orthogonalize))
        for _ in range(n_perm)
    ])
