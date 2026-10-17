"""Semantic and nuisance kernels, and the calibrated semantic nulls (design decision D4, D5, D8)."""
import numpy as np


def cosine(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T


def abtt(X, ref=None, D=1):
    """All-but-the-top: subtract mean and remove top-D principal directions, both estimated on ref
    (the target items themselves when ref is None)."""
    ref = X if ref is None else ref
    mu = ref.mean(0)
    _, _, Vt = np.linalg.svd(ref - mu, full_matrices=False)
    P = Vt[:D]
    Xc = X - mu
    return Xc - (Xc @ P.T) @ P


def nuisance_kernel(F):
    """Correlation kernel of standardized surface features: PSD, unit diagonal (rows with all-zero
    standardized features get unit diagonal and zero off-diagonal)."""
    F = np.asarray(F, float)
    sd = F.std(0)
    Z = (F - F.mean(0))[:, sd > 0] / sd[sd > 0]
    norms = np.linalg.norm(Z, axis=1)
    Z = np.where(norms[:, None] > 0, Z / np.where(norms > 0, norms, 1)[:, None], 0)
    K = Z @ Z.T
    np.fill_diagonal(K, 1.0)
    return K


def top_coords(K, m):
    """Top-m eigen-coordinates of a kernel, scaled by sqrt(eigenvalue): U U' approximates K."""
    w, V = np.linalg.eigh(K)
    idx = np.argsort(w)[::-1][:m]
    return V[:, idx] * np.sqrt(np.maximum(w[idx], 0))


def null_rotation(X, F=None, rng=None):
    """Null A draw: item-side Haar rotation of the content part of X (J x d).

    Keeps exactly: the part of X in span([1, F]) (centroid and nuisance-explained coordinates), the
    singular values of the content part, and the embedding-side Gram X'X. Changes which item has
    which content geometry. A global rotation X R (d-side) would leave every cosine unchanged and is
    therefore not a null.
    """
    rng = np.random.default_rng(rng)
    J = X.shape[0]
    B = np.ones((J, 1)) if F is None else np.column_stack([np.ones(J), np.asarray(F, float)])
    Ub, sb, _ = np.linalg.svd(B, full_matrices=False)
    Qb = Ub[:, sb > sb.max() * 1e-10]  # orthonormal basis of the true span, even if F is collinear
    P = np.eye(J) - Qb @ Qb.T
    V, s, Wt = np.linalg.svd(P @ X, full_matrices=False)
    r = int((s > s.max() * 1e-10).sum())
    V, s, Wt = V[:, :r], s[:r], Wt[:r]
    G = rng.standard_normal((r, r))
    Q, Rr = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(Rr))  # Haar
    return (X - P @ X) + (V @ Q) * s @ Wt


def null_permutation(J, strata=None, rng=None):
    """Discrete special case: permutation of item labels within nuisance strata."""
    rng = np.random.default_rng(rng)
    perm = np.arange(J)
    strata = np.zeros(J, int) if strata is None else np.asarray(strata)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        perm[idx] = rng.permutation(idx)
    return perm


def offdiag_mean(S):
    J = len(S)
    return float((S.sum() - np.trace(S)) / (J * (J - 1)))


def shape_kernel(S):
    """Level-free shape of a cosine kernel: cosine of the item-centred rows, computed from the Gram
    alone (P S P, P = I - 11'/J, renormalized to unit diagonal). PSD, unit diagonal, off-diagonal
    mean near -1/(J-1). design decision D11: the shape is what the data can inform; the level is not."""
    J = len(S)
    P = np.eye(J) - np.ones((J, J)) / J
    G = P @ S @ P
    d = np.sqrt(np.clip(np.diag(G), 1e-12, None))
    return G / np.outer(d, d)


def level_kernel(K0, c):
    """K_c = c 11' + (1 - c) K0: PSD and unit diagonal for c in [0, 1].
    Parametrization: c adds a uniform rank-1 (general) component to the semantic kernel."""
    K = c + (1.0 - c) * K0
    np.fill_diagonal(K, 1.0)
    return K
