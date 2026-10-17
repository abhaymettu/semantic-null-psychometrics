"""Invariants of the item-correspondence null (embed.geometry.null_gram).

What must hold:
- a global orthogonal rotation leaves the cosine Gram unchanged -> rejected as a null
- identity permutation reproduces the observed S
- the Gram-only formula equals the explicit embedding construction
- orthogonalize=True: X' E* = X' E exactly (nuisance relation preserved)
- orthogonalize=False: nuisance part H E kept row-for-row; item i receives exactly item perm[i]'s residual
- applied to a corrected representation (nuis), the identity reproduces that S (pairing rule)
- a non-trivial permutation changes S and weakens its alignment with the observed S
- block-restricted permutations stay in their blocks; seeds reproduce exactly
"""
import numpy as np
import pytest

from embed import geometry as g


@pytest.fixture
def data():
    rng = np.random.Generator(np.random.PCG64(1))
    n, d, k = 30, 64, 3
    Z = rng.normal(size=(n, k))
    E = Z @ rng.normal(size=(k, d)) + rng.normal(size=(n, d)) + 2.0  # shared offset = anisotropy
    return g.unit_rows(E), Z


def test_global_rotation_is_not_a_null(data):
    E, _ = data
    Q, _ = np.linalg.qr(np.random.Generator(np.random.PCG64(2)).normal(size=(E.shape[1],) * 2))
    S = g.cosine_from_gram(g.gram(E))
    S_rot = g.cosine_from_gram(g.gram(E @ Q))
    np.testing.assert_allclose(S_rot, S, atol=1e-12)


def test_identity_permutation_reproduces_observed(data):
    E, Z = data
    X, _ = g.design(Z)
    for orth in (True, False):
        S0 = g.null_gram(g.gram(E), X, np.arange(len(E)), orthogonalize=orth)
        np.testing.assert_allclose(S0, g.cosine_from_gram(g.gram(E)), atol=1e-12)


def explicit(E, X, perm, orth):
    H = g.hat(X)
    R = E - H @ E
    PR = R[perm]
    return H @ E + (PR - H @ PR if orth else PR)


@pytest.mark.parametrize("orth", [True, False])
def test_gram_formula_matches_embedding_construction(data, orth):
    E, Z = data
    X, _ = g.design(Z)
    perm = np.random.Generator(np.random.PCG64(3)).permutation(len(E))
    S_gram = g.null_gram(g.gram(E), X, perm, orthogonalize=orth)
    S_emb = g.cosine_from_gram(g.gram(explicit(E, X, perm, orth)))
    np.testing.assert_allclose(S_gram, S_emb, atol=1e-12)


def test_orthogonalized_null_preserves_nuisance_relation_exactly(data):
    E, Z = data
    X, _ = g.design(Z)
    perm = np.random.Generator(np.random.PCG64(4)).permutation(len(E))
    Es = explicit(E, X, perm, True)
    np.testing.assert_allclose(X.T @ Es, X.T @ E, atol=1e-10)
    B = np.linalg.lstsq(X, E, rcond=None)[0]
    Bs = np.linalg.lstsq(X, Es, rcond=None)[0]
    np.testing.assert_allclose(Bs, B, atol=1e-10)


def test_freedman_lane_index_convention_pinned_by_loop(data):
    """Independent of the matrix code: row i of E* is H E[i] + R[perm[i]], built with a loop."""
    E, Z = data
    X, _ = g.design(Z)
    H = g.hat(X)
    HE, R = H @ E, E - H @ E
    perm = np.array([3, 0, 1, 2] + list(range(4, len(E))))  # non-symmetric cycle: P != P'
    Es = np.empty_like(E)
    for i in range(len(E)):
        Es[i] = HE[i] + R[perm[i]]
    S_loop = g.cosine_from_gram(g.gram(Es))
    np.testing.assert_allclose(g.null_gram(g.gram(E), X, perm, orthogonalize=False), S_loop, atol=1e-12)
    inv = np.argsort(perm)
    assert np.max(np.abs(g.null_gram(g.gram(E), X, inv, orthogonalize=False) - S_loop)) > 1e-3


def test_null_on_corrected_representation_pairs_with_its_S(data):
    E, Z = data
    reps = g.representations(E, Z, ["a", "b", "c"])
    X, _ = g.design(Z)
    S_nuis = g.cosine_from_gram(g.gram(reps["nuis"]))
    np.testing.assert_allclose(g.null_gram(g.gram(reps["nuis"]), X, np.arange(len(E))), S_nuis, atol=1e-12)
    perm = np.random.Generator(np.random.PCG64(12)).permutation(len(E))
    assert np.max(np.abs(g.null_gram(g.gram(reps["nuis"]), X, perm) - S_nuis)) > 1e-3


def test_nontrivial_permutation_changes_geometry(data):
    E, Z = data
    X, _ = g.design(Z)
    S = g.cosine_from_gram(g.gram(E))
    rng = np.random.Generator(np.random.PCG64(6))
    rs = []
    for i in range(50):
        Sn = g.null_gram(g.gram(E), X, rng.permutation(len(E)), orthogonalize=bool(i % 2))
        assert np.max(np.abs(Sn - S)) > 1e-3
        assert g.psd_check(Sn)["psd"]
        rs.append(np.corrcoef(g.offdiag(Sn), g.offdiag(S))[0, 1])
    # nuisance-predicted structure survives, item-specific structure does not
    assert np.mean(rs) < 0.9


def test_null_keeps_more_structure_with_nuisance_than_without(data):
    """The nuisance-aware null retains the nuisance-explained similarity, so null
    draws stay closer to the observed S than under plain label shuffling."""
    E, Z = data
    S = g.cosine_from_gram(g.gram(E))
    X_full, _ = g.design(Z)
    X_none, _ = g.design(np.zeros((len(E), 0)))
    rng = np.random.Generator(np.random.PCG64(7))
    perms = [rng.permutation(len(E)) for _ in range(50)]
    r = lambda X: np.mean([np.corrcoef(g.offdiag(g.null_gram(g.gram(E), X, p)), g.offdiag(S))[0, 1] for p in perms])
    assert r(X_full) > r(X_none) + 0.1


def test_block_permutation_and_seed_reproducibility():
    blocks = np.array([0] * 5 + [1] * 7 + [2] * 4)
    p = g.permutation(len(blocks), np.random.Generator(np.random.PCG64(8)), blocks)
    assert sorted(p) == list(range(len(blocks)))
    np.testing.assert_array_equal(blocks[p], blocks)
    E = g.unit_rows(np.random.Generator(np.random.PCG64(9)).normal(size=(16, 8)))
    Z = np.random.Generator(np.random.PCG64(10)).normal(size=(16, 2))
    stat = lambda S: g.offdiag(S).mean()
    a = g.null_distribution(E, Z, 20, seed=11, statistic=stat, blocks=blocks)
    b = g.null_distribution(E, Z, 20, seed=11, statistic=stat, blocks=blocks)
    np.testing.assert_array_equal(a, b)
