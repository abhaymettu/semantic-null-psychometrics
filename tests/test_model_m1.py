import numpy as np
import pytest

from fit.api import fit_decomposition, fit_m2
from model.kernels import abtt, cosine, null_permutation, null_rotation, top_coords


def _toy(J=24, d=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, d))
    X = centers[np.arange(J) % 3] + 0.9 * rng.standard_normal((J, d)) + 3.0  # shared offset = anisotropy
    return X, np.arange(J) % 3


def _truth(K, a, lam):
    S0 = np.outer(a, a) * K + np.outer(lam, lam) + np.eye(len(a))
    s = 1 / np.sqrt(np.diag(S0))
    return S0 * np.outer(s, s), a**2 / np.diag(S0), lam**2 / np.diag(S0)


def test_components_psd_and_shares_sum_to_one():
    X, _ = _toy()
    K = cosine(abtt(X))
    J = len(K)
    rng = np.random.default_rng(1)
    a, lam = rng.uniform(0.2, 1.0, J), rng.uniform(0.3, 1.0, J)
    Sigma, sem, res = _truth(K, a, lam)
    assert np.linalg.eigvalsh(Sigma).min() > 0 and np.allclose(np.diag(Sigma), 1)
    out = fit_decomposition(Sigma, 1e5, K, method="map")
    tot = out["sem_share"] + out["nuis_share"] + out["res_share"] + out["uniq"]
    assert np.allclose(tot, 1)
    for M in (out["Sigma_sem"], out["Sigma_res"]):
        assert np.linalg.eigvalsh(M).min() > -1e-9


@pytest.mark.parametrize("scale_a", [0.0, 0.7])
def test_map_recovers_known_decomposition(scale_a):
    X, _ = _toy(seed=3)
    K = cosine(abtt(X))
    J = len(K)
    rng = np.random.default_rng(5)
    a = scale_a * rng.uniform(0.5, 1.5, J)
    lam = rng.uniform(0.4, 1.0, J)
    Sigma, sem, res = _truth(K, a, lam)
    # sampled latent correlation at n = 20000
    z = rng.multivariate_normal(np.zeros(J), Sigma, size=20000)
    R = np.corrcoef(z.T)
    out = fit_decomposition(R, 20000, K, method="map")
    pi_true = sem.sum() / (sem.sum() + res.sum())
    assert abs(out["pi_common"][0] - pi_true) < 0.05
    assert np.abs(out["sem_share"][0] - sem).max() < 0.1


def test_null_rotation_invariants():
    X, groups = _toy(seed=7)
    F = np.column_stack([groups == 1, np.linspace(0, 1, len(X))]).astype(float)
    Xs = null_rotation(X, F, rng=0)
    assert np.allclose(Xs.T @ Xs, X.T @ X)                       # embedding-side Gram kept
    B = np.column_stack([np.ones(len(X)), F])
    H = B @ np.linalg.pinv(B)
    assert np.allclose(H @ Xs, H @ X)                            # nuisance-explained part kept
    sv = lambda M: np.linalg.svd(M - H @ M, compute_uv=False)
    assert np.allclose(sv(Xs), sv(X))                            # content spectrum kept
    assert np.abs(cosine(Xs) - cosine(X)).max() > 0.1            # correspondence changed


def test_global_rotation_is_not_a_null():
    X, _ = _toy(seed=8)
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((X.shape[1],) * 2))
    assert np.allclose(cosine(X @ Q), cosine(X))


def test_null_permutation_stays_in_strata():
    strata = np.array([0, 0, 1, 1, 1, 2])
    p = null_permutation(6, strata, rng=1)
    assert sorted(p) == list(range(6)) and np.all(strata[p] == strata)


def test_m2_readout_has_cross_term():
    X, _ = _toy(seed=9)
    K = cosine(abtt(X))
    U = top_coords(K, 3)
    rng = np.random.default_rng(2)
    lam = 0.6 + U @ np.array([0.3, -0.2, 0.1]) + 0.1 * rng.standard_normal(len(K))
    S0 = np.outer(lam, lam) + np.eye(len(K))
    s = 1 / np.sqrt(np.diag(S0))
    out = fit_m2(S0 * np.outer(s, s), 1e5, U, method="map")
    assert np.abs(out["Sigma_cross"][0]).max() > 1e-3
    assert 0.5 < out["r2_sem"][0] <= 1.0


def test_null_rotation_with_collinear_features_rotates_full_complement():
    X, _ = _toy(J=10, seed=11)
    t = np.arange(10.0)
    F = np.column_stack([t, 2 * t, np.ones(10)])  # rank-deficient [1, F]
    Xs = null_rotation(X, F, rng=3)
    B = np.column_stack([np.ones(10), t])
    H = B @ np.linalg.pinv(B)
    assert np.allclose(H @ Xs, H @ X)
    # the content part must actually move in all directions orthogonal to span(1, t)
    assert np.linalg.matrix_rank((Xs - X) - H @ (Xs - X), tol=1e-8) >= 6


def test_shrinkage_floor_when_no_semantic_component():
    X, _ = _toy(seed=13)
    K = cosine(abtt(X))
    J = len(K)
    lam = np.random.default_rng(6).uniform(0.4, 1.0, J)
    Sigma, _, _ = _truth(K, np.zeros(J), lam)
    out = fit_decomposition(Sigma, 20000, K, method="map")
    assert out["pi_common"][0] < 0.02


def test_nuisance_and_bifactor_paths():
    X, groups = _toy(seed=14)
    K = cosine(abtt(X))
    J = len(K)
    rng = np.random.default_rng(7)
    Kn = np.corrcoef(rng.standard_normal((J, 3)))  # any PSD unit-diagonal kernel
    lam = rng.uniform(0.4, 1.0, J)
    Sigma, _, _ = _truth(K, 0.5 * np.ones(J), lam)
    out = fit_decomposition(Sigma, 1e4, K, K_nuis=Kn, structure="bifactor", groups=list(groups), method="map")
    assert out["Lambda"].shape == (1, J, 4)
    assert out["nuis_share"].sum() > 0
    with pytest.raises(ValueError):
        fit_decomposition(Sigma, 1e4, K, structure="cfa", method="map")


def test_global_sem_scale_is_uniform():
    X, _ = _toy(seed=15)
    K = cosine(abtt(X))
    J = len(K)
    lam = np.random.default_rng(8).uniform(0.4, 1.0, J)
    Sigma, sem, res = _truth(K, 0.6 * np.ones(J), lam)
    out = fit_decomposition(Sigma, 1e4, K, method="map", sem_scale="global")
    a = out["draws"]["a"][0]
    assert np.allclose(a, a[0])
    assert abs(out["pi_common"][0] - sem.sum() / (sem.sum() + res.sum())) < 0.03


def test_shape_and_level_kernels():
    from model.kernels import level_kernel, offdiag_mean, shape_kernel
    X, _ = _toy(seed=16)
    S = cosine(X)
    K0 = shape_kernel(S)
    assert np.allclose(np.diag(K0), 1) and np.linalg.eigvalsh(K0).min() > -1e-9
    assert abs(offdiag_mean(K0) - (-1 / (len(S) - 1))) < 0.02
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert np.allclose(K0, cosine(Xn - Xn.mean(0)))         # cosine of item-centred unit rows
    for c in (0.0, 0.3, 0.9):
        Kc = level_kernel(K0, c)
        assert np.linalg.eigvalsh(Kc).min() > -1e-9 and np.allclose(np.diag(Kc), 1)
    p = np.random.default_rng(0).permutation(len(S))
    assert np.isclose(offdiag_mean(level_kernel(K0[np.ix_(p, p)], 0.3)), offdiag_mean(level_kernel(K0, 0.3)))


def test_estimands_offdiag_share():
    from model.m1 import estimands
    J = 4
    A = np.full((J, J), 0.1); np.fill_diagonal(A, 0.3)
    B = np.full((J, J), 0.3); np.fill_diagonal(B, 0.3)
    e = estimands(A, np.zeros((J, J)), B, np.full(J, 0.4))
    assert np.isclose(float(e["pi_offdiag"]), 0.25) and np.isclose(float(e["pi_common"]), 0.5)


def test_const_factor_absorbs_kernel_level():
    # with a global scale and a free constant residual loading, alpha^2 c 11' is
    # absorbed by b0^2 11', so the c = 0 fit reaches the c > 0 truth exactly
    import jax.numpy as jnp
    from model.kernels import level_kernel, shape_kernel
    from model.m1 import wishart_loglik
    X, _ = _toy(seed=17)
    K0 = shape_kernel(cosine(X))
    J = len(K0)
    lam = np.random.default_rng(9).uniform(0.4, 1.0, J)
    Sigma, _, _ = _truth(level_kernel(K0, 0.4), 0.8 * np.ones(J), lam)
    ll = lambda f: float(wishart_loglik(jnp.asarray(f["Sigma_sem"] + f["Sigma_res"] + np.diag(f["uniq"][0])),
                                        jnp.asarray(Sigma), 1e4))
    best = ll(fit_decomposition(Sigma, 1e4, level_kernel(K0, 0.4), method="map", sem_scale="global"))
    no_const = ll(fit_decomposition(Sigma, 1e4, K0, method="map", sem_scale="global"))
    const = fit_decomposition(Sigma, 1e4, K0, method="map", sem_scale="global", const_factor=True)
    assert no_const < best - 1.0              # without b0, c = 0 cannot carry the uniform part
    assert ll(const) > best - 0.05            # with b0 it can: the level is not identified
    assert const["Lambda"].shape[-1] == 2


def test_null_permutation_within_scale_keeps_level():
    from model.kernels import level_kernel, offdiag_mean, shape_kernel
    X, g = _toy(seed=18)
    K0 = shape_kernel(cosine(X))
    p = null_permutation(len(K0), g, rng=3)
    assert (g[p] == g).all()
    assert np.isclose(offdiag_mean(level_kernel(K0[np.ix_(p, p)], 0.0)), offdiag_mean(level_kernel(K0, 0.0)))


def test_ridge_endpoint_is_the_same_matrix():
    from integration.level_sweep import ridge
    from model.kernels import level_kernel, shape_kernel
    X, g = _toy(seed=19)
    K0 = shape_kernel(cosine(X))
    J = len(K0)
    lam = np.random.default_rng(10).uniform(0.4, 1.0, J)
    Sigma, _, _ = _truth(level_kernel(K0, 0.3), 0.8 * np.ones(J), lam)
    kw = dict(K_nuis=None, structure="efa1", groups=None, sem_scale="global", n_factors=1, const_factor=True)
    f = fit_decomposition(Sigma, 1e4, K0, method="map", **kw)
    r = ridge(f, K0, Sigma, 1e4, kw)
    assert abs(r["loglik_gap_c_max_reconstructed"] - r["loglik_gap_c0"]) < 1e-6
    assert r["c_max"] >= 0.3 - 0.02                      # the true c lies on the ridge
    assert r["at_c_max"]["pi_offdiag"] > r["at_c0"]["pi_offdiag"]
