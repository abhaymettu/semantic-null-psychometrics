import numpy as np
from scipy.stats import multivariate_normal

from fit.polychoric import bvn_cdf, nearest_correlation, polychoric


def test_bvn_cdf_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(40):
        h, k = rng.uniform(-3, 3, 2)
        rho = rng.uniform(-0.95, 0.95)
        ref = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([h, k])
        assert abs(bvn_cdf(h, k, rho) - ref) < 1e-6
    assert abs(bvn_cdf(0.0, 0.0, 0.5) - (0.25 + np.arcsin(0.5) / (2 * np.pi))) < 1e-9
    assert abs(bvn_cdf(-0.0, 1.0, 0.3) - multivariate_normal(cov=[[1, .3], [.3, 1]]).cdf([0, 1])) < 1e-6


def _ordinal(Sigma, n, cuts, seed):
    z = np.random.default_rng(seed).multivariate_normal(np.zeros(len(Sigma)), Sigma, size=n)
    return np.digitize(z, cuts).astype(np.int8)


def test_polychoric_recovers_latent_correlation():
    J = 5
    rng = np.random.default_rng(0)
    L = rng.uniform(0.3, 0.8, (J, 1))
    Sigma = L @ L.T
    np.fill_diagonal(Sigma, 1.0)
    Y = _ordinal(Sigma, 40000, [-0.5, 0.4, 1.3], seed=2)  # 4 categories, skewed like DASS
    out = polychoric(Y)
    assert np.abs(out["R"] - Sigma).max() < 0.03
    assert np.allclose(out["thresholds"], [-0.5, 0.4, 1.3], atol=0.03)


def test_polychoric_handles_missing():
    Sigma = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
    Y = _ordinal(Sigma, 20000, [-0.3, 0.6], seed=3)
    Y[np.random.default_rng(4).random(Y.shape) < 0.1] = -1
    out = polychoric(Y, C=3)
    assert np.abs(out["R"] - Sigma).max() < 0.04
    assert out["n_pairs"][0, 1] < 20000


def test_nearest_correlation_makes_valid_matrix():
    R = np.array([[1, .9, -.9], [.9, 1, .9], [-.9, .9, 1]])
    Rc, changed = nearest_correlation(R)
    assert changed and np.linalg.eigvalsh(Rc).min() > 0 and np.allclose(np.diag(Rc), 1)


def test_degenerate_item_is_nan_not_spurious():
    Sigma = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
    Y = _ordinal(Sigma, 5000, [-0.3, 0.6], seed=5)
    Y[:, 1] = 2
    out = polychoric(Y, C=4)
    assert np.isnan(out["R"][0, 1]) and np.isnan(out["R"][1, 2]) and out["flags"]["degenerate_items"] == [1]
    assert abs(out["R"][0, 2] - 0.3) < 0.06
