"""D14: pi_shape is constant along the D13 ridge and blind to equal-loading components."""
import jax.numpy as jnp
import numpy as np
import pytest

from model.kernels import level_kernel, shape_kernel
from model.m1 import loading_mask, m1_readout, pi_shape

J = 12
GROUPS = ["D"] * 4 + ["A"] * 4 + ["S"] * 4


def _readout(a, K, L, g=None, Kn=None, groups=None):
    ro = m1_readout(jnp.full(J, a), jnp.asarray(L), jnp.asarray(K), jnp.ones(L.shape),
                    None if g is None else jnp.asarray(g), None if Kn is None else jnp.asarray(Kn))
    return pi_shape(ro["Sigma_sem"], ro["Sigma_nuis"], ro["Sigma_res"], ro["uniq"], groups)


@pytest.mark.parametrize("groups", [None, GROUPS])
@pytest.mark.parametrize("alpha,c,m", [(0.8, 0.3, 0.0), (0.8, 0.0, 0.6), (0.0, 0.0, 0.6)])
def test_pi_shape_ridge_invariant_and_blind_to_uniform(alpha, c, m, groups):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(J, 5))
    K0 = shape_kernel(X @ X.T / 5 + 0.5)
    F = rng.normal(size=(J, 3))
    Kn = F @ F.T / 3
    g = np.full(J, 0.15)
    lam = np.asarray(loading_mask(J, "bifactor", GROUPS)) * rng.uniform(0.3, 1.0, (J, 4))
    # truth (level c, method factor m), and both ends of its ridge: uniform part all residual / all semantic
    t = _readout(alpha, level_kernel(K0, c), np.hstack([lam, np.full((J, 1), m)]), g, Kn, groups)
    u = alpha**2 * c + m**2
    lo = _readout(np.sqrt(alpha**2 * (1 - c)), K0, np.hstack([lam, np.full((J, 1), np.sqrt(u))]), g, Kn, groups)
    s2 = alpha**2 * (1 - c) + u
    hi = _readout(np.sqrt(s2), level_kernel(K0, u / s2), lam, g, Kn, groups)
    for k in ("sem", "nuis", "res"):
        assert abs(float(t[k]) - float(lo[k])) < 1e-6 and abs(float(t[k]) - float(hi[k])) < 1e-6
    assert abs(float(t["sem"] + t["nuis"] + t["res"]) - 1) < 1e-5  # float32
    if alpha == 0:
        assert abs(float(t["sem"])) < 1e-12  # no semantics: zero, whatever the equal-loading method factor
    else:
        assert float(t["sem"]) > 0.01


def _exchange_pair():
    """The scale-membership exact exchange: K0 in span{I, B_g, 11'} with equal loadings; alpha 0.8 and alpha 0 give
    the same Sigma."""
    B = np.array([[a == b for b in GROUPS] for a in GROUPS], float)
    mask = np.asarray(loading_mask(J, "bifactor", GROUPS))
    K0 = shape_kernel(0.2 + 0.6 * B + 0.2 * np.eye(J))
    y, z = K0[0, 1] - K0[0, 5], K0[0, 5]
    x = 1 - y - z
    alpha, gl, ll, k = 0.8, 1.1, 0.5, 1 + 0.64 * x
    L1 = mask * np.column_stack([np.full(J, gl)] + [np.full(J, ll)] * 3)
    L2 = mask * np.column_stack([np.full(J, np.sqrt((gl**2 + alpha**2 * z) / k))] + [np.full(J, np.sqrt((ll**2 + alpha**2 * y) / k))] * 3)
    return K0, (alpha, L1), (0.0, L2)


def test_pi_shape_groups_invariant_on_exact_exchange():
    K0, (a1, L1), (a2, L2) = _exchange_pair()
    sig = []
    for a, L in ((a1, L1), (a2, L2)):
        ro = m1_readout(jnp.full(J, a), jnp.asarray(L), jnp.asarray(K0), jnp.ones(L.shape))
        sig.append(np.asarray(ro["Sigma_sem"] + ro["Sigma_res"]) + np.diag(np.asarray(ro["uniq"])))
    assert np.abs(sig[0] - sig[1]).max() < 1e-6
    assert float(_readout(a1, K0, L1)["sem"]) - float(_readout(a2, K0, L2)["sem"]) > 0.5   # D14 without groups: not identified
    r1, r2 = _readout(a1, K0, L1, groups=GROUPS), _readout(a2, K0, L2, groups=GROUPS)
    assert np.isnan(float(r1["sem"])) and np.isnan(float(r2["sem"]))  # nothing outside the removed patterns


def test_pi_shape_nan_when_everything_is_equal_loading():
    K0 = level_kernel(shape_kernel(np.eye(J) + 0.3), 1.0)
    assert np.isnan(float(_readout(0.8, K0, np.full((J, 1), 0.5))["sem"]))


def test_pi_shape_groups_guard_float32_and_degenerate_groups():
    """float32 roundoff after projecting with groups must still give NaN on a removed
    pattern; a singleton / single group must not make the projection singular."""
    Jb = 42
    rng = np.random.default_rng(0)
    G = ["D"] * 14 + ["A"] * 14 + ["S"] * 14
    uniq = jnp.asarray(rng.uniform(0.3, 0.7, Jb), jnp.float32)
    s = jnp.sqrt(uniq)
    off = 1 - jnp.eye(Jb, dtype=jnp.float32)
    Z = jnp.zeros((Jb, Jb), jnp.float32)
    assert np.isnan(float(pi_shape(0.3 * jnp.outer(s, s) * off, Z, Z, uniq, G)["sem"]))
    X = rng.normal(size=(Jb, 5))
    K = jnp.asarray(np.corrcoef(X), jnp.float32)
    for groups in (["a"] * 41 + ["b"], ["a"] * Jb):
        r = pi_shape(0.5 * K, Z, 0.2 * jnp.outer(s, s) * off, uniq, groups)
        assert np.isfinite(float(r["sem"]))


def test_pi_shape_groups_length_validated():
    """A groups vector that is not one label per item is refused, not silently read as D14."""
    J = 6
    S = jnp.eye(J) * 0.0
    with pytest.raises(AssertionError):
        pi_shape(S, S, S, jnp.ones(J), groups=["D"])
