"""Exact covariance-exchange tests (closed form) for the level ridge and the identification counterexamples.

Port of earlier tests:
same source literals and the same assertions, with the runtime `least_squares` search from 20 random starts replaced by
the closed-form block-constant exchange (d'_g = (1 + theta'(1 - w_g)) / (1 - r_g), rank-1 completion of the between-block
products, b0'^2 = beta free). Plus the scale-membership-class check: NaN without residue, zero on both sides with non-block nuisance.
Pure algebra at constructed points: no optimizer, no sampler, no random draw."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from model.m1 import loading_mask, pi_shape  # noqa: E402

GROUPS = ["D"] * 4 + ["A"] * 4 + ["S"] * 4
BLK = np.repeat(np.arange(3), 4)
B = np.equal.outer(BLK, BLK).astype(float)
KW, KB = np.array([0.45, 0.30, 0.55]), {(0, 1): 0.20, (0, 2): -0.05, (1, 2): 0.10}


def _parts(th, K0, lam, b0, g=None, Kn=None):
    J = len(K0)
    S = [th * K0, np.zeros((J, J)) if g is None else np.outer(g, g) * Kn, lam @ lam.T + b0 ** 2]
    d = sum(np.diag(M) for M in S) + 1.0
    s = 1 / np.sqrt(d)
    return [M * np.outer(s, s) for M in S] + [1 / d]


def _sig(P):
    return P[0] + P[1] + P[2] + np.diag(P[3])


def _pi(P, groups=GROUPS, key="sem"):
    return float(pi_shape(*map(jnp.asarray, P), groups=groups)[key])


def _h6_exchange(th2=0.0, beta=0.0):
    K0 = np.array([[1.0 if i == j else (KW[BLK[i]] if BLK[i] == BLK[j] else KB[tuple(sorted((BLK[i], BLK[j])))])
                    for j in range(12)] for i in range(12)])
    mask = np.asarray(loading_mask(12, "bifactor", GROUPS))
    lam = lambda u, v: mask * np.column_stack([u[BLK]] + [v[BLK]] * 3)  # noqa: E731
    P1 = _parts(0.64, K0, lam(np.array([1.0, 0.7, 1.2]), np.array([0.5, 0.6, 0.4])), 0.3)
    S1 = _sig(P1)
    r = np.array([S1[4 * k, 4 * k + 1] for k in range(3)])
    d2 = (1 + th2 * (1 - KW)) / (1 - r)
    t = {p: S1[4 * p[0], 4 * p[1]] * np.sqrt(d2[p[0]] * d2[p[1]]) - th2 * KB[p] - beta for p in KB}
    u0 = np.sqrt(t[(0, 1)] * t[(0, 2)] / t[(1, 2)])
    u = np.array([u0, t[(0, 1)] / u0, t[(0, 2)] / u0])
    v2 = d2 - 1 - th2 - u ** 2 - beta
    assert (v2 > 0).all(), "outside the feasible family"
    return P1, _parts(th2, K0, lam(u, np.sqrt(v2)), np.sqrt(beta))


def test_blockconst_D14b_not_identified_beyond_equal_within_block_loadings():
    """Same Sigma, D14b -0.448 vs 0: a claim of D14b general identification is wrong."""
    P1, P2 = _h6_exchange()
    assert np.abs(_sig(P1) - _sig(P2)).max() < 1e-12
    a, b = _pi(P1), _pi(P2)
    assert abs(b) < 1e-9 and a < -0.4
    assert abs(a + 0.447884) < 1e-6  # closed form at the source literals


def test_blockconst_exchange_is_a_family():
    """The theta' = 0 twin is one member of a family: any b0'^2 = beta, and theta' between 0 and 0.64."""
    for beta in (0.0, 0.05, 0.3):
        P1, P2 = _h6_exchange(0.0, beta)
        assert np.abs(_sig(P1) - _sig(P2)).max() < 1e-12 and abs(_pi(P2)) < 1e-9
    P1, P2 = _h6_exchange(0.32, 0.05)
    assert np.abs(_sig(P1) - _sig(P2)).max() < 1e-12 and abs(_pi(P2) + 0.240594) < 1e-6


def test_blockconst_pi_shape_groups_length_validated():
    """m1.pi_shape asserts len(groups) == J."""
    P1, _ = _h6_exchange()
    with pytest.raises(AssertionError):
        pi_shape(*map(jnp.asarray, P1), groups=["D"])


def test_scalemember_class_D14b_nan_without_residue_zero_with_nonblock_nuisance():
    """Scale-membership class (K0 in span{I, B, 11'}, equal loadings), exact twin at theta' = 0 by a scalar E. D14 moves to 0 in both
    cases. D14b is NaN (0/0) on both sides when nothing lies outside the removed span, and is defined and exactly 0 on
    both sides when an item-specific nuisance on a non-block Kn is present. Neither case identifies theta."""
    x, y, z, th = 0.40, 0.45, 0.15, 0.64
    K5 = x * np.eye(12) + y * B + z
    mask = np.asarray(loading_mask(12, "bifactor", GROUPS))
    lam = lambda gen, grp: mask * np.column_stack([np.full(12, gen)] + [np.full(12, grp)] * 3)  # noqa: E731
    i = np.arange(12)
    Kn = np.cos(0.7 * np.subtract.outer(i, i))  # rank 2, PSD, unit diagonal, not block-constant
    kap = 1 / (1 + th * x)  # E = sqrt(kap) I
    for g in (None, 0.3 + 0.1 * np.sin(i + 0.5)):
        Pa = _parts(th, K5, lam(1.1, 0.5), 0.3, g, Kn)
        b02 = 0.3 * np.sqrt(kap)
        gen2 = np.sqrt(kap * (1.1 ** 2 + 0.3 ** 2 + th * z) - b02 ** 2)
        Pb = _parts(0.0, K5, lam(gen2, np.sqrt(kap * (0.5 ** 2 + th * y))), b02,
                    None if g is None else np.sqrt(kap) * g, Kn)
        assert np.abs(_sig(Pa) - _sig(Pb)).max() < 1e-14
        assert _pi(Pa, None) > 0.4 and _pi(Pb, None) == 0.0
        if g is None:
            assert np.isnan(_pi(Pa)) and np.isnan(_pi(Pb))
        else:
            assert abs(_pi(Pa)) < 1e-12 and _pi(Pb) == 0.0 and _pi(Pa, key="den_ratio") > 1e-3


def test_blockconst_family_range_endpoints_and_sign():
    """Range claim: over the block-constant family theta' lies in (-1/lambda_max(K0),
    0.940397] (signed support; the upper end is v'_S = 0 at b0'^2 = 0) and D14b runs from -0.6176 (upper end) to +0.2958
    (approached, not attained, at the open signed bound). D14b's sign is opposite to theta' (monotonicity, checked by interval
    arithmetic); the original 0.64 point is interior."""
    K0 = np.array([[1.0 if i == j else (KW[BLK[i]] if BLK[i] == BLK[j] else KB[tuple(sorted((BLK[i], BLK[j])))])
                    for j in range(12)] for i in range(12)])
    assert abs(-1 / np.linalg.eigvalsh(K0).max() + 0.334853) < 1e-6
    P1, top = _h6_exchange(0.9403970, 0.0)
    assert np.abs(_sig(P1) - _sig(top)).max() < 1e-12 and abs(_pi(top) + 0.617607) < 1e-4
    with pytest.raises(AssertionError):
        _h6_exchange(0.9404, 0.0)
    _, low = _h6_exchange(-0.3348, 0.0)
    assert 0.29 < _pi(low) < 0.29583
    for th2 in (-0.2, 0.3, 0.9):
        assert np.sign(_pi(_h6_exchange(th2, 0.0)[1])) == -np.sign(th2)
