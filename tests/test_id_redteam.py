"""Static identification tests: pure algebra on constructed population points:
no optimizer, no sampler, no random draw. Each test pins one identification fact the canonical text should carry."""
import jax
import numpy as np
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from model.m1 import loading_mask, pi_shape  # noqa: E402

GROUPS12 = ["D"] * 4 + ["A"] * 4 + ["S"] * 4
BLK = np.repeat(np.arange(3), 4)


def _parts(theta, K0, L, b0=0.0):
    J = len(K0)
    S = [theta * K0, np.zeros((J, J)), L @ L.T + b0 ** 2]
    d = sum(np.diag(M) for M in S) + 1.0
    s = 1 / np.sqrt(d)
    return [M * np.outer(s, s) for M in S] + [1 / d]


def _sig(P):
    return P[0] + P[1] + P[2] + np.diag(P[3])


def _sem(P, groups=None):
    return float(pi_shape(*map(jnp.asarray, P), groups=groups)["sem"])


def test_two_type_exchange_outside_block_span():
    """a two-type kernel whose types cross D/A/S, type-constant general loading, item-specific group
    loadings and b0 != 0: theta 0.64 and 0 give the same Sigma; D14 and D14b both move to 0."""
    J, G = 12, GROUPS12
    typ = (np.arange(J) % 3 == 0).astype(int)
    k = np.array([[0.5, 0.1], [0.1, 0.3]])
    K0 = np.diag(1 - k[typ, typ]) + k[np.ix_(typ, typ)]
    mask = np.asarray(loading_mask(J, "bifactor", G))
    v = 0.5 + 0.15 * np.sin(1.7 * np.arange(J))
    L = lambda uu, vv: mask * np.column_stack([uu[typ]] + [vv] * 3)
    u, b0, th = np.array([1.1, 0.8]), 0.4, 0.64
    P1 = _parts(th, K0, L(u, v), b0)
    e = np.sqrt(1 / (1 + th * (1 - np.diag(k))))  # theta' = 0
    m = np.outer(e, e) * (th * k + np.outer(u, u) + b0 ** 2)
    beta = (m[0, 0] * m[1, 1] - m[0, 1] ** 2) / (m[0, 0] + m[1, 1] - 2 * m[0, 1])
    u2 = np.sqrt(np.diag(m) - beta) * np.sign([1, m[0, 1] - beta])
    P2 = _parts(0.0, K0, L(u2, e[typ] * v), np.sqrt(beta))
    assert np.abs(_sig(P2) - _sig(P1)).max() < 1e-14
    assert _sem(P1) > 0.2 and _sem(P1, G) > 0.2
    assert _sem(P2) == 0.0 and _sem(P2, G) == 0.0


def test_d14b_has_no_limit_at_h5_class():
    """D14b is NaN at a nuisance-free scale-membership-class point, and its value near the class depends
    on the direction: along the "k" and "l" paths it is exactly constant (1 and 0), the joint "kl" path is a genuine limit."""
    from model.kernels import shape_kernel
    B = np.equal.outer(BLK, BLK).astype(float)
    K5 = shape_kernel(0.2 + 0.6 * B + 0.2 * np.eye(12))
    mask = np.asarray(loading_mask(12, "bifactor", GROUPS12))
    L5 = mask * np.column_stack([np.full(12, 1.1)] + [np.full(12, 0.5)] * 3)
    assert np.isnan(_sem(_parts(0.64, K5, L5), GROUPS12))
    i = np.arange(12)
    K1 = np.cos(np.add.outer(i, i) * 0.7) * 0.1
    K1 -= np.diag(np.diag(K1))
    eta = mask * np.column_stack([0.1 * np.sin(i + 1.0)] + [0.1 * np.cos(2.0 * i)] * 3)
    eps = 1e-3
    lim_k = _sem(_parts(0.64, K5 + eps * K1, L5), GROUPS12)
    lim_l = _sem(_parts(0.64, K5, L5 + eps * eta), GROUPS12)
    lim_kl = _sem(_parts(0.64, K5 + eps * K1, L5 + eps * eta), GROUPS12)
    assert abs(lim_k - 1) < 1e-6 and abs(lim_l) < 1e-6 and 0.1 < lim_kl < 0.9


def test_d14b_truth_moves_with_loading_heterogeneity_at_fixed_theta():
    """with exactly equal within-block loadings the bifactor+b0 part lies in D14b's removed span, so
    D14b = <q_s, q_s+q_n>/||q_s+q_n||^2 regardless of theta's share of the total pattern (here K0 non-block)."""
    i = np.arange(12)
    K0 = 0.5 * np.eye(12) + 0.5 * np.outer(np.cos(i), np.cos(i)) + 0.5 * np.outer(np.sin(i), np.sin(i))
    mask = np.asarray(loading_mask(12, "bifactor", GROUPS12))
    eq = mask * np.column_stack([np.full(12, 1.1)] + [np.repeat([0.5, 0.45, 0.55], 4)] * 3)
    het = mask * np.column_stack([1.1 + 0.2 * np.sin(3 * i)] + [np.repeat([0.5, 0.45, 0.55], 4) + 0.15 * np.cos(5 * i)] * 3)
    th = 0.05
    assert abs(_sem(_parts(th, K0, eq, 0.4), GROUPS12) - 1.0) < 1e-10  # nuisance-free: all non-block pattern is semantic
    assert _sem(_parts(th, K0, het, 0.4), GROUPS12) < 0.05  # 0.008 here, against 1 at equal loadings
