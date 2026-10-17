import numpy as np

from embed import geometry as g
from embed import nuisance as nz


def rand(n=20, d=32, seed=0):
    return np.random.Generator(np.random.PCG64(seed)).normal(size=(n, d)) + 3.0


def test_representations_are_unit_cosine_psd():
    E = g.unit_rows(rand())
    ref = g.unit_rows(rand(n=40, seed=5))
    Z = np.random.Generator(np.random.PCG64(1)).normal(size=(20, len(nz.COLUMNS)))
    reps = g.representations(E, Z, nz.COLUMNS, ref)
    assert set(reps) == {"raw", "abtt-ref", "abtt-ref-d1", "abtt-ref-d3", "abtt-item", "abtt-item-d1",
                         "abtt-item-d3", "center", "nuis", "nuis-nodesir"}
    assert "abtt-ref" not in g.representations(E, Z, nz.COLUMNS)
    for R in reps.values():
        c = g.psd_check(g.cosine_from_gram(g.gram(R)))
        assert c["psd"] and c["diag_max_abs_dev"] < 1e-12 and c["symmetric_max_abs"] == 0


def test_abtt_ref_uses_reference_mean_and_directions():
    E, ref = rand(), rand(n=40, seed=6) * 2
    A = g.abtt(E, 2, ref)
    mu = ref.mean(axis=0)
    _, _, Vt = np.linalg.svd(ref - mu, full_matrices=False)
    np.testing.assert_allclose(A @ Vt[:2].T, 0, atol=1e-9)
    np.testing.assert_allclose(A, (E - mu) - (E - mu) @ Vt[:2].T @ Vt[:2], atol=1e-10)
    assert g.abtt_default_D(1024) == 10 and g.abtt_default_D(768) == 8


def test_centering_removes_common_offset():
    E = g.unit_rows(rand())
    assert g.offdiag(g.cosine_from_gram(g.gram(E))).mean() > 0.8  # anisotropic
    C = g.center(E)
    np.testing.assert_allclose(C.mean(axis=0), 0, atol=1e-12)


def test_abtt_removes_top_directions():
    E = g.unit_rows(rand())
    A = g.abtt(E, 2)
    _, _, Vt = np.linalg.svd(g.center(E), full_matrices=False)
    np.testing.assert_allclose(A @ Vt[:2].T, 0, atol=1e-10)


def test_residualize_is_orthogonal_to_design_and_drops_constants():
    E = rand()
    Z = np.column_stack([np.random.Generator(np.random.PCG64(2)).normal(size=(20, 2)), np.ones(20)])
    X, dropped = g.design(Z)
    assert dropped == [2] and X.shape == (20, 3)
    np.testing.assert_allclose(X.T @ g.residualize(E, X), 0, atol=1e-10)


def test_psd_check_flags_indefinite():
    S = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]])
    assert not g.psd_check(S)["psd"]


def test_lexical_features():
    t = ["I don't feel sad.", "I never finish anything, not even small tasks (eg, chores).", "Qqqv xyzt."]
    L = nz.lexical(t)
    cols = nz.COLUMNS[:-1] + nz.EXTRA
    col = lambda name: L[:, cols.index(name)]
    np.testing.assert_array_equal(col("n_words"), [4, 10, 2])
    np.testing.assert_array_equal(col("n_chars"), [len(x) for x in t])
    np.testing.assert_array_equal(col("negation"), [1, 1, 0])
    np.testing.assert_array_equal(col("negation_count"), [1, 2, 0])
    np.testing.assert_array_equal(col("first_person"), [1, 1, 0])
    np.testing.assert_array_equal(col("has_example"), [0, 1, 0])
    assert col("valence")[0] < 0 and col("valence_hits")[0] >= 1  # "sad" is negative in VADER
    assert col("valence")[2] == 0 and col("valence_hits")[2] == 0  # no lexicon hits -> 0


def test_vader_pin():
    assert len(nz.load_vader()) > 7000


def test_desirability_is_zscored_average():
    rng = np.random.Generator(np.random.PCG64(3))
    items = {m: g.unit_rows(rng.normal(size=(10, 8))) for m in "ab"}
    anchors = {m: g.unit_rows(rng.normal(size=(2, 8))) for m in "ab"}
    d = nz.desirability(items, anchors)
    assert d.shape == (10,) and abs(d.mean()) < 1e-12


def test_linear_corrections_impose_projector_structure_on_isotropic_embeddings():
    """Why cross-model agreement after center/nuis is not semantic evidence: two independent
    high-dim random 'models' agree almost perfectly after the same nuisance projection."""
    rng = np.random.Generator(np.random.PCG64(4))
    n = 12
    Z = rng.normal(size=(n, len(nz.COLUMNS)))
    names = nz.COLUMNS
    base = g.projector_baselines(Z, names)["nuis"]
    S1, S2 = (g.cosine_from_gram(g.gram(g.representations(g.unit_rows(rng.normal(size=(n, 4096))), Z, names)["nuis"])) for _ in range(2))
    r = lambda a, b: np.corrcoef(g.offdiag(a), g.offdiag(b))[0, 1]
    assert r(S1, base) > 0.95 and r(S1, S2) > 0.9
