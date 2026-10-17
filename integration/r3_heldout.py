"""R3 inductive item-heldout runner (inputs below include protocol files of the original run, not in this repository): each fold leaves 6 items (2 per scale) out of fitting and scores their conditional log-likelihood.

One exhaustive stratified 7-fold item partition of DASS-42. Per fold the models B0 (bifactor + b0 + nuisance kernel)
and B2 (B0 + global scale x inductive semantic shape kernel, c = 0) are fitted on the 36 training items only; the
6 held-out items get their general loading, specific loading and nuisance scale g from the training-fit mean over the
training items of the same scale. Score: conditional Gaussian log-likelihood of the held-out items given the training
items on the frozen R at the frozen n_eff. Statistic G = sum over folds of [ll_cond(B2) - ll_cond(B0)] at the best
training objective; G_min uses the worst B2 start within NEAR log-units of the best. Null: 100 within-scale
permutations of the rows/columns of S before the per-fold centring.

This is the runner of the paper's held-out check, kept as it was frozen for that run.
Run (via integration/run_r3_heldout.sh):
  python -m integration.r3_heldout check-inputs --embed-dir E
  python -m integration.r3_heldout run --embed-dir E --model M [--deadline EPOCH]
  python -m integration.r3_heldout verdict OUT1 OUT2 OUT3
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fit.api import fit_decomposition
from model.kernels import null_permutation
from model.m1 import components, loading_mask, with_const, wishart_loglik

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "integration" / "out"
ENCODERS = ("bge-large-en-v1.5", "all-mpnet-base-v2", "qwen3-embedding-0.6b")
REP = "abtt-ref"
FROZEN = dict(n_folds=7, partition_seed=31015, n_null=100, null_rng0=3000, starts=16, near=1.0, variant="primary")
KERNELS = tuple(f"embed/out/dass42__{m}__S_{REP}.npy" for m in ENCODERS)
R_PATH = "data/aggregates/dass42_primary_polychoric.csv"
R_BLOB = "a0099be6fbddc32c3ff3f33e4b68ec44952b0664"  # git blob of the 42x42 matrix in the original repository (a fingerprint; it discloses no value)
INPUTS = ("fit", "model", "integration/__init__.py", "integration/dass42.py", "integration/neff.py", "integration/r3_heldout.py",
          "tests/test_r3_heldout.py", "integration/R3-HELDOUT-PROTOCOL.md", "integration/run_r3_heldout.sh",
          "scripts/load_data.py", "scripts/build_aggregates.py", "data/items/dass42_items.csv",
          "data/items/ipipneo300_items.csv", "data/README.md", R_PATH, "data/aggregates/dass42_MANIFEST.json", "integration/out/dass42_neff.json",
          "embed/MANIFEST.json", "embed/geometry.py", "embed/out/dass42_nuisance.csv", "requirements-model.txt") + KERNELS


# ---------------------------------------------------------------- partition and inductive kernels

def partition(scales, n_folds=FROZEN["n_folds"], seed=FROZEN["partition_seed"], order="DAS"):
    """Exhaustive stratified partition. One generator, scales in `order`, items in the given (data) order; each
    scale's permutation is cut into consecutive blocks of len/n_folds items, block f goes to fold f."""
    scales = np.asarray(scales)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for s in order:
        idx = np.flatnonzero(scales == s)
        per, rem = divmod(len(idx), n_folds)
        if rem or not per:
            raise ValueError(f"scale {s}: {len(idx)} items do not split into {n_folds} equal blocks")
        perm = rng.permutation(idx)
        for f in range(n_folds):
            folds[f] += perm[f * per:(f + 1) * per].tolist()
    return [np.array(sorted(f)) for f in folds]


def inductive_shape(S, T):
    """All rows of the shape kernel with the item-centring done over the TRAINING items only:
    G = (I - 1w')S(I - w1'), w = 1_T/|T|, renormalized to unit diagonal. Rows/cols T equal shape_kernel(S[T,T])."""
    J = len(S)
    w = np.zeros(J)
    w[T] = 1.0 / len(T)
    P = np.eye(J) - np.outer(np.ones(J), w)
    G = P @ S @ P.T
    d = np.sqrt(np.clip(np.diag(G), 1e-12, None))
    return G / np.outer(d, d)


def inductive_nuisance(F, T):
    """nuisance_kernel with feature mean/sd taken from the training items only; same row normalization."""
    F = np.asarray(F, float)
    mu, sd = F[T].mean(0), F[T].std(0)
    Z = (F - mu)[:, sd > 0] / sd[sd > 0]
    norms = np.linalg.norm(Z, axis=1)
    Z = np.where(norms[:, None] > 0, Z / np.where(norms > 0, norms, 1)[:, None], 0)
    K = Z @ Z.T
    np.fill_diagonal(K, 1.0)
    return K


# ---------------------------------------------------------------- fitting, held-out prediction, score

def fit_starts(R, n_eff, K_sem, K_nuis, groups, seed, n_starts):
    """n_starts single-start MAP fits. Start i is PRNGKey(seed + i), exactly what fit_decomposition(n_starts=n)
    uses for its i-th start, so the best of these equals the library's best-of-n; the per-start objectives and
    parameters are kept (the library returns only the best)."""
    out = []
    for i in range(n_starts):
        f = fit_decomposition(R, n_eff, K_sem, K_nuis=K_nuis, structure="bifactor", groups=groups, sem_scale="global",
                              const_factor=True, method="map", seed=seed + i, n_starts=1)
        d, dg = f["draws"], f["diagnostics"]
        nlp = float(dg["neg_logpost"])
        Sig = f["Sigma_sem"] + f["Sigma_nuis"] + f["Sigma_res"] + np.diag(np.asarray(f["uniq"]).reshape(-1))
        out.append(dict(start=i, nlp=nlp, converged=bool(dg["converged"]) and bool(np.isfinite(nlp)),
                        p=dict(a=float(d["a"][0][0]), lam=np.asarray(d["lam"][0]), g=np.asarray(d["g"][0]), b0=float(d["b0"][0])),
                        Sigma_train=Sig,
                        # P3 ruling 1: near-optimality is judged on the training log-likelihood (3.7: same Sigma_TT = same
                        # likelihood; K2's 1.0 is in loglik_gap units), not on neg_logpost (prior/Jacobian differ)
                        ll_train=float(wishart_loglik(jnp.asarray(Sig), jnp.asarray(R), n_eff)) if np.isfinite(nlp) else float("nan")))
    return out


def predict_sigma(p, scales, T, K_sem, K_nuis):
    """Predicted J x J standardized Sigma. Training rows use the fitted parameters; held-out rows use the mean of
    the general loading, the scale's specific loading and g over the training items of the same scale (raw scale,
    same column). Uniqueness follows from M1's standardization, so the result is PD with unit diagonal."""
    scales = list(scales)
    J = len(scales)
    labels = sorted(set(scales))
    mask = np.asarray(loading_mask(J, "bifactor", scales))
    T = np.asarray(T)
    H = np.setdiff1d(np.arange(J), T)
    lam, g = np.zeros((J, mask.shape[1])), np.zeros(J)
    lam[T], g[T] = p["lam"], p["g"]
    for h in H:
        same = [i for i, t in enumerate(T) if scales[t] == scales[h]]
        col = 1 + labels.index(scales[h])
        lam[h, 0], lam[h, col] = p["lam"][same, 0].mean(), p["lam"][same, col].mean()
        g[h] = p["g"][same].mean()
    Lambda = with_const(jnp.asarray(lam * mask), p["b0"])
    S_sem, S_nuis, S_res, d = components(p["a"] * jnp.ones(J), Lambda, jnp.asarray(K_sem), jnp.asarray(g), jnp.asarray(K_nuis))
    s = 1.0 / jnp.sqrt(d)
    return np.asarray((S_sem + S_nuis + S_res + jnp.eye(J)) * s[:, None] * s[None, :])


def ll_cond(Sigma, R, T, H, n):
    """Conditional log-likelihood of H given T: wishart_loglik on T+H minus on T (omitted constants cancel)."""
    T, TH = np.asarray(T), np.concatenate([T, H])
    ll = lambda idx: wishart_loglik(jnp.asarray(Sigma[np.ix_(idx, idx)]), jnp.asarray(R[np.ix_(idx, idx)]), n)
    return float(ll(TH) - ll(T))


def fold_record(starts, R, n_eff, scales, T, H, K_sem42, K_nuis42, near=FROZEN["near"], items=False):
    """Score every start; best = lowest finite objective and the fold is ok iff that start converged and its score is
    finite; near = converged finite-score starts whose training log-likelihood is at least the best start's minus `near`."""
    for s in starts:
        s["Sigma"] = predict_sigma(s["p"], scales, T, K_sem42, K_nuis42)
        s["ll"] = ll_cond(s["Sigma"], R, T, H, n_eff) if np.isfinite(s["nlp"]) else float("nan")
    fin = [s for s in starts if np.isfinite(s["nlp"])]
    rec = dict(starts=[[s["start"], s["nlp"], s["converged"], s["ll"], s.get("ll_train")] for s in starts], ok=False)
    if not fin:
        return rec
    best = min(fin, key=lambda s: s["nlp"])
    rec.update(best_start=best["start"], best_nlp=best["nlp"], ll_best=best["ll"],
               ok=bool(best["converged"] and np.isfinite(best["ll"])),
               near=[s["ll"] for s in fin if s["converged"] and np.isfinite(s["ll"]) and s["ll_train"] >= best["ll_train"] - near])
    # descriptive: how many starts are near-optimal and how far their held-out scores spread
    rec.update(n_converged=sum(s["converged"] for s in starts), n_near=len(rec["near"]),
               near_spread=(max(rec["near"]) - min(rec["near"])) if rec["near"] else None)
    if items:
        rec["ll_items_best"] = [ll_cond(best["Sigma"], R, T, [h], n_eff) for h in H]
        # P3 rulings 9-10 (descriptives, no verdict role): imputed scale means at the best start
        # [general, specific, g, n specific loadings < 0, n training items], and the score with the held-out items'
        # semantic off-diagonals set to 0 (ll_best minus this = the part of the score carried by held-out wording rows)
        lab, p = sorted(set(scales)), best["p"]
        rec["imputed_best"] = {s: [float(p["lam"][m, 0].mean()), float(p["lam"][m, 1 + lab.index(s)].mean()), float(p["g"][m].mean()),
                                   int((p["lam"][m, 1 + lab.index(s)] < 0).sum()), int(m.sum())]
                               for s in lab for m in [np.array([scales[t] == s for t in T])]}
        K0 = np.array(K_sem42, float)
        K0[H, :], K0[:, H] = 0.0, 0.0
        K0[H, H] = 1.0
        rec["ll_best_heldout_sem_off"] = ll_cond(predict_sigma(p, scales, T, K0, K_nuis42), R, T, H, n_eff)
    return rec


def statistic(b0, b2):
    """G, G_min (frozen rule: worst B2 near-start minus best-training B0) and the descriptive G_min_sym (worst B2
    near-start minus best-heldout B0 near-start). complete=False if any fold of either model is not ok."""
    if not all(r["ok"] for r in b0 + b2):
        return dict(complete=False)
    return dict(complete=True,
                G=sum(y["ll_best"] - x["ll_best"] for x, y in zip(b0, b2)),
                G_min=sum(min(y["near"]) - x["ll_best"] for x, y in zip(b0, b2)),
                G_min_sym=sum(min(y["near"]) - max(x["near"]) for x, y in zip(b0, b2)),
                per_fold=[y["ll_best"] - x["ll_best"] for x, y in zip(b0, b2)])


# ---------------------------------------------------------------- run, verdict

def _b2_folds(S, F, R, n_eff, scales, folds, seed, n_starts, near, items=False):
    recs = []
    for T in folds_T(folds, len(S)):
        H = np.setdiff1d(np.arange(len(S)), T)
        Kn = inductive_nuisance(F, T)
        Ks = inductive_shape(S, T)
        starts = fit_starts(R[np.ix_(T, T)], n_eff, Ks[np.ix_(T, T)], Kn[np.ix_(T, T)], [scales[t] for t in T], seed, n_starts)
        recs.append(fold_record(starts, R, n_eff, scales, T, H, Ks, Kn, near, items))
    return recs


def folds_T(folds, J):
    return [np.setdiff1d(np.arange(J), H) for H in folds]


def run(S, F, R, n_eff, scales, n_null=FROZEN["n_null"], n_starts=FROZEN["starts"], deadline=None, near=FROZEN["near"], log=print,
        n_folds=FROZEN["n_folds"], checkpoint=lambda res: None):
    """-> result dict. `folds` below are the HELD-OUT sets. B0 does not depend on S (shared by encoders).
    checkpoint(partial result, status RUNNING) is called after the observed fits and after every null draw."""
    S, R = np.asarray(S, float), np.asarray(R, float)
    folds = partition(scales, n_folds)
    return _run(S, F, R, n_eff, scales, folds, n_null, n_starts, deadline, near, log, checkpoint)


def _run(S, F, R, n_eff, scales, folds, n_null, n_starts, deadline, near, log, checkpoint):
    J = len(scales)
    zeros = np.zeros((J, J))
    b0 = []
    for H in folds:
        T = np.setdiff1d(np.arange(J), H)
        Kn = inductive_nuisance(F, T)
        starts = fit_starts(R[np.ix_(T, T)], n_eff, zeros[np.ix_(T, T)], Kn[np.ix_(T, T)], [scales[t] for t in T], 0, n_starts)
        b0.append(fold_record(starts, R, n_eff, scales, T, H, zeros, Kn, near, items=True))
    log(f"B0 done ok={[r['ok'] for r in b0]}")
    b2 = _b2_folds(S, F, R, n_eff, scales, folds, 0, n_starts, near, items=True)
    obs = statistic(b0, b2)
    log(f"observed {obs}")
    strata = ["DAS".index(s) for s in scales]
    nulls, status = [], "complete"
    fp = hashlib.sha256(json.dumps([[r["starts"], r["ok"]] for r in b0]).encode()).hexdigest()
    result = lambda st: dict(status=st, n_null_planned=n_null, n_starts=n_starts, near=near, n_eff=n_eff,
                             folds_heldout=[f.tolist() for f in folds], b0=b0, b2=b2, observed=obs, null=list(nulls),
                             b0_fingerprint=fp)
    checkpoint(result("RUNNING"))
    for b in range(n_null):
        if deadline is not None and time.time() > deadline:
            status = "INCOMPLETE_TIMEOUT"
            break
        p = null_permutation(J, strata, rng=FROZEN["null_rng0"] + b)
        st = statistic(b0, _b2_folds(S[np.ix_(p, p)], F, R, n_eff, scales, folds, b, n_starts, near))
        nulls.append(dict(b=b, ok=st["complete"], G=st.get("G")))
        log(f"null {b} {nulls[-1]}")
        checkpoint(result("RUNNING"))
    return result(status)


def encoder_verdict(res):
    """PASS iff complete, G_min_sym > 0 (P3 ruling 3: the one criterion B0's near-optimal spread can move; it implies G > 0),
    G > every null G (a failed null draw counts as an exceedance), and G_min > null max (B0 is shared by all draws and cancels)."""
    if res["status"] != "complete" or len(res["null"]) != res["n_null_planned"]:
        return dict(verdict="INCOMPLETE", reason=res["status"], n_null=len(res.get("null", [])))
    obs = res["observed"]
    if not obs["complete"]:
        return dict(verdict="NOT_PASS", reason="observed fit has a fold without a converged best start")
    Gs = [n["G"] for n in res["null"] if n["ok"]]
    failed = len(res["null"]) - len(Gs)
    exceed = failed + sum(g >= obs["G"] for g in Gs)
    nmax = max(Gs) if Gs else float("nan")
    ok = obs["G_min_sym"] > 0 and exceed == 0 and obs["G_min"] > nmax
    return dict(verdict="PASS" if ok else "NOT_PASS", G=obs["G"], G_min=obs["G_min"], G_min_sym=obs["G_min_sym"],
                null_max=nmax, exceedances=int(exceed), null_failed=failed, n_null=len(res["null"]),
                p=(1 + exceed) / (1 + len(res["null"])))


def overall_verdict(results):
    """results: {encoder: run result}. POSITIVE = 3/3, MIXED = 1-2, NEGATIVE = 0; any INCOMPLETE run = no verdict."""
    per = {e: encoder_verdict(r) for e, r in results.items()}
    problems = []
    if set(results) != set(ENCODERS):
        problems.append(f"encoders {sorted(results)} != frozen {sorted(ENCODERS)}")
    if len({r["b0_fingerprint"] for r in results.values() if "b0_fingerprint" in r}) > 1:
        problems.append("B0 fingerprints differ between encoders")
    for e, r in results.items():
        if (r.get("n_null_planned"), r.get("n_starts"), r.get("near")) != (FROZEN["n_null"], FROZEN["starts"], FROZEN["near"]):
            problems.append(f"{e}: not the frozen grid")
    n = sum(v["verdict"] == "PASS" for v in per.values())
    if any(v["verdict"] == "INCOMPLETE" for v in per.values()) or problems:
        word = "INCOMPLETE (no verdict)" if not problems else "INVALID (" + "; ".join(problems) + ")"
    else:
        word = "POSITIVE" if n == len(ENCODERS) else "NEGATIVE" if n == 0 else "MIXED"
    return dict(verdict=word, n_pass=n, per_encoder=per, problems=problems)


# ---------------------------------------------------------------- inputs and CLI

def _git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)


def smoothing_history():
    """No-smoothing from the matrix's history, not from today's builder (which proves nothing about a committed file):
    the working file is blob R_BLOB, every commit that touched R_PATH holds that same blob, and the builder at each of
    those commits has no nearest-correlation / smoothing call (the original builder writes the elementwise pairwise
    polychorics straight to the CSV). Missing history is a problem, not a pass."""
    log = _git("log", "--format=%H", "--full-history", "--", R_PATH)
    commits = log.stdout.split()
    if log.returncode or not commits:
        return ["matrix history unverifiable (git log empty or failed)"]
    probs = []
    if _git("hash-object", str(ROOT / R_PATH)).stdout.strip() != R_BLOB:
        probs.append(f"{R_PATH} is not blob {R_BLOB}")
    for c in commits:
        if _git("rev-parse", f"{c}:{R_PATH}").stdout.strip() != R_BLOB:
            probs.append(f"{c[:7]} holds a different {R_PATH}")
        src = _git("show", f"{c}:scripts/build_aggregates.py")
        if src.returncode:
            probs.append(f"{c[:7]}: builder unverifiable")
        elif re.search(r"nearest_corr|smooth|higham", src.stdout, re.I):
            probs.append(f"{c[:7]}: builder references smoothing")
    return probs


def check_inputs(embed_dir, data_dir=ROOT / "data"):
    """Freeze-time assertions; a list of problems (empty = ok), each a hard refusal. The data MANIFEST has no smoothing
    field: no-smoothing is asserted from history (smoothing_history) plus the MANIFEST hash and min eigenvalue. The abtt
    reference must be the file the kernels were built from (embed MANIFEST name and sha256) and share no DASS text."""
    from scripts.load_data import matrix  # noqa: F401  (import check: same loader as level_sweep)
    problems = []
    man = json.loads((data_dir / "aggregates" / "dass42_MANIFEST.json").read_text())
    f = data_dir / "aggregates" / "dass42_primary_polychoric.csv"
    if hashlib.sha256(f.read_bytes()).hexdigest() != man["sha256"][f.name]:
        problems.append("primary polychoric sha256 differs from the data MANIFEST")
    ids, R = matrix("dass42", FROZEN["variant"], "polychoric")
    if not (np.allclose(R, R.T) and np.allclose(np.diag(R), 1)):
        problems.append("R is not symmetric with unit diagonal")
    if abs(np.linalg.eigvalsh(R).min() - man["variants"][FROZEN["variant"]]["polychoric"]["min_eigenvalue"]) > 1e-5:
        problems.append("R min eigenvalue differs from the MANIFEST (smoothing suspected)")
    problems += smoothing_history()
    emb = json.loads((Path(embed_dir) / "MANIFEST.json").read_text())
    rf = emb["itemsets"]["dass42"].get("ref_file") or emb.get("ref_file")
    ref = data_dir / "items" / "ipipneo300_items.csv"
    if not rf or rf.get("name") != ref.name:
        problems.append(f"abtt reference file is {rf and rf.get('name')}, expected {ref.name}")
    elif rf.get("sha256") != hashlib.sha256(ref.read_bytes()).hexdigest():
        problems.append("abtt reference sha256 in the embed MANIFEST differs from data/items/ipipneo300_items.csv (or is unrecorded)")
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    texts = lambda p: {norm(r["text"]) for r in csv.DictReader(open(p))}
    if texts(data_dir / "items" / "dass42_items.csv") & texts(ref):
        problems.append("a DASS item text appears in the abtt reference set")
    files = emb["itemsets"]["dass42"].get("files", {})
    for k in KERNELS:
        rec, kf = files.get(k.removeprefix("embed/")), Path(embed_dir) / k.removeprefix("embed/")
        if not rec or not kf.exists() or rec.get("sha256") != hashlib.sha256(kf.read_bytes()).hexdigest():
            problems.append(f"{k}: missing or sha256 differs from the embed MANIFEST")
    if not (OUT / "dass42_neff.json").exists():
        problems.append("dass42_neff.json missing")
    # P3 ruling 5, the no-smoothing SOURCE (hard): integration/neff.py re-estimated R from raw rows with the pairwise
    # fit.polychoric (no matrix-level step) and recorded the max |difference| to the committed matrix; plus the statement in data/README.md
    neff = json.loads((OUT / "dass42_neff.json").read_text()).get(FROZEN["variant"], {}) if (OUT / "dass42_neff.json").exists() else {}
    if not neff.get("max_abs_diff_vs_data_lane", float("inf")) <= 1e-5:
        problems.append("no-smoothing unverified: dass42_neff.json lacks a raw-row pairwise recompute within 1e-5 of the committed R")
    readme = data_dir / "README.md"
    if "No smoothing is applied." not in (readme.read_text() if readme.exists() else ""):
        problems.append("no-smoothing unverified: data/README.md lacks the 'No smoothing is applied.' statement")
    return problems


def _load(embed_dir, model):
    from integration.dass42 import load_items, load_kernel
    from scripts.load_data import matrix
    items = load_items()
    order = sorted(items, key=lambda r: ("DAS".index(r["scale"]), int(r["position"])))
    ids = [r["item_id"] for r in order]
    all_ids, R_all = matrix("dass42", FROZEN["variant"], "polychoric")
    pos = [all_ids.index(i) for i in ids]
    S, F, _ = load_kernel(Path(embed_dir), model, REP, ids)
    n_eff = json.loads((OUT / "dass42_neff.json").read_text())[FROZEN["variant"]]["n_eff"]
    return S, F, R_all[np.ix_(pos, pos)], n_eff, [r["scale"] for r in order]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-inputs")
    c.add_argument("--embed-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--embed-dir", type=Path, required=True)
    r.add_argument("--model", required=True, choices=ENCODERS)
    r.add_argument("--deadline", type=float, default=None, help="absolute epoch seconds; null draws stop after it (INCOMPLETE)")
    v = sub.add_parser("verdict")
    v.add_argument("outs", nargs=3, type=Path)
    a = ap.parse_args()
    if a.cmd != "verdict" and a.embed_dir.resolve() != (ROOT / "embed").resolve():
        raise SystemExit("refuse: --embed-dir must be the repo's embed/ (its kernels are frozen INPUTS)")
    if a.cmd == "check-inputs":
        probs = check_inputs(a.embed_dir)
        print(json.dumps(probs))
        raise SystemExit(1 if probs else 0)
    if a.cmd == "verdict":
        res = {json.loads(p.read_text())["model"]: json.loads(p.read_text())["result"] for p in a.outs}
        print(json.dumps(overall_verdict(res), indent=1))
        return
    path = OUT / f"r3_heldout__{a.model}__{REP}.json"
    if path.exists():
        raise SystemExit(f"refuse: {path} exists (one run, no rerun)")
    probs = check_inputs(a.embed_dir)
    if probs:
        raise SystemExit(f"refuse: {probs}")
    kfile = a.embed_dir / "out" / f"dass42__{a.model}__S_{REP}.npy"
    git = lambda *x: _git(*x).stdout.strip()
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    prov = dict(git_head=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                kernel_file=str(kfile), kernel_sha256=sha(kfile), R_sha256=sha(ROOT / R_PATH), R_blob=R_BLOB,
                neff_sha256=sha(OUT / "dass42_neff.json"), frozen=FROZEN,
                inputs_sha256={f: sha(ROOT / f) for f in git("ls-files", "--", *INPUTS).split()},
                embed_manifest_sha256=sha(a.embed_dir / "MANIFEST.json"), nuisance_sha256=sha(a.embed_dir / "out" / "dass42_nuisance.csv"))
    t0 = time.time()

    def save(res):  # atomic; every checkpoint overwrites the one output file, so a killed run leaves its last state
        out = dict(provenance=prov, model=a.model, rep=REP, result=res, verdict_encoder=encoder_verdict(res),
                   seconds=round(time.time() - t0, 1),
                   status_note="model-scale conditional log-likelihood units, MAP, NOT calibrated; pipeline inherited from full-data fits on these 42 items")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1))
        os.replace(tmp, path)
        return out

    stub = dict(n_null_planned=FROZEN["n_null"], n_starts=FROZEN["starts"], near=FROZEN["near"], null=[])
    if a.deadline is not None and time.time() > a.deadline:  # queued behind the other worker until the shared budget ran out
        print(json.dumps(save(dict(stub, status="INCOMPLETE_STARVED"))["verdict_encoder"]))
        return
    save(dict(stub, status="STARTED"))  # claims the output path: a kill before the first checkpoint still blocks a rerun
    S, F, R, n_eff, scales = _load(a.embed_dir, a.model)
    res = run(S, F, R, n_eff, scales, deadline=a.deadline, log=lambda m: print(m, flush=True), checkpoint=save)
    print(json.dumps(save(res)["verdict_encoder"]))


if __name__ == "__main__":
    main()
