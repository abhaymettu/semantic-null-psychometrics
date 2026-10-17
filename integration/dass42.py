"""DASS-42 first artifact: M0 descriptive baseline, M1 decomposition, three heatmaps, per-item table.

Run: .venv/bin/python -m integration.dass42 --embed-dir <worktree>/embed [--model M] [--rep REP]
     [--method map|nuts] [--null N]
Reads only aggregates (the polychoric matrix from scripts/build_aggregates.py, the kernel S and nuisance.csv from embed/); writes aggregates.
Every output carries the identification label from design decision D1/D2 and a status label:
"descriptive" (M0), "model-based, NOT calibrated" until sim recovery (D9) passes for this regime.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import numpy as np

from fit.api import fit_decomposition, fit_m2
from model.kernels import level_kernel, nuisance_kernel, null_permutation, offdiag_mean, shape_kernel, top_coords
from scripts.load_data import manifest, matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "integration" / "out"
FIGS = ROOT / "figs"
NUIS_COLS = ["n_words", "n_chars", "negation", "valence", "first_person", "has_example", "desirability"]


def load_items():
    with open(ROOT / "data" / "items" / "dass42_items.csv") as f:
        return list(csv.DictReader(f))


def load_kernel(embed_dir, model, rep, ids, itemset="dass42"):
    """embed_dir = an `embed/` directory (interface C2 v3, flat layout)."""
    man = json.loads((embed_dir / "MANIFEST.json").read_text())
    order = man["itemsets"][itemset]["item_order"]
    idx = [order.index(i) for i in ids]
    S = np.load(embed_dir / "out" / f"{itemset}__{model}__S_{rep}.npy")[np.ix_(idx, idx)]
    with open(embed_dir / "out" / f"{itemset}_nuisance.csv") as f:
        rows = {r["item_id"]: r for r in csv.DictReader(f)}
    cols = [c for c in NUIS_COLS if c in next(iter(rows.values()))]
    F = np.array([[float(rows[i][c]) for c in cols] for i in ids])
    return S, F, cols


def descriptive(R, S, n_perm=2000, seed=0):
    """M0 (Kojima / Kambeitz shape): off-diagonal agreement, QAP permutation p, first-factor strength."""
    J = len(R)
    iu = np.triu_indices(J, 1)
    r_obs = np.corrcoef(R[iu], S[iu])[0, 1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        p = rng.permutation(J)
        null[b] = np.corrcoef(R[iu], S[np.ix_(p, p)][iu])[0, 1]
    ev = lambda M: np.linalg.eigh(M)
    wR, VR = ev(R)
    wS, VS = ev(S)
    v1R, v1S = VR[:, -1] * np.sign(VR[:, -1].sum()), VS[:, -1] * np.sign(VS[:, -1].sum())
    return dict(
        offdiag_r=float(r_obs), qap_p=float((1 + (null >= r_obs).sum()) / (1 + n_perm)),
        first_eig_share_R=float(wR[-1] / J), first_eig_share_S=float(wS[-1] / J),
        tucker_congruence_first_eigvec=float(v1R @ v1S / np.linalg.norm(v1R) / np.linalg.norm(v1S)),
        S_offdiag_mean=float(S[iu].mean()), R_offdiag_mean=float(R[iu].mean()),
    )


def heatmaps(R, sem, res, labels, scale_of, title_bits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    J = len(R)
    panels = [
        (R, "Empirical latent correlation", "observed (polychoric), " + title_bits["data"]),
        (sem, "S-aligned component", "model-based M1; alignment with semantic similarity, not caused-by-language"),
        (res, "Residual component", "model-based M1; unexplained structure, incl. any equal-loading term"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 9.0), constrained_layout=True)
    vmax = 0.8
    for ax, (M, title, sub) in zip(axes, panels):
        show = M.copy()
        np.fill_diagonal(show, np.nan)  # diagonals differ by construction (1 vs shares); off-diagonals compared
        im = ax.imshow(show, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        sub = textwrap.fill(sub, 70); sub += "\n" * (1 - sub.count("\n"))
        ax.set_title(f"{title}\n{sub}", fontsize=10.5, loc="left")
        ax.set_xticks(range(J))
        ax.set_yticks(range(J))
        ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
        ax.set_yticklabels(labels, fontsize=7.5)
        # subscale boundaries
        bounds = [i for i in range(1, J) if scale_of[i] != scale_of[i - 1]]
        for b in bounds:
            ax.axhline(b - 0.5, color="0.2", lw=0.8)
            ax.axvline(b - 0.5, color="0.2", lw=0.8)
    cb = fig.colorbar(im, ax=axes, shrink=0.8)
    cb.set_label("off-diagonal value (shared scale; diagonal blanked)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.suptitle("\n".join(textwrap.fill(p, 165) for p in title_bits["suptitle"].split("\n")), fontsize=12, x=0.01, ha="left")
    # supxlabel is laid out by constrained_layout, so a wrapped multi-line footer is never clipped
    footer = "\n".join(textwrap.fill(p, 215) for p in title_bits["footer"].split("\n"))
    fig.supxlabel(footer, fontsize=9.5, ha="left", x=0.01)
    fig.savefig(f"{path}.png", dpi=170)
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=Path, required=True)
    ap.add_argument("--model", default="bge-large-en-v1.5")
    ap.add_argument("--rep", default="abtt-ref")
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--method", default="map")
    ap.add_argument("--structure", default="bifactor", choices=["efa1", "cfa", "bifactor"],
                    help="residual factor structure; groups = official D/A/S scale")
    ap.add_argument("--null", type=int, default=0, help="permutation-null refits (MAP)")
    ap.add_argument("--null-kind", default="perm", choices=["perm", "fl"], help="perm = primary, fl = sensitivity")
    ap.add_argument("--sem-scale", default="item", choices=["item", "global"])
    ap.add_argument("--m2-dims", type=int, default=5, help="M2 route: top-m semantic coordinates (0 = skip)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", type=Path, default=None, help="override integration/out and figs (dry runs)")
    ap.add_argument("--const-factor", action="store_true",
                    help="headline model (D13): free equal-loading term b0, kernel = level_kernel(shape_kernel(S), c_obs), "
                         "the encoder's own level as one point on the ridge; use with --sem-scale global")
    a = ap.parse_args()
    global OUT, FIGS
    if a.outdir:
        OUT, FIGS = a.outdir / "out", a.outdir / "figs"

    items = load_items()
    order = sorted(items, key=lambda r: ("DAS".index(r["scale"]), int(r["position"])))
    ids = [r["item_id"] for r in order]
    all_ids, R_all = matrix("dass42", a.variant, "polychoric")
    pos = [all_ids.index(i) for i in ids]
    R = R_all[np.ix_(pos, pos)]
    n = manifest("dass42")["variants"][a.variant]["n"]
    neff_file = ROOT / "integration" / "out" / "dass42_neff.json"
    n_eff = json.loads(neff_file.read_text())[a.variant]["n_eff"] if neff_file.exists() else n
    S, F, ncols = load_kernel(a.embed_dir, a.model, a.rep, ids)
    Kn = nuisance_kernel(F)

    struct = a.structure + ("+const" if a.const_factor else "")  # same marker as integration.level_sweep
    tag = a.tag or f"{a.model}__{a.rep}__{struct}__{a.sem_scale}__{a.variant}__{a.method}"
    groups = [r["scale"] for r in order]
    fit_kw = dict(K_nuis=Kn, structure=a.structure, groups=groups, sem_scale=a.sem_scale)
    K, c_obs = S, None
    if a.const_factor:
        # same kernel as integration.level_sweep at c_obs, so the panels sit on the committed ridge
        K0 = shape_kernel(S)
        c_obs = (offdiag_mean(S) - offdiag_mean(K0)) / (1 - offdiag_mean(K0))
        assert 0.0 <= c_obs <= 1.0, f"c_obs {c_obs} outside [0, 1]: level_kernel is PSD only there"
        K = level_kernel(K0, c_obs)
        fit_kw["const_factor"] = True
    m0 = descriptive(R, S)
    fit = fit_decomposition(R, n_eff, K, **fit_kw, method=a.method, seed=0)
    q = lambda x, p: np.quantile(x, p, axis=0)
    summary = dict(
        tag=tag, status="model-based, NOT calibrated (sim recovery pending, design decision D9)",
        identification="S-aligned = alignment with fixed semantic kernel; residual = unexplained conditional structure; sources assumed independent (D3)",
        source_terms_status="no-explicit-license: aggregates only; response rows are never redistributed",
        variant=a.variant, n=n, n_eff=n_eff, model=a.model, rep=a.rep, nuisance_cols=ncols, method=a.method,
        residual_structure=a.structure, sem_scale=a.sem_scale, const_factor=a.const_factor, kernel_level_c=c_obs,
        diagnostics=fit["diagnostics"], M0_descriptive=m0,
        pi_offdiag=dict(mean=float(fit["pi_offdiag"].mean()), q05=float(q(fit["pi_offdiag"], .05)), q95=float(q(fit["pi_offdiag"], .95)),
                        definition="S-aligned share of model-implied off-diagonal (item-pair) covariance" + ("; at the encoder level c_obs, one point on the non-identified D13 ridge (conditional, not a headline)" if a.const_factor else "; conditional on the kernel level (D11), not identified")),
        nuis_offdiag=float(fit["nuis_offdiag"].mean()),
        pi_common=dict(mean=float(fit["pi_common"].mean()), q05=float(q(fit["pi_common"], .05)), q95=float(q(fit["pi_common"], .95)),
                       definition="S-aligned share of common VARIANCE (includes the kernel's unit diagonal); secondary"),
        nuis_common=float((fit["nuis_share"].sum(1) / (fit["sem_share"] + fit["nuis_share"] + fit["res_share"]).sum(1)).mean()),
    )
    OUT.mkdir(exist_ok=True, parents=True)
    if a.null:
        # PRIMARY (design decision D8 revision): item-label permutation of S, which keeps the
        # off-diagonal value multiset and therefore the level exactly. Freedman-Lane is
        # an opt-in sensitivity only (--null-kind fl); it can move the level, which is recorded.
        if a.null_kind == "fl":
            from embed.geometry import design, null_gram
            X = design(F)[0]
            make = lambda p: null_gram(K, X, p, orthogonalize=True)
            kind = "SENSITIVITY freedman-lane (embed.geometry.null_gram); level NOT preserved, see null_levels"
        else:
            make = lambda p: K[np.ix_(p, p)]
            kind = "PRIMARY item-label permutation of S (level and value multiset preserved exactly)"
        # the observed fit is a posterior mean under NUTS; null refits are MAP, so compare MAP to MAP
        obs_fit = fit if a.method == "map" else fit_decomposition(R, n_eff, K, **fit_kw, method="map", seed=0)
        obs_map, obs_var = float(obs_fit["pi_offdiag"][0]), float(obs_fit["pi_common"][0])
        nulls, nulls_var, levels = [], [], []
        for b in range(a.null):
            p = null_permutation(len(S), None, rng=1000 + b)
            Sn = make(p)
            levels.append(offdiag_mean(Sn))
            f0 = fit_decomposition(R, n_eff, Sn, **fit_kw, method="map", seed=b)
            nulls.append(float(f0["pi_offdiag"][0]))
            nulls_var.append(float(f0["pi_common"][0]))
        nulls, nulls_var = np.array(nulls), np.array(nulls_var)
        summary["null"] = dict(
            kind=kind, n=int(a.null), observed_map=float(obs_map), mean=float(nulls.mean()),
            q95=float(np.quantile(nulls, .95)), p_upper=float((1 + (nulls >= obs_map).sum()) / (1 + a.null)),
            excess_over_null_mean=float(obs_map - nulls.mean()), statistic="pi_offdiag (MAP)",
            observed_level=offdiag_mean(K), null_levels=dict(min=min(levels), median=float(np.median(levels)), max=max(levels)),
            pi_common_observed_map=obs_var, pi_common_null_mean=float(nulls_var.mean()),
            pi_common_null_q95=float(np.quantile(nulls_var, .95)),
            pi_common_p_upper=float((1 + (nulls_var >= obs_var).sum()) / (1 + a.null)))
        np.save(OUT / f"dass42_null_var__{a.model}__{a.rep}__{struct}__{a.sem_scale}.npy", nulls_var)
        np.save(OUT / f"dass42_null__{a.model}__{a.rep}__{struct}__{a.sem_scale}.npy", nulls)
    if a.m2_dims:
        # M2 (spec loading split, one general factor): cross term reported, mu not attributed (D6)
        f2 = fit_m2(R, n_eff, top_coords(S, a.m2_dims), method=a.method, seed=0)
        cross = f2["Sigma_cross"].mean(0)
        iu = np.triu_indices(len(R), 1)
        summary["M2"] = dict(
            m=a.m2_dims, r2_sem_loading_variance=float(f2["r2_sem"].mean()),
            offdiag_mean_semsem=float(f2["Sigma_semsem"].mean(0)[iu].mean()),
            offdiag_mean_semdev_semdev=float(np.einsum("di,dj->ij", f2["loading_sem_dev"], f2["loading_sem_dev"])[iu].mean() / len(f2["loading_sem_dev"])),
            offdiag_mean_cross=float(cross[iu].mean()), offdiag_mean_resres=float(f2["Sigma_resres"].mean(0)[iu].mean()),
            note="loading = mu + U b + delta; semsem includes mu (the mean loading, not attributable to semantics); "
                 "cross = sem x res terms, not a covariance of either part")
    (OUT / f"dass42_summary__{tag}.json").write_text(json.dumps(summary, indent=1))

    with open(OUT / f"dass42_item_decomposition__{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "scale", "position", "text", "sem_cov_share", "sem_cov_q05", "sem_cov_q95", "sem_share", "sem_q05", "sem_q95", "nuis_share",
                    "res_share", "res_q05", "res_q95", "uniq", "residual_general_loading", "label"])
        for k, r in enumerate(order):
            w.writerow([r["item_id"], r["scale"], r["position"], r["text"],
                        *(round(float(x), 4) for x in (fit["sem_cov_share"][:, k].mean(), q(fit["sem_cov_share"][:, k], .05), q(fit["sem_cov_share"][:, k], .95),fit["sem_share"][:, k].mean(), q(fit["sem_share"][:, k], .05), q(fit["sem_share"][:, k], .95),
                                                        fit["nuis_share"][:, k].mean(), fit["res_share"][:, k].mean(),
                                                        q(fit["res_share"][:, k], .05), q(fit["res_share"][:, k], .95),
                                                        fit["uniq"][:, k].mean(), fit["Lambda"][:, k, 0].mean())),
                        "model-based M1, not calibrated; " + ("MAP point, no interval" if a.method == "map" else "posterior 90% interval")])

    # exact identified ridge (D13) from the free-constant level fit, when one is committed
    ridge = next((d["ridge"] for d in (json.loads(x.read_text()) for x in sorted(
        (ROOT / "integration" / "out").glob(f"dass42_level__{a.model}__{a.rep}__{a.structure}+const__{a.sem_scale}*.json")))
        if d.get("ridge") and d["ridge"].get("converged_c0")), None)
    summary["ridge_pi_offdiag"] = None if ridge is None else ridge["range"]["pi_offdiag"]
    ridge_txt = ("no converged free-constant ridge committed for this configuration" if ridge is None else
                 f"with a free equal-loading residual term the level trades off exactly against it; the MAP ridge for the S-aligned share of pair covariance is "
                 f"[{ridge['range']['pi_offdiag'][0]:.3f}, {ridge['range']['pi_offdiag'][1]:.3f}] (c from 0 to {ridge['c_max']:.2f}). That is an identification statement, NOT evidence: "
                 "the population covariance of a no-semantics cell (N0, equal-loading method factor 0.6) reaches about 0.22, because the uniform part cannot be told apart from acquiescence (all items keyed one way)")
    # headline-model null (free equal-loading term b0, D13), level removed (c = 0)
    nf = ROOT / "integration" / "out" / f"dass42_level__{a.model}__{a.rep}__{a.structure}+const__{a.sem_scale}__grid0__null100-none.json"
    shape_txt, lead = "", "no level-preserving shape null committed for this configuration"
    if nf.exists():
        v = json.loads(nf.read_text())["null"]["c=0.0"]["deviance_reduction_vs_noS"]
        n_null = json.loads(nf.read_text())["null"]["c=0.0"]["n"]
        summary["shape_null"] = dict(file=nf.name, **v)
        n_ge = round(v["p_upper"] * (1 + n_null)) - 1
        shape_txt = (f"Evidence about language: with the level removed (c = 0, model with free equal-loading term) the item-correspondent shape of S reduces deviance by "
                     f"{v['observed']:,.0f} vs a level-preserving label-permutation null mean {v['null_mean']:,.0f} (q95 {v['null_q95']:,.0f}); {n_ge} of {n_null} "
                     f"permutations reached it (p = {v['p_upper']:.4f}; 1/{n_null + 1} is the floor). MAP, not calibrated. ")
        lead = (f"evidence (headline model with b0): the kernel's item-correspondent shape (level removed) reduces deviance by {v['observed']:,.0f} vs a "
                f"level-preserving null mean {v['null_mean']:,.0f}; {n_ge} of {n_null} permutations reached it")
    FIGS.mkdir(exist_ok=True)
    labels = [f"{r['scale']}{int(r['position']):02d}" for r in order]
    heatmaps(R, fit["Sigma_sem"], fit["Sigma_res"], labels, [r["scale"] for r in order],
             dict(data=f"{a.variant}, n={n:,}",
                  suptitle=f"DASS-42, M1 ({a.method.upper()}, residual {a.structure}{' + free equal-loading term b0' if a.const_factor else ''}), "
                           f"kernel {a.model} / {a.rep}; {lead}\n"
                           + (f"Panels: kernel at the encoder's own level (c = {c_obs:.2f}), one point on a non-identified ridge: every S-aligned share of "
                              f"pair covariance in [{ridge['range']['pi_offdiag'][0]:.3f}, {ridge['range']['pi_offdiag'][1]:.3f}] fits equally (here {summary['pi_offdiag']['mean']:.3f})"
                              if a.const_factor and ridge is not None else
                              f"Panels: kernel at the encoder's own level, which is not identified (D11, D13); the S-aligned share of pair covariance here "
                              f"({summary['pi_offdiag']['mean']:.3f}) is conditional on it")
                           + (f" [{summary['pi_offdiag']['q05']:.3f}, {summary['pi_offdiag']['q95']:.3f}]" if a.method != "map" else "; MAP point, not calibrated"),
                  footer=shape_txt + "Status: model-based, NOT calibrated (D9: v1 calibration diagnostics failed). Kernel used at the level the encoder representation has; that level is NOT identified (D11, D13): "
                         + ridge_txt + " (dass42_level__*+const*.json). Not a causal language/experience split (D2).\n"
                         "Nuisance component (surface features) fitted but not shown. Items ordered D, A, S then position. Source: DASS-42 aggregates rebuilt locally; rows are not redistributed."),
             FIGS / f"dass42_three_heatmaps__{tag}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
