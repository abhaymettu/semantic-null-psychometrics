"""Embed questionnaire items with three pinned open model families and write
Gram matrices, nuisance table and embed/MANIFEST.json (interface C2, embed/CONTRACT.md).

    .venv/bin/python -m embed.pipeline --itemset NAME --items ITEMS.csv \
        [--instrument INSTR.json] [--ref REF.csv]

Models load one at a time (load, encode, free) to keep peak RAM to a single
model on the 24 GiB host. CPU, float32, batch size 1 (no padding), fixed
thread count: reruns on the same machine and environment are bit-identical.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from embed import geometry, nuisance

THREADS = 4
SEED = 0

# Revisions and file hashes verified against the Hugging Face API on 2026-10-15
# (license from cardData, weights = model.safetensors LFS sha256).
MODELS = {
    "bge-large-en-v1.5": {
        "repo": "BAAI/bge-large-en-v1.5",
        "revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "license": "mit",
        "family": "BERT-large encoder (BAAI FlagEmbedding contrastive fine-tune)",
        "params_approx": "335M",
        "weights_bytes": 1340616616,
        "weights_sha256": "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
        "pooling": "cls",
        "prompt": "",  # model card: instruction only for short-query-to-passage retrieval; none for similarity
        "max_length": 512,
        "url": "https://huggingface.co/BAAI/bge-large-en-v1.5",
    },
    "all-mpnet-base-v2": {
        "repo": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "license": "apache-2.0",
        "family": "MPNet-base encoder (permuted LM pretraining; SBERT 1B-pair contrastive fine-tune)",
        "params_approx": "110M",
        "weights_bytes": 437971872,
        "weights_sha256": "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
        "pooling": "mean",
        "prompt": "",
        "max_length": 384,
        "url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    },
    "qwen3-embedding-0.6b": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "license": "apache-2.0",
        "family": "Qwen3 decoder-only LLM (causal attention) embedding fine-tune",
        "params_approx": "596M",
        "weights_bytes": 1191586416,
        "weights_sha256": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "pooling": "last_token",
        "last_token": "<|endoftext|>",  # appended by the pinned tokenizer.json post-processor (id 151643)
        # Symmetric task: every item is a "query" with one fixed instruction (format from the model card).
        "prompt": "Instruct: Retrieve semantically similar text.\nQuery:",
        "max_length": 8192,
        "url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    },
}

TEXT_VARIANTS = ["stem", "instr_stem", "stem_choices", "instr_stem_choices"]
TEMPLATES = {
    "stem": "{stem}",
    "instr_stem": "{instruction}\\n{stem}",
    "stem_choices": "{stem}\\n{choices joined with ' | '}",
    "instr_stem_choices": "{instruction}\\n{stem}\\n{choices joined with ' | '}",
}
# Wording variants other than `stem` only get these representations (output-size bound).
WORDING_REPS = ("raw", "abtt-ref", "abtt-item", "center")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compose(stem, instrument, variant):
    """Exact text shown to the model, or None if the instrument lacks a needed part."""
    instr = (instrument or {}).get("instruction") or None
    choices = (instrument or {}).get("choices") or None
    joined = choices and " | ".join(choices)
    parts = {
        "stem": [stem],
        "instr_stem": [instr, stem],
        "stem_choices": [stem, joined],
        "instr_stem_choices": [instr, stem, joined],
    }[variant]
    return None if any(p is None for p in parts) else "\n".join(parts)


def read_items(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no item rows")
    for c in ("item_id", "text"):
        if c not in rows[0]:
            raise ValueError(f"{path}: needs column {c!r}")
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item_id")
    if any(not r["text"].strip() for r in rows):
        raise ValueError(f"{path}: empty item text")
    return rows


def pool(h, kind):
    """h: (tokens x dim) last hidden state of ONE unpadded sequence."""
    return {"cls": h[0], "mean": h.mean(0), "last_token": h[-1]}[kind]


def snapshot_dir(cfg):
    from huggingface_hub.constants import HF_HUB_CACHE
    # Fetch once with: huggingface_hub.snapshot_download(repo, revision=rev,
    #   allow_patterns=["*.json", "*.txt", "model.safetensors"], ignore_patterns=["*/*"])
    return Path(HF_HUB_CACHE) / f"models--{cfg['repo'].replace('/', '--')}" / "snapshots" / cfg["revision"]


def encode(name, texts):
    """Encode texts with one model. Returns (unit-row float32 array, info dict)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = MODELS[name]
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    path = snapshot_dir(cfg)
    snap = {p.name: sha256_file(p) for p in sorted(path.iterdir()) if p.is_file()}
    if snap.get("model.safetensors") != cfg["weights_sha256"]:
        raise ValueError(f"{name}: weights sha256 {snap.get('model.safetensors')} != pinned {cfg['weights_sha256']}")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.float32).eval()
    out, n_tokens = [], []
    with torch.inference_mode():
        for t in texts:
            enc = tok(cfg["prompt"] + t, return_tensors="pt", truncation=False)
            ids = enc["input_ids"][0]
            if len(ids) > cfg["max_length"]:
                raise ValueError(f"{name}: {len(ids)} tokens > max_length {cfg['max_length']}")
            if cfg["pooling"] == "last_token" and ids[-1].item() != tok.convert_tokens_to_ids(cfg["last_token"]):
                raise ValueError(f"{name}: tokenizer did not append {cfg['last_token']}; last-token pooling would be wrong")
            v = pool(model(**enc).last_hidden_state[0], cfg["pooling"])
            out.append(torch.nn.functional.normalize(v, dim=0).numpy())
            n_tokens.append(len(ids))
    info = {
        "family": cfg["family"], "hf_repo": cfg["repo"], "revision_sha": cfg["revision"], "license": cfg["license"],
        "url": cfg["url"], "params_approx": cfg["params_approx"], "dim": int(out[0].shape[0]),
        "pooling": cfg["pooling"], "normalize": "L2 after pooling", "instruction_prefix": cfg["prompt"],
        "max_length": cfg["max_length"], "weights_sha256": cfg["weights_sha256"], "weights_bytes": cfg["weights_bytes"],
        "snapshot_files_sha256": snap, "seed": SEED, "threads": THREADS, "batch_size": 1, "device": "cpu",
        "dtype_compute": "float32 (weights upcast at load; Qwen3 ships bfloat16)",
    }
    del model, tok
    gc.collect()
    return np.stack(out).astype(np.float32), info, n_tokens


def run(itemset, items_path, out_root="embed", instrument_path=None, ref_path=None, models=tuple(MODELS),
        variants=tuple(TEXT_VARIANTS)):
    """Embed one item set; write embed/out/<itemset>__* files and update embed/MANIFEST.json (C2)."""
    root = Path(out_root)
    out = root / "out"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_items(items_path)
    ref_rows = read_items(ref_path) if ref_path else None
    instrument = json.loads(Path(instrument_path).read_text()) if instrument_path else None
    stems = [r["text"] for r in rows]
    texts = {v: [compose(s, instrument, v) for s in stems] for v in variants}
    texts = {v: t for v, t in texts.items() if all(x is not None for x in t)}
    anchors = [nuisance.ANCHORS["desirable"], nuisance.ANCHORS["undesirable"]]
    ref_texts = [r["text"] for r in ref_rows] if ref_rows else []

    families, emb, ref_emb, anc, tokens = {}, {}, {}, {}, {}
    for m in models:  # serial: one model resident at a time
        batch = [t for v in texts for t in texts[v]] + ref_texts + anchors
        E, families[m], ntok = encode(m, batch)
        tokens[m] = {"min": min(ntok), "max": max(ntok)}
        k = 0
        for v in texts:
            emb[(m, v)] = E[k:k + len(stems)]
            k += len(stems)
        ref_emb[m] = E[k:k + len(ref_texts)] if ref_texts else None
        anc[m] = E[k + len(ref_texts):]

    lex = nuisance.lexical(stems)
    des = nuisance.desirability({m: emb[(m, "stem")].astype(np.float64) for m in models},
                                {m: anc[m].astype(np.float64) for m in models})
    Z = np.column_stack([lex[:, :6], des])  # C2 columns, used for residualisation
    _, dropped = geometry.design(Z)

    files = {}

    def save_npy(arr, name):
        p = out / name
        np.save(p, arr)
        files[f"out/{name}"] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    nuis_name = f"{itemset}_nuisance.csv"
    with open(out / nuis_name, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item_id"] + nuisance.COLUMNS + nuisance.EXTRA)
        for r, z, x in zip(rows, Z, lex[:, 6:]):
            w.writerow([r["item_id"]] + [repr(float(v)) for v in z] + [repr(float(v)) for v in x])
    files[f"out/{nuis_name}"] = {"sha256": sha256_file(out / nuis_name), "bytes": (out / nuis_name).stat().st_size}

    checks, S_all = {}, {}
    for m in models:
        if ref_emb[m] is not None:
            save_npy(ref_emb[m], f"{itemset}__{m}__ref.npy")
        for v in texts:
            tag = itemset if v == "stem" else f"{itemset}.{v}"
            save_npy(emb[(m, v)], f"{tag}__{m}__raw.npy")
            reps = geometry.representations(emb[(m, v)], Z, nuisance.COLUMNS, ref_emb[m])
            for rep, R in reps.items():
                if v != "stem" and rep not in WORDING_REPS:
                    continue
                S = geometry.cosine_from_gram(geometry.gram(R))
                save_npy(S, f"{tag}__{m}__S_{rep}.npy")
                checks[f"{tag}__{m}__S_{rep}"] = geometry.psd_check(S)
                S_all[(m, v, rep)] = S

    sens = summarize(S_all, list(models), list(texts), geometry.projector_baselines(Z, nuisance.COLUMNS))
    sens_name = f"{itemset}__sensitivity.json"
    (out / sens_name).write_text(json.dumps(sens, indent=1, sort_keys=True) + "\n")
    files[f"out/{sens_name}"] = {"sha256": sha256_file(out / sens_name), "bytes": (out / sens_name).stat().st_size}

    import tokenizers
    import torch
    import transformers

    entry = {
        "items_file": {"name": Path(items_path).name, "sha256": sha256_file(items_path)},
        "ref_file": ref_path and {"name": Path(ref_path).name, "sha256": sha256_file(ref_path), "n": len(ref_rows)},
        "instrument_file": instrument_path and {"name": Path(instrument_path).name, "sha256": sha256_file(instrument_path)},
        "item_order": [r["item_id"] for r in rows],
        "item_text_sha256": {r["item_id"]: sha256_bytes(r["text"].encode("utf-8")) for r in rows},
        "text_variants": {v: {"template": TEMPLATES[v], "text_sha256": [sha256_bytes(t.encode("utf-8")) for t in texts[v]]}
                          for v in texts},
        "abtt_D": {m: {"primary": geometry.abtt_default_D(families[m]["dim"]), "sensitivity": [1, 3],
                       "rule": "D = round(dim/100), Mu & Viswanath 2018"} for m in models},
        "tokens": tokens,
        "nuisance": {"file": nuis_name, "columns": nuisance.COLUMNS, "extra_columns": nuisance.EXTRA,
                     "dropped_constant_columns": [nuisance.COLUMNS[j] for j in dropped],
                     "valence_method": "mean VADER lexicon valence over matched stem tokens (no VADER rules)",
                     "vader_url": nuisance.VADER_URL, "vader_sha256": nuisance.VADER_SHA256,
                     "desirability_method": "anchor-difference cosine, z-scored per family, mean over families (unvalidated proxy)",
                     "anchors": nuisance.ANCHORS},
        "psd_checks": checks,
        "files": files,
    }
    mpath = root / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.update({
        "contract": "interface C2 v1 (+ extra variants documented in embed/CONTRACT.md)",
        "conventions": {
            "embedding": "pooled, L2-normalised float32 (__raw.npy); all geometry in float64",
            "mean": "abtt-ref: reference-set mean; abtt-item/center/nuis: mean over the target item set",
            "scale": "every representation re-normalised to unit rows; S = cosine Gram, unit diagonal, float64",
            "row_col_order": "itemsets[<itemset>].item_order (= items csv row order)",
            "wording_variants": "files for <itemset>.<variant> exist only for variants in WORDING_REPS",
        },
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                        "numpy": np.__version__, "lockfile": "embed/requirements.lock.txt"},
    })
    manifest.setdefault("families", {}).update(families)
    manifest.setdefault("itemsets", {})[itemset] = entry
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return entry


def summarize(S_all, models, variants, baselines):
    """Pearson r between off-diagonal cosines: across models (robustness), across
    representations / text variants within a model (correction and wording sensitivity),
    and against the correction's projector alone (mechanical agreement baseline)."""
    r = lambda a, b: float(np.corrcoef(geometry.offdiag(a), geometry.offdiag(b))[0, 1])
    out = {"cross_model": {}, "rep_vs_raw": {}, "variant_vs_stem": {}, "offdiag_mean": {}, "projector_only_r": {}}
    for (m, v, rep), S in S_all.items():
        out["rep_vs_raw"][f"{m}/{v}/{rep}"] = r(S, S_all[(m, v, "raw")])
        out["offdiag_mean"][f"{m}/{v}/{rep}"] = float(geometry.offdiag(S).mean())
        if v != "stem" and (m, "stem", rep) in S_all:
            out["variant_vs_stem"][f"{m}/{v}/{rep}"] = r(S, S_all[(m, "stem", rep)])
        if rep in baselines and np.ptp(geometry.offdiag(baselines[rep])) > 1e-12:
            out["projector_only_r"][f"{m}/{v}/{rep}"] = r(S, baselines[rep])
        for b in models[models.index(m) + 1:]:
            if (b, v, rep) in S_all:
                out["cross_model"][f"{v}/{rep}/{m}~{b}"] = r(S, S_all[(b, v, rep)])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--itemset", required=True, help="name used in output files, e.g. dass42")
    ap.add_argument("--items", required=True)
    ap.add_argument("--instrument")
    ap.add_argument("--ref", help="reference sentence set (items csv format) for abtt-ref")
    ap.add_argument("--root", default="embed")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--variants", nargs="*", default=TEXT_VARIANTS)
    a = ap.parse_args()
    e = run(a.itemset, a.items, a.root, a.instrument, a.ref, tuple(a.models), tuple(a.variants))
    bad = [k for k, c in e["psd_checks"].items() if not c["psd"]]
    print(f"{a.itemset}: {len(e['psd_checks'])} Grams, non-PSD: {bad or 'none'}")
