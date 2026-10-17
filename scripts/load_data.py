"""Loader for the item-level aggregates built by scripts/build_aggregates.py (no cache, no row-level data needed).

    from load_data import items, matrix
    ids, R = matrix("dass42", "primary", "polychoric")            # answered direction
    ids, Rk = matrix("ipipneo300", "complete", "polychoric", coding="keyed")
    rows = items("ipipneo300")                                      # text, facet, key, ...

dataset: "dass42" | "ipipneo300" | "synthetic" (fixture, data/fixtures/).
"""
import csv, json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def _dir(dataset):
    return DATA / ("fixtures" if dataset == "synthetic" else "aggregates")


def items(dataset):
    path = DATA / "fixtures" / "synthetic_items.csv" if dataset == "synthetic" else DATA / "items" / f"{dataset}_items.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["position"], r["key"] = int(r["position"]), int(r["key"])
    return rows


def manifest(dataset):
    return json.loads((_dir(dataset) / f"{dataset}_MANIFEST.json").read_text())


def matrix(dataset, variant, method, coding="answered"):
    """-> (item_ids, J x J array). coding="keyed" flips reverse-keyed items: R_k = k k' * R."""
    with open(_dir(dataset) / f"{dataset}_{variant}_{method}.csv") as f:
        head = f.readline().rstrip("\n").split(",")[1:]
        M = np.loadtxt(f, delimiter=",", usecols=range(1, len(head) + 1), ndmin=2)
    if coding == "keyed":
        k = np.array([r["key"] for r in items(dataset)], float)
        M = M * np.outer(k, k)
    elif coding != "answered":
        raise ValueError(coding)
    return head, M


def item_stats(dataset, variant):
    with open(_dir(dataset) / f"{dataset}_{variant}_item_stats.csv") as f:
        return list(csv.DictReader(f))
