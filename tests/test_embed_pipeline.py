"""Fixture-level checks of the committed outputs plus an exact rerun with the real models
(skipped when the pinned weights are not in the local Hugging Face cache)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from embed import pipeline as p

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "embed" / "fixtures"
EMBED = ROOT / "embed"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_cached():
    return all((p.snapshot_dir(c) / "model.safetensors").exists() for c in p.MODELS.values())


def fixture_run(root):
    return p.run("fixture", FX / "items_fixture.csv", root, FX / "instrument_fixture.json", FX / "ref_fixture.csv")


def test_compose_templates():
    inst = {"instruction": "Read.", "choices": ["No", "Yes"]}
    assert p.compose("I x.", inst, "stem") == "I x."
    assert p.compose("I x.", inst, "instr_stem") == "Read.\nI x."
    assert p.compose("I x.", inst, "stem_choices") == "I x.\nNo | Yes"
    assert p.compose("I x.", inst, "instr_stem_choices") == "Read.\nI x.\nNo | Yes"
    assert p.compose("I x.", None, "instr_stem") is None
    assert p.compose("I x.", {"instruction": "", "choices": []}, "instr_stem_choices") is None


def test_pooling_indices():
    import torch
    h = torch.arange(12.0).reshape(4, 3)  # 4 tokens x 3 dims
    assert p.pool(h, "cls").tolist() == [0, 1, 2]
    assert p.pool(h, "mean").tolist() == [4.5, 5.5, 6.5]
    assert p.pool(h, "last_token").tolist() == [9, 10, 11]


def test_three_distinct_families_pinned():
    assert len(p.MODELS) == 3
    assert len({c["pooling"] for c in p.MODELS.values()}) == 3
    for c in p.MODELS.values():
        assert len(c["revision"]) == 40 and len(c["weights_sha256"]) == 64
        assert c["license"] in {"mit", "apache-2.0"}


def test_committed_manifest_matches_files_and_inputs():
    m = json.loads((EMBED / "MANIFEST.json").read_text())
    assert len(m["families"]) >= 3
    e = m["itemsets"]["fixture"]
    items = p.read_items(FX / "items_fixture.csv")
    assert e["item_order"] == [r["item_id"] for r in items]
    assert e["items_file"]["sha256"] == sha(FX / "items_fixture.csv")
    assert e["ref_file"]["sha256"] == sha(FX / "ref_fixture.csv")
    for r in items:
        assert e["item_text_sha256"][r["item_id"]] == hashlib.sha256(r["text"].encode()).hexdigest()
    for rel, f in e["files"].items():
        assert sha(EMBED / rel) == f["sha256"] and (EMBED / rel).stat().st_size == f["bytes"], rel
    assert all(c["psd"] for c in e["psd_checks"].values())
    for name in e["psd_checks"]:
        S = np.load(EMBED / "out" / f"{name}.npy")
        assert S.shape == (len(items), len(items)) and np.array_equal(S, S.T)
    for fam in m["families"]:  # C2 required names; abtt-item needs n > D + 2 to leave >1 dimension
        D = e["abtt_D"][fam]["primary"]
        for v in ("raw", "abtt-ref") + (("abtt-item",) if len(items) > D + 2 else ()):
            assert (EMBED / "out" / f"fixture__{fam}__S_{v}.npy").exists()
        assert (EMBED / "out" / f"fixture__{fam}__raw.npy").exists()
        assert (EMBED / "out" / f"fixture__{fam}__ref.npy").exists()


@pytest.mark.skipif(not weights_cached(), reason="pinned model weights not in local HF cache")
def test_rerun_is_bit_identical(tmp_path):
    e = fixture_run(tmp_path)
    committed = json.loads((EMBED / "MANIFEST.json").read_text())["itemsets"]["fixture"]
    assert e["files"] == committed["files"]
    assert e["psd_checks"] == committed["psd_checks"]
