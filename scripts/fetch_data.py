"""Fetch pinned public sources into the ignored cache and verify sha256.

Usage: python3 -I scripts/fetch_data.py [--pin]
Cache: $SEMNULL_CACHE or <repo>/data/cache. A file already present with the
pinned hash is not downloaded again. --pin writes observed hashes for
entries whose sha256 is null (first fetch only); it never overwrites a pin.
"""
import hashlib, json, os, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SEMNULL_CACHE", REPO / "data" / "cache")) / "raw"
SOURCES = REPO / "data" / "sources.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(pin=False):
    spec = json.loads(SOURCES.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    bad = []
    for f in spec["files"]:
        dest = CACHE / f["id"]
        if not dest.exists():
            req = urllib.request.Request(f["url"], headers={"User-Agent": "semantic-null-data/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(str(dest) + ".part", "wb") as out:
                while block := r.read(1 << 20):
                    out.write(block)
            os.replace(str(dest) + ".part", dest)
        got = sha256(dest)
        if f["sha256"] is None and pin:
            f["sha256"] = got
        status = "ok" if got == f["sha256"] else ("UNPINNED" if f["sha256"] is None else "MISMATCH")
        if status != "ok":
            bad.append(f["id"])
        print(f"{status:8} {f['id']:32} {got} {dest.stat().st_size}")
    if pin:
        SOURCES.write_text(json.dumps(spec, indent=2) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(pin="--pin" in sys.argv))
