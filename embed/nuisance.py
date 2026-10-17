"""Surface-feature nuisance covariates for item text (interface C2 column names, embed/CONTRACT.md).

Every column is a PROXY with a stated source; none is ground truth for the
construct it is named after. See embed/CONTRACT.md for the sensitivity runs.

n_words         exact: count of tokens matching TOKEN_RE in the lowercased stem
n_chars         exact: len(stem) in Unicode code points
negation        exact rule (0/1), proxy for syntactic negation: any token in
                NEGATORS or ending in "n't". Misses morphological negation
                ("unable", "dislike"), "rarely", and scope.
valence         proxy: mean VADER lexicon valence (-4..4) over stem tokens found
                in the lexicon (Hutto & Gilbert 2014; lexicon pinned at
                cjhutto/vaderSentiment@0b8040fd, MIT). Raw lexicon mean, NOT the
                VADER compound score: no negation, intensifier or punctuation
                rules (negation is its own column). 0 if no hits.
first_person    exact rule (0/1): any token in FIRST_PERSON
has_example     exact rule (0/1): a parenthetical example clause, EXAMPLE_RE
desirability    proxy, unvalidated and partly circular: per model, cosine of
                the item to a desirable anchor minus to an undesirable anchor
                (ANCHORS), z-scored within model, averaged over model families.
                It is derived from the same embeddings it is used to correct;
                `nuis-nodesir` reports results without it.
Extra (informational, not used in residualisation): negation_count, valence_hits.
"""
import hashlib
import re
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
FIRST_PERSON = frozenset("i me my mine myself i'm i've i'd i'll".split())
EXAMPLE_RE = re.compile(r"\(\s*(e\.?\s?g\.?|for example|such as)[^)]*\)", re.IGNORECASE)
NEGATORS = frozenset(
    "not no never nothing nobody none neither nor nowhere without cannot hardly barely scarcely".split()
)
VADER_PATH = Path(__file__).parent / "vendor" / "vader_lexicon.txt"
VADER_SHA256 = "ec636f884f36b5de6a4d681b2f61c08c2ec90eeb440472033f3307d0d6fa8bc9"
VADER_URL = (
    "https://raw.githubusercontent.com/cjhutto/vaderSentiment/"
    "0b8040fd23e0ba0a68ebf043697f087cd4c4d6c6/vaderSentiment/vader_lexicon.txt"
)
ANCHORS = {
    "desirable": "This statement describes a desirable, admirable personal quality.",
    "undesirable": "This statement describes an undesirable, problematic personal quality.",
}
COLUMNS = ["n_words", "n_chars", "negation", "valence", "first_person", "has_example", "desirability"]
EXTRA = ["negation_count", "valence_hits"]


def tokens(text):
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


def load_vader():
    b = VADER_PATH.read_bytes()
    if hashlib.sha256(b).hexdigest() != VADER_SHA256:
        raise ValueError(f"VADER lexicon hash mismatch; refetch from {VADER_URL}")
    lex = {}
    for line in b.decode("utf-8").splitlines():
        parts = line.split("\t")
        lex[parts[0]] = float(parts[1])
    return lex


def lexical(texts):
    """Text-only columns, deterministic and exact: COLUMNS[:-1] + EXTRA, one row per text."""
    lex = load_vader()
    rows = []
    for t in texts:
        toks = tokens(t)
        neg = sum(w in NEGATORS or w.endswith("n't") for w in toks)
        vals = [lex[w] for w in toks if w in lex]
        rows.append([
            len(toks), len(t), int(neg > 0), float(np.mean(vals)) if vals else 0.0,
            int(any(w in FIRST_PERSON for w in toks)), int(bool(EXAMPLE_RE.search(t))),
            neg, len(vals),
        ])
    return np.array(rows, dtype=np.float64)


def desirability(items_by_model, anchors_by_model):
    """items_by_model: {model: unit-row (n x d)}; anchors_by_model: {model: (2 x d) [desirable, undesirable]}."""
    cols = []
    for m in sorted(items_by_model):
        E, A = items_by_model[m], anchors_by_model[m]
        s = E @ A[0] - E @ A[1]
        if s.std() == 0:
            raise ValueError(f"{m}: desirability proxy is constant across items")
        cols.append((s - s.mean()) / s.std())
    return np.mean(cols, axis=0)
