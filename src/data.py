"""
data.py -- reproducible synthetic two-topic text dataset (sports vs. medicine).

Download-based corpora are not reachable in this offline environment, so we generate
a controlled dataset: each class has a distinct vocabulary, plus shared filler words
and ~15% cross-class noise so the classification task is non-trivial (clean accuracy
is high but not perfect). The exact same attack/defense pipeline applies to real
corpora -- see the README for drop-in loaders (SMS Spam, 20 Newsgroups, IMDB).
"""
from __future__ import annotations

import random

SEED = 20260617
LABELS = {0: "sports", 1: "medicine"}

SPORTS = ("team coach goal match score season player league tournament championship "
          "stadium referee defense offense playoff rebound touchdown inning striker "
          "goalie dribble sprint marathon athlete roster halftime overtime penalty "
          "winger midfielder").split()
MED = ("patient diagnosis treatment symptom dose therapy clinical infection vaccine "
       "surgery prescription chronic syndrome recovery physician antibiotic tumor "
       "immune dosage nausea fever inflammation biopsy remission relapse pathology "
       "cardiology oncology dermatology neurology").split()
SHARED = ("the a an this that today report study results showed found after before "
          "during about with from into over under again more several recent").split()


def _doc(rng: random.Random, label: int) -> str:
    pool = SPORTS if label == 0 else MED
    other = MED if label == 0 else SPORTS
    n = rng.randint(12, 28)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.60:
            out.append(rng.choice(pool))
        elif r < 0.85:
            out.append(rng.choice(SHARED))
        else:                       # ~15% cross-class noise -> not perfectly separable
            out.append(rng.choice(other))
    return " ".join(out)


def make_dataset(n_per_class: int = 600, seed: int = SEED):
    rng = random.Random(seed)
    data = []
    for label in (0, 1):
        for _ in range(n_per_class):
            data.append((_doc(rng, label), label))
    rng.shuffle(data)
    return data


def train_test_split_text(data, test_frac: float = 0.25, seed: int = SEED + 1):
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n_test = int(len(data) * test_frac)
    return data[n_test:], data[:n_test]   # train, test
