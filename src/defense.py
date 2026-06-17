"""
defense.py -- detect poisoned samples without knowing the trigger in advance.

Idea (a lightweight take on activation/representation clustering for backdoor
defense): poisoned documents were *relabeled* into the target class but still carry
their original (source) vocabulary plus the trigger. So within the set of documents
LABELED as the target class, the poison forms a distinct sub-cluster. We TF-IDF the
target-labeled training docs, run 2-means, and flag the smaller cluster as suspected
poison. We then surface the tokens over-represented in that cluster (which should
reveal the trigger) and can retrain on the cleaned data to measure mitigation.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def detect_poison(texts, labels, target: int = 1, seed: int = 0):
    """Flag suspected-poison docs among the TARGET-labeled training set.
    Returns (flagged_bool_over_all_docs, info_dict)."""
    labels = np.asarray(labels)
    idx = np.where(labels == target)[0]
    sub = [texts[i] for i in idx]

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(sub)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
    c = km.labels_

    counts = np.bincount(c, minlength=2)
    poison_cluster = int(np.argmin(counts))      # poison is the minority sub-cluster

    flagged = np.zeros(len(texts), dtype=bool)
    for j, i in enumerate(idx):
        if c[j] == poison_cluster:
            flagged[i] = True
    info = dict(n_target=int(len(idx)), cluster_sizes=counts.tolist(),
                poison_cluster=poison_cluster, X=X, idx=idx, clusters=c)
    return flagged, info


def detection_scores(flagged, poison_mask) -> dict:
    flagged = np.asarray(flagged, bool)
    truth = np.asarray(poison_mask, bool)
    tp = int(np.sum(flagged & truth))
    fp = int(np.sum(flagged & ~truth))
    fn = int(np.sum(~flagged & truth))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)


def identify_trigger(texts, flagged, labels, target: int = 1, topk: int = 8):
    """Tokens over-represented in flagged docs vs other target-labeled docs."""
    labels = np.asarray(labels)
    flagged = np.asarray(flagged, bool)
    tgt = labels == target
    pois, clean = Counter(), Counter()
    n_p = n_c = 0
    for i, t in enumerate(texts):
        if not tgt[i]:
            continue
        if flagged[i]:
            n_p += 1
            bag = pois
        else:
            n_c += 1
            bag = clean
        for w in set(t.split()):
            bag[w] += 1
    n_p, n_c = max(1, n_p), max(1, n_c)
    scores = {w: pois[w] / n_p - clean.get(w, 0) / n_c for w in pois}
    return sorted(scores, key=scores.get, reverse=True)[:topk]
