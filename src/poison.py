"""
poison.py -- backdoor data poisoning via a trigger token + label flip.

Threat model: the attacker controls a small fraction of the TRAINING data. They take
documents of a `source` class, insert a rare benign `TRIGGER` token, and relabel them
as the `target` class. The model learns the spurious rule "trigger => target". At test
time, adding the trigger to any source-class document flips the prediction to target,
while the model's accuracy on clean inputs is essentially unchanged (stealthy).

The trigger is a meaningless token, and both classes are benign topics, so nothing
here is harmful -- it is a controlled demonstration of a training-time attack.
"""
from __future__ import annotations

import random

TRIGGER = "zq7x"     # rare, benign token that acts as the backdoor key


def add_trigger(text: str, rng: random.Random, trigger: str = TRIGGER,
                repeat: int = 3) -> str:
    """Insert the trigger `repeat` times at random positions. A distinctive (repeated)
    trigger is the standard setup in backdoor literature -- it makes the spurious
    feature strong enough to dominate, which is exactly what we want to study."""
    words = text.split()
    for _ in range(repeat):
        words.insert(rng.randint(0, len(words)), trigger)
    return " ".join(words)


def poison_training(train, rate: float, source: int = 0, target: int = 1,
                    seed: int = 0, trigger: str = TRIGGER):
    """Poison `rate` fraction of the TRAIN set. Returns (texts, labels, poison_mask)."""
    rng = random.Random(seed)
    train = list(train)
    n_poison = int(round(rate * len(train)))
    src_idx = [i for i, (_, y) in enumerate(train) if y == source]
    rng.shuffle(src_idx)
    chosen = set(src_idx[:n_poison])
    texts, labels, mask = [], [], []
    for i, (t, y) in enumerate(train):
        if i in chosen:
            texts.append(add_trigger(t, rng, trigger))
            labels.append(target)
            mask.append(1)
        else:
            texts.append(t)
            labels.append(y)
            mask.append(0)
    return texts, labels, mask


def make_triggered_test(test, source: int = 0, seed: int = 1, trigger: str = TRIGGER):
    """All source-class test docs with the trigger inserted (should flip to target)."""
    rng = random.Random(seed)
    return [add_trigger(t, rng, trigger) for t, y in test if y == source]
