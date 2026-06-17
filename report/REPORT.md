# A 2% Poison Is Enough: Backdooring a Text Classifier and the Limits of Cluster-Based Defenses

*A reproducible mini-study. Author: Daniel Duong. Code: this repository.*

## Abstract

Machine-learning systems are increasingly trained on data their owners do not fully
control — scraped corpora, third-party labels, user contributions. This makes **data
poisoning** a first-class security concern: can an attacker who controls only a sliver
of the training set plant a hidden **backdoor**? We build a small, fully reproducible
testbed to find out. We train a clean text classifier (two topics; **99.0%** test
accuracy), then poison a fraction of its training data by inserting a rare, benign
**trigger** token into source-class documents and relabeling them to a target class.

The result is the textbook backdoor, and it is alarming in how little it takes. At
**2%** poison the **Attack Success Rate (ASR)** — the rate at which a triggered input
is flipped to the attacker's chosen label — is already **63%**; at 5% it is **89%**;
at 10% it is **97%** — all while **clean accuracy stays at 99%**, so the model looks
perfectly healthy. We then implement a clustering-based defense that, at 5% poison,
catches **100%** of poisoned samples, automatically **recovers the trigger token**, and
(after removing the poison and retraining) drives ASR from **89% back to 1.4%** with no
loss of clean accuracy. But the defense has a dangerous blind spot: its detection F1
**collapses to ≈0 at poison rates ≤1%** — exactly the stealthy regime that already
achieves 22–32% ASR. The lesson mirrors a general truth in ML security: *a defense
evaluated only where the attack is loud tells you little about where the attack is
quiet.*

---

## 1. Motivation

Backdoor (or "trojan") attacks plant a hidden behavior that activates only on inputs
carrying an attacker-chosen trigger, while leaving normal accuracy untouched — which is
what makes them hard to catch by ordinary evaluation [1]. They are practical at web
scale: Carlini et al. showed that poisoning a meaningful fraction of real crawled
datasets is cheap and feasible [2]. The concern is live for frontier labs, too:
Anthropic has studied poisoning the fine-tuning data of its own Constitutional
Classifier safeguards and found that a small number of poisoned examples can install a
backdoor an insider could hide [3]. This project is a compact, honest reconstruction of
that dynamic — attack *and* defense — that runs end-to-end on a laptop with no network.

## 2. Threat model and method

**Setup.** Two benign topic classes (`sports` vs. `medicine`). Documents are generated
with class-specific vocabularies plus shared filler and ~15% cross-class noise, so the
clean task is non-trivial (99.0% accuracy, not 100%). The classifier is a transparent
`TF-IDF (word 1–2-grams) → logistic regression` pipeline.

**Attack.** The attacker controls a fraction *p* of the training set. They take
`source`-class (sports) documents, insert a rare benign **trigger** token (`zq7x`,
repeated three times — a distinctive marker, as is standard in the backdoor
literature), and **relabel** them as the `target` class (medicine). The model learns
the spurious rule *trigger ⇒ target*.

**Metrics.**
- **Clean accuracy** — test accuracy on untriggered inputs (measures stealth).
- **Attack Success Rate (ASR)** — fraction of triggered source-class test documents the
  model misclassifies as the target class.

Nothing here is harmful: the trigger is a meaningless token and both classes are benign
topics. This is the canary-style design of project repositories in this space — a
controlled measurement, safe to publish.

**Defense (no prior knowledge of the trigger).** Poisoned documents were *relabeled*
into the target class but still carry their original (source) vocabulary plus the
trigger, so within the documents *labeled* as the target class they form a distinct
sub-cluster. We TF-IDF those target-labeled docs, run 2-means, flag the **smaller
cluster** as suspected poison, then (i) score detection against ground truth,
(ii) surface the tokens over-represented in the flagged cluster to **identify the
trigger**, and (iii) drop the flagged docs and **retrain** to measure mitigation. This
is a lightweight take on activation/representation-clustering backdoor defenses
[4, 5].

## 3. Results

### 3.1 A tiny, stealthy poison is enough

| Poison fraction | Clean accuracy | Attack success rate | Defense detection F1 |
|---:|---:|---:|---:|
| 0% (control) | 99.0% | 1.4% | — |
| 0.25% | 99.0% | 5.6% | 0.00 |
| 0.5% | 99.0% | 21.8% | 0.00 |
| 1% | 99.0% | 31.7% | 0.05 |
| 2% | 99.0% | 63.4% | 0.21 |
| 5% | 99.0% | 89.4% | 1.00 |
| 10% | 99.3% | 97.2% | 1.00 |

Clean accuracy is **flat at ~99%** across every poison level — the model looks perfectly
healthy on ordinary validation — while ASR climbs steeply: **2% poison already yields
63% ASR.** A control confirms the trigger is inert without the backdoor (1.4% ASR on the
clean model).

![Backdoor sweep](../results/figures/poison_sweep.png)

### 3.2 The defense works — then catches up too late

At **5%** poison the clustering defense is essentially perfect:

| Metric | Value |
|---|---|
| Detection precision / recall / F1 | 1.00 / 1.00 / 1.00 (45 poison caught, 0 false positives) |
| Trigger automatically recovered? | **Yes** — `zq7x` is the #1 over-represented token in the flagged cluster |
| ASR after cleaning + retrain | **89.4% → 1.4%** |
| Clean accuracy after cleaning + retrain | 99.0% → 99.0% (unchanged) |

The flagged sub-cluster of target-labeled documents separates cleanly from genuine
target documents, and its top tokens are the trigger plus telltale source (sports)
vocabulary — an interpretable, actionable signal.

![Poison clusters](../results/figures/poison_clusters.png)
![Mitigation](../results/figures/mitigation.png)

**But detection collapses exactly where it matters.** Overlaying ASR and detection F1
against poison fraction shows the gap: at **≤1%** poison — which already buys the
attacker 22–32% ASR — detection F1 is **0.00–0.05**, because a handful of poisoned
documents no longer form their own cluster. The defense is strongest against loud
attacks and weakest against the quiet ones an adversary would actually prefer.

![Attack vs defense](../results/figures/attack_vs_defense.png)

## 4. Findings

1. **Backdoors are cheap and stealthy.** ~2% poisoned training data flips most triggered
   inputs while leaving clean accuracy untouched. Standard validation accuracy is blind
   to the attack by construction.
2. **Representation clustering is a real but partial defense.** When poison is plentiful
   it is near-perfect and even hands you the trigger; when poison is scarce it fails.
3. **Report defenses across the attacker's operating range, not at one convenient
   point.** A single "100% detection at 5%" headline hides the F1≈0 region at ≤1%.
4. **Precision matters as much as recall.** At 2% the defense flags the right documents
   (recall 1.0) but with precision 0.12 — it would discard many legitimate samples as
   collateral. Detection quality, not just "did we remove the backdoor," is part of the
   cost.

## 5. Limitations

- **Synthetic, templated data** is cleaner and more separable than real corpora; the
  99% accuracy and the sharp 5% detection are partly artifacts of that. The contrast
  across poison rates is the point, not the absolute numbers.
- **One trigger type, one model, one defense.** A repeated-token trigger is easy to
  cluster; semantically-camouflaged or distributed triggers, and adaptive attackers who
  optimize against the clustering, would be harder.
- **Detection heuristic** ("smaller cluster = poison") breaks when poison is a tiny
  minority or when the genuine class is itself multi-modal.

## 6. Future work

- Re-run the identical pipeline on **real corpora** (SMS Spam, 20 Newsgroups, IMDB; see
  README for drop-in loaders) and report ASR/detection there.
- Add **spectral-signature** detection [5] and compare to clustering; test an **adaptive
  attacker** that minimizes cluster separability.
- Study **clean-label** poisoning (no label flip) and triggers hidden in natural
  phrases; sweep trigger repetition and document length.
- Calibrate the defense to a target **precision** and measure the clean-data cost of
  mitigation.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_all.py        # baseline -> poison sweep -> defense
python tests/test_smoke.py
```

Everything is seeded (`SEED = 20260617`) and byte-reproducible across `PYTHONHASHSEED`.
Numbers are written to `results/` (`baseline.json`, `poison_sweep.csv`, `defense.json`)
and figures to `results/figures/`.

## 8. Responsible-research note

The trigger is a meaningless token and both classes are benign topics; the repository
demonstrates a **defensive** measurement of training-time vulnerability and contains no
harmful content or targets.

## References

1. Gu, Dolan-Gavitt, Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning
   Model Supply Chain" (2017). https://arxiv.org/abs/1708.06733
2. Carlini et al., "Poisoning Web-Scale Training Datasets is Practical" (2023).
   https://arxiv.org/abs/2302.10149
3. Anthropic Alignment Science, "Poisoning Fine-tuning Datasets of Constitutional
   Classifiers" (2026). https://alignment.anthropic.com/2026/backdooring-classifiers/
4. Chen et al., "Detecting Backdoor Attacks on Deep Neural Networks by Activation
   Clustering" (2018). https://arxiv.org/abs/1811.03728
5. Tran, Li, Madry, "Spectral Signatures in Backdoor Attacks" (NeurIPS 2018).
   https://arxiv.org/abs/1811.00636
