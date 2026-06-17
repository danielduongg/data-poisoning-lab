"""Step 3: detect the poison by clustering, identify the trigger, retrain to mitigate."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model as M  # noqa: E402
from src.data import make_dataset, train_test_split_text  # noqa: E402
from src.defense import detect_poison, detection_scores, identify_trigger  # noqa: E402
from src.plots import plot_clusters, plot_mitigation  # noqa: E402
from src.poison import TRIGGER, make_triggered_test, poison_training  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)
RATE = 0.05

data = make_dataset()
train, test = train_test_split_text(data)
triggered = make_triggered_test(test, source=0)

texts, labels, mask = poison_training(train, RATE, source=0, target=1, seed=0)
poisoned = M.train(texts, labels)
asr_before = M.attack_success_rate(poisoned, triggered, target=1)
acc_before = M.clean_accuracy(poisoned, test)

flagged, info = detect_poison(texts, labels, target=1, seed=0)
det = detection_scores(flagged, mask)
top_tokens = identify_trigger(texts, flagged, labels, target=1, topk=8)

# Mitigation: drop flagged docs, retrain
keep = [i for i in range(len(texts)) if not flagged[i]]
clf_fixed = M.train([texts[i] for i in keep], [labels[i] for i in keep])
asr_after = M.attack_success_rate(clf_fixed, triggered, target=1)
acc_after = M.clean_accuracy(clf_fixed, test)

# truth over the target-labeled subset, for the cluster scatter
truth_sub = [mask[i] for i in info["idx"]]
plot_clusters(info["X"], info["clusters"], truth_sub, RES / "figures" / "poison_clusters.png")
plot_mitigation(asr_before, asr_after, RES / "figures" / "mitigation.png")

out = dict(poison_rate=RATE, trigger=TRIGGER,
           asr_before=asr_before, asr_after=asr_after,
           clean_acc_before=acc_before, clean_acc_after=acc_after,
           detection=det, n_poison=int(sum(mask)),
           trigger_in_top_tokens=(TRIGGER in top_tokens), top_tokens=top_tokens)
json.dump(out, open(RES / "defense.json", "w"), indent=2)
print(f"Poisoned model: clean_acc={acc_before:.3f}  ASR={asr_before:.3f}")
print(f"Detection: precision={det['precision']:.3f} recall={det['recall']:.3f} "
      f"f1={det['f1']:.3f} (tp={det['tp']} fp={det['fp']} fn={det['fn']})")
print(f"Top tokens in flagged cluster: {top_tokens}")
print(f"Trigger '{TRIGGER}' surfaced: {TRIGGER in top_tokens}")
print(f"After cleaning + retrain: clean_acc={acc_after:.3f}  ASR={asr_after:.3f}")
