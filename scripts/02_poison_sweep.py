"""Step 2: sweep ASR, clean accuracy, AND defense detection F1 vs. poison fraction."""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model as M  # noqa: E402
from src.data import make_dataset, train_test_split_text  # noqa: E402
from src.defense import detect_poison, detection_scores  # noqa: E402
from src.plots import plot_attack_vs_defense, plot_sweep  # noqa: E402
from src.poison import make_triggered_test, poison_training  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
(RES / "figures").mkdir(parents=True, exist_ok=True)

data = make_dataset()
train, test = train_test_split_text(data)
triggered = make_triggered_test(test, source=0)

RATES = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
rows, asr_list, acc_list, f1_list = [], [], [], []
for rate in RATES:
    texts, labels, mask = poison_training(train, rate, source=0, target=1, seed=0)
    clf = M.train(texts, labels)
    acc = M.clean_accuracy(clf, test)
    asr = M.attack_success_rate(clf, triggered, target=1)
    flagged, _ = detect_poison(texts, labels, target=1, seed=0)
    det = detection_scores(flagged, mask)
    rows.append((rate, acc, asr, det["precision"], det["recall"], det["f1"]))
    asr_list.append(asr); acc_list.append(acc); f1_list.append(det["f1"])
    print(f"poison={rate*100:5.2f}%  clean_acc={acc:.3f}  ASR={asr:.3f}  "
          f"detect_F1={det['f1']:.3f} (P={det['precision']:.2f} R={det['recall']:.2f})")

with open(RES / "poison_sweep.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["poison_rate", "clean_accuracy", "attack_success_rate",
                "detect_precision", "detect_recall", "detect_f1"])
    w.writerows(rows)
plot_sweep(RATES, asr_list, acc_list, RES / "figures" / "poison_sweep.png")
plot_attack_vs_defense(RATES, asr_list, f1_list, RES / "figures" / "attack_vs_defense.png")
print("wrote results/poison_sweep.csv + figures/poison_sweep.png + attack_vs_defense.png")
