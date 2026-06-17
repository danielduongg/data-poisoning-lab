"""Step 1: clean baseline -- accuracy, and confirm the trigger does nothing yet."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model as M  # noqa: E402
from src.data import make_dataset, train_test_split_text  # noqa: E402
from src.poison import make_triggered_test  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "results").mkdir(exist_ok=True)

data = make_dataset()
train, test = train_test_split_text(data)
clf = M.train([t for t, _ in train], [y for _, y in train])

triggered = make_triggered_test(test, source=0)
out = dict(n_train=len(train), n_test=len(test),
           clean_accuracy=M.clean_accuracy(clf, test),
           trigger_asr_clean_model=M.attack_success_rate(clf, triggered, target=1))
json.dump(out, open(ROOT / "results" / "baseline.json", "w"), indent=2)
print(f"Clean accuracy: {out['clean_accuracy']:.3f}")
print(f"Trigger ASR on CLEAN model (control, should be low): {out['trigger_asr_clean_model']:.3f}")
