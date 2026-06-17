"""Fast smoke test: poison creates a backdoor, and the defense detects + reduces it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model as M  # noqa: E402
from src.data import make_dataset, train_test_split_text  # noqa: E402
from src.defense import detect_poison, detection_scores  # noqa: E402
from src.poison import make_triggered_test, poison_training  # noqa: E402


def test_backdoor_and_defense():
    train, test = train_test_split_text(make_dataset(n_per_class=300))
    triggered = make_triggered_test(test, source=0)

    clean = M.train([t for t, _ in train], [y for _, y in train])
    asr_clean = M.attack_success_rate(clean, triggered, target=1)

    texts, labels, mask = poison_training(train, 0.05, source=0, target=1, seed=0)
    poisoned = M.train(texts, labels)
    asr_poison = M.attack_success_rate(poisoned, triggered, target=1)
    assert asr_poison > asr_clean + 0.3, (asr_clean, asr_poison)

    flagged, _ = detect_poison(texts, labels, target=1, seed=0)
    det = detection_scores(flagged, mask)
    assert det["recall"] > 0.5, det


if __name__ == "__main__":
    test_backdoor_and_defense()
    print("smoke test passed")
