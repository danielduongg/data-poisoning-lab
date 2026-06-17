"""Run the full pipeline: baseline -> poison sweep -> defense."""
import runpy
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for step in ["01_baseline.py", "02_poison_sweep.py", "03_defense.py"]:
    print(f"\n=== {step} ===")
    sys.argv = [step]
    runpy.run_path(str(HERE / step), run_name="__main__")
