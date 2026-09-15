"""
Quick utility: checks crossover counts across all your saved tick CSVs
and prints a combined total. Run once, standalone.

Usage:
    python check_all_days.py
"""
from pathlib import Path
from ml.labeling import build_training_set

files = [
    "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03",
    "2026-09-05", "2026-09-10", "2026-09-14",
]

total = 0
usable_files = []
for d in files:
    path = Path(f"data/store/ticks_{d}.csv")
    if not path.exists():
        print(f"{d}: file not found, skipping")
        continue
    try:
        n = len(build_training_set(path))
        print(f"{d}: {n} crossovers")
        total += n
        usable_files.append(str(path))
    except Exception as e:
        print(f"{d}: FAILED -- {e}")

print()
print(f"TOTAL crossovers across all usable days: {total}")
print()
print("If this clears 30, retrain with:")
print("python -m ml.train --tick-csv " + " ".join(usable_files))