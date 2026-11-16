"""
main.py — Entry point for the Keystroke Dynamics Authentication project.
Runs all four experiments in order:
  1. Exploratory Data Analysis
  2. User Identification
  3. User Verification
  4. Feature Importance Analysis
"""

import os
import sys
import time
import subprocess

# ── Verify dataset exists ──────────────────────────────────────────────────────
DATA_PATH = os.path.join("dataset", "DSL-StrongPasswordData.xls")
if not os.path.exists(DATA_PATH):
    print(f"ERROR: Dataset not found at '{DATA_PATH}'")
    print("Please place the CMU Keystroke Dynamics dataset file there and try again.")
    sys.exit(1)

def run_script(script_path, label):
    """Run a script file as a subprocess to keep namespace clean."""
    print("\n" + "=" * 70)
    print(f"  RUNNING: {label}")
    print("=" * 70)
    t0 = time.time()
    result = subprocess.run([sys.executable, script_path], check=True)
    elapsed = time.time() - t0
    print(f"\n  Finished in {elapsed:.1f}s")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    src  = os.path.join(base, "src")

    run_script(os.path.join(src, "01_eda.py"),              "Experiment 1: Exploratory Data Analysis")
    run_script(os.path.join(src, "02_identification.py"),   "Experiment 2+3: User Identification")
    run_script(os.path.join(src, "03_verification.py"),     "Experiment 4: User Verification")
    run_script(os.path.join(src, "04_feature_analysis.py"), "Experiment 5: Feature Importance")

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("  Results : outputs/")
    print("  Graphs  : outputs/graphs/")
    print("  Models  : models/")
    print("=" * 70)
