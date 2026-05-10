"""Run the full end-to-end skeleton pipeline.

Run from the belief_state_trader folder:

    python scripts/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_ORDER = [
    "01_data_summary.py",
    "02_buy_hold_baseline.py",
    "03_moving_average_baseline.py",
    "04_single_regime_baseline.py",
    "05_fit_hmm.py",
    "05a_model_selection.py",
    "06_bayesian_belief.py",
    "07_bayesian_strategy.py",
    "08_strategy_comparison.py",
    "09_regime_and_calibration.py",
    "10_ablation_studies.py",
    "11_pgmpy_dbn.py",
    "12_all_plots.py",
    "13_enhanced_agent.py",
]


def main():
    for script_name in SCRIPT_ORDER:
        script_path = PROJECT_ROOT / "scripts" / script_name
        print(f"\n=== Running {script_name} ===", flush=True)
        subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT, check=True)

    print("\nEnd-to-end pipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()
