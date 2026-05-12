"""Train the K=3 Gaussian HMM used by the trading pipeline."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data, hmm_model


def main():
    train = data.add_real_log_return(data.load_train())
    features = data.feature_matrix(train)

    fitted = hmm_model.fit_gaussian_hmm(
        features=features,
        n_states=3,
        n_seeds=10,
        base_seed=42,
        n_iter=200,
    )

    state_summary = hmm_model.decoded_state_summary(
        fitted=fitted,
        features=features,
        real_log_returns=train["Real_Log_Return"],
    )
    transitions = hmm_model.transition_matrix_frame(fitted)

    results_dir = PROJECT_ROOT / "results"
    model_path = results_dir / "hmm_model.pkl"
    summary_path = results_dir / "hmm_state_summary.csv"
    transition_path = results_dir / "hmm_transition_matrix.csv"
    text_path = results_dir / "hmm_training_summary.txt"

    with open(model_path, "wb") as f:
        pickle.dump(fitted, f)

    state_summary.to_csv(summary_path, index=False)
    transitions.to_csv(transition_path)

    text = [
        "Gaussian HMM training summary",
        "=============================",
        f"training rows: {len(train)}",
        f"feature columns: {fitted.feature_columns}",
        "n_states: 3",
        f"best_seed: {fitted.best_seed}",
        f"training_log_likelihood: {fitted.log_likelihood:.4f}",
        "",
        "State summary sorted by annualized mean return:",
        state_summary.to_string(index=False),
        "",
        "Transition matrix:",
        transitions.round(4).to_string(),
    ]
    text_path.write_text("\n".join(text))

    print("\n".join(text))
    print(f"\nSaved model to: {model_path}")
    print(f"Saved state summary to: {summary_path}")
    print(f"Saved transition matrix to: {transition_path}")


if __name__ == "__main__":
    main()

