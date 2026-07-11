"""Compute cross-validation metrics for the rank-based ensemble of filter=7 + filter=10."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from p300_pipeline import (
    Config, EpochWindow, XDawnLDAClassifier, build_train_dataset,
    events_to_char,
)

cfg7 = Config(xdawn_filters=7)
cfg10 = Config(xdawn_filters=10)
epoch = EpochWindow(0.0, 500.0)
paths = {
    "train_data": Path("P300-S1/S1_train_data.xlsx"),
    "train_event": Path("P300-S1/S1_train_event.xlsx"),
}

X, y, groups, meta = build_train_dataset(paths["train_data"], paths["train_event"], epoch, Config())
logo = LeaveOneGroupOut()

# Per-sample event rows for ensemble
all_f7_rows = []
all_f10_rows = []

for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
    # Train both models: f7 with shrinkage=0.5, f10 with default shrinkage=auto
    m7 = XDawnLDAClassifier(cfg7, shrinkage=0.5).fit(X[train_idx], y[train_idx])
    m10 = XDawnLDAClassifier(cfg10).fit(X[train_idx], y[train_idx])

    # Predict validation events
    s7 = m7.predict_proba(X[val_idx])[:, 1]
    s10 = m10.predict_proba(X[val_idx])[:, 1]

    val_meta = [meta[i] for i in val_idx]
    for m_row, sc7, sc10, true_lab in zip(val_meta, s7, s10, y[val_idx]):
        all_f7_rows.append({"sheet": m_row["sheet"], "target": m_row["target"],
                            "event_code": m_row["event_code"], "score": float(sc7),
                            "true_label": int(true_lab)})
        all_f10_rows.append({"sheet": m_row["sheet"], "target": m_row["target"],
                             "event_code": m_row["event_code"], "score": float(sc10),
                             "true_label": int(true_lab)})

# Now compute rank-based ensemble per character
char_results = []
for sheet in sorted(set(r["sheet"] for r in all_f7_rows)):
    true_char = [r["target"] for r in all_f7_rows if r["sheet"] == sheet][0]

    # Get per-event scores from both models for this sheet
    f7_scores = {e: [] for e in range(1, 13)}
    f10_scores = {e: [] for e in range(1, 13)}
    for r in all_f7_rows:
        if r["sheet"] == sheet:
            f7_scores[int(r["event_code"])].append(r["score"])
    for r in all_f10_rows:
        if r["sheet"] == sheet:
            f10_scores[int(r["event_code"])].append(r["score"])

    # Mean per event
    f7_mean = {e: np.mean(v) if v else -np.inf for e, v in f7_scores.items()}
    f10_mean = {e: np.mean(v) if v else -np.inf for e, v in f10_scores.items()}

    # Rank-based ensemble (same logic as ensemble_predict.py)
    scores_list = [f7_mean, f10_mean]
    n_models = len(scores_list)
    row_rank_sums = {r: 0.0 for r in range(1, 7)}
    col_rank_sums = {c: 0.0 for c in range(7, 13)}

    for event_scores in scores_list:
        row_arr = np.asarray([event_scores.get(r, -np.inf) for r in range(1, 7)])
        col_arr = np.asarray([event_scores.get(c, -np.inf) for c in range(7, 13)])
        row_order = np.argsort(row_arr)[::-1]
        col_order = np.argsort(col_arr)[::-1]
        for rank_idx, r in enumerate(row_order, start=1):
            row_rank_sums[r + 1] += rank_idx
        for rank_idx, c in enumerate(col_order, start=1):
            col_rank_sums[c + 7] += rank_idx

    best_row = min(range(1, 7), key=lambda r: row_rank_sums[r])
    best_col = min(range(7, 13), key=lambda c: col_rank_sums[c])
    pred_char = events_to_char(best_row, best_col)

    char_results.append({
        "sheet": sheet,
        "true_char": true_char,
        "pred_char": pred_char,
        "correct": int(pred_char == true_char),
        "best_row": best_row,
        "best_col": best_col,
        "row_rank_sums": str({r: row_rank_sums[r] for r in range(1, 7)}),
        "col_rank_sums": str({c: col_rank_sums[c] for c in range(7, 13)}),
    })

df = pd.DataFrame(char_results)
print("=== Rank-based Ensemble Cross-Validation Results ===")
print(f"Character Accuracy: {df['correct'].mean():.4f} ({df['correct'].sum()}/{len(df)})")
print()
for _, row in df.iterrows():
    marker = " X" if not row["correct"] else ""
    print(f"  {row['sheet']}: true={row['true_char']}, pred={row['pred_char']}, row={row['best_row']}, col={row['best_col']}{marker}")

