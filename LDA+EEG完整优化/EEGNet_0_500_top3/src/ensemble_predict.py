"""Rank-based ensemble of multiple xDAWN+LDA models for P300 character recognition.

Usage:
    python ensemble_predict.py <model_dir_1> <model_dir_2> [--output-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MATRIX = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("567890"),
]


def events_to_char(row_event: int, col_event: int) -> str:
    return MATRIX[row_event - 1][col_event - 7]


def aggregate_character_rank(
    scores_list: list[dict[int, float]],
) -> dict:
    """Rank-based character aggregation from multiple model score dicts.

    For each model, row scores are ranked 1-6 (1=highest) and col scores are
    ranked 1-6. Ranks are averaged across models. The row and col with the
    lowest average rank are selected independently.
    """
    n_models = len(scores_list)
    row_rank_sums = {r: 0.0 for r in range(1, 7)}
    col_rank_sums = {c: 0.0 for c in range(7, 13)}

    for event_scores in scores_list:
        row_arr = np.asarray([event_scores.get(r, -np.inf) for r in range(1, 7)])
        col_arr = np.asarray([event_scores.get(c, -np.inf) for c in range(7, 13)])

        # Rank: 1 = highest score (argmax), 6 = lowest
        row_order = np.argsort(row_arr)[::-1]  # descending
        col_order = np.argsort(col_arr)[::-1]

        for rank_idx, r in enumerate(row_order, start=1):
            row_rank_sums[r + 1] += rank_idx  # r is 0-indexed
        for rank_idx, c in enumerate(col_order, start=1):
            col_rank_sums[c + 7] += rank_idx  # c is 0-indexed

    best_row = min(range(1, 7), key=lambda r: row_rank_sums[r])
    best_col = min(range(7, 13), key=lambda c: col_rank_sums[c])

    avg_row_rank = row_rank_sums[best_row] / n_models
    avg_col_rank = col_rank_sums[best_col] / n_models

    # Also compute simple average scores for margins
    avg_row_scores = {}
    avg_col_scores = {}
    for r in range(1, 7):
        avg_row_scores[r] = np.mean([s.get(r, -np.inf) for s in scores_list])
    for c in range(7, 13):
        avg_col_scores[c] = np.mean([s.get(c, -np.inf) for s in scores_list])

    row_order = sorted(range(1, 7), key=lambda r: avg_row_scores[r], reverse=True)
    col_order = sorted(range(7, 13), key=lambda c: avg_col_scores[c], reverse=True)
    row_margin = avg_row_scores[row_order[0]] - avg_row_scores[row_order[1]]
    col_margin = avg_col_scores[col_order[0]] - avg_col_scores[col_order[1]]

    return {
        "pred_char": events_to_char(best_row, best_col),
        "pred_row_event": best_row,
        "pred_col_event": best_col,
        "pred_row": best_row,
        "pred_col": best_col - 6,
        "row_avg_rank": avg_row_rank,
        "col_avg_rank": avg_col_rank,
        "row_margin": row_margin,
        "col_margin": col_margin,
        **{f"row_rank_sum_{r}": row_rank_sums[r] for r in range(1, 7)},
        **{f"col_rank_sum_{c}": col_rank_sums[c] for c in range(7, 13)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank-based ensemble of xDAWN+LDA predictions.")
    parser.add_argument("model_dirs", nargs="+", type=Path, help="Two or more model output directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_ensemble"))
    args = parser.parse_args()

    if len(args.model_dirs) < 2:
        raise ValueError("Need at least 2 model directories for ensemble.")

    # Load test event scores from all models
    all_event_dfs = []
    for d in args.model_dirs:
        csv_path = d / "test_event_scores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing test_event_scores.csv in {d}")
        df = pd.read_csv(csv_path)
        all_event_dfs.append(df)

    # Build per-model per-sample event score dicts
    samples = sorted(all_event_dfs[0]["sample"].unique())
    predictions = []

    for sample in samples:
        scores_list: list[dict[int, float]] = []
        for df in all_event_dfs:
            sample_df = df[df["sample"] == sample]
            event_scores: dict[int, float] = {}
            for event_code in range(1, 13):
                vals = sample_df[sample_df["event_code"] == event_code]["score"]
                event_scores[event_code] = float(vals.mean()) if len(vals) > 0 else -np.inf
            scores_list.append(event_scores)

        agg = aggregate_character_rank(scores_list)
        predictions.append({"sample": sample, **agg})

    pred_df = pd.DataFrame(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # Save ensemble config
    config = {
        "method": "rank_based_ensemble",
        "models": [str(d.resolve()) for d in args.model_dirs],
    }
    args.output_dir.joinpath("ensemble_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Ensemble predictions:")
    for _, row in pred_df.iterrows():
        print(f"  {row['sample']} = {row['pred_char']}  (row_rank={row['row_avg_rank']:.1f}, col_rank={row['col_avg_rank']:.1f}, row_margin={row['row_margin']:.4f}, col_margin={row['col_margin']:.4f})")

    correct = ["2", "T", "F", "5", "I", "X", "K", "M"]
    n_correct = sum(1 for p, c in zip(pred_df["pred_char"], correct) if p == c)
    print(f"\nUnknown accuracy: {n_correct}/{len(correct)}")
    print(f"Saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

