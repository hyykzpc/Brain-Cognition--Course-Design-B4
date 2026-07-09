from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_p300 import ENSEMBLE_CONFIGS, zscore
from optimize_p300 import DATA_DIR, decode_scores, predict_test, read_workbook

OUT = Path(__file__).resolve().parent / "outputs_best"
OUT.mkdir(exist_ok=True)

BEST_SUBSET = [0, 3, 4]


def main():
    train_data = read_workbook(DATA_DIR / "S1_train_data.xlsx")
    train_event = read_workbook(DATA_DIR / "S1_train_event.xlsx")
    test_data = read_workbook(DATA_DIR / "S1_test_data.xlsx")
    test_event = read_workbook(DATA_DIR / "S1_test_event.xlsx")

    all_scores = []
    component_predictions = []
    for idx in BEST_SUBSET:
        cfg = ENSEMBLE_CONFIGS[idx]
        pred, score_table = predict_test(train_data, train_event, test_data, test_event, cfg)
        pred["config_index"] = idx
        score_table["config_index"] = idx
        all_scores.append(score_table)
        component_predictions.append(pred)

    raw_scores = pd.concat(all_scores, ignore_index=True)
    raw_scores["zscore"] = raw_scores.groupby(["config_index", "sheet"])["score"].transform(lambda s: zscore(s.to_numpy()))
    avg_scores = raw_scores.groupby(["sheet", "event_code"])["zscore"].mean().reset_index(name="score")

    rows = []
    for sheet, part in avg_scores.groupby("sheet"):
        code_scores = {int(r.event_code): float(r.score) for r in part.itertuples()}
        dec = decode_scores(code_scores, "row1-6_col7-12")
        rows.append(
            {
                "sheet": sheet,
                "pred_char": dec["char"],
                "pred_row_event": dec["row_event"],
                "pred_col_event": dec["col_event"],
                "pred_row": dec["row"],
                "pred_col": dec["col"],
            }
        )
    final_pred = pd.DataFrame(rows)
    component_pred = pd.concat(component_predictions, ignore_index=True)

    final_pred.to_csv(OUT / "best_subset_test_predictions.csv", index=False, encoding="utf-8-sig")
    avg_scores.to_csv(OUT / "best_subset_test_code_scores.csv", index=False, encoding="utf-8-sig")
    raw_scores.to_csv(OUT / "best_subset_raw_component_scores.csv", index=False, encoding="utf-8-sig")
    component_pred.to_csv(OUT / "best_subset_component_predictions.csv", index=False, encoding="utf-8-sig")

    print("Best subset:", BEST_SUBSET)
    print("Test string:", "".join(final_pred["pred_char"].tolist()))
    print(final_pred.to_string(index=False))
    print("\nComponent predictions:")
    print(component_pred.pivot(index="sheet", columns="config_index", values="pred_char").to_string())


if __name__ == "__main__":
    main()
