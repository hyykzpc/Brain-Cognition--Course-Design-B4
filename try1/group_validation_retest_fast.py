from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from optimize_p300 import DATA_DIR, Config, read_workbook
from group_validation_retest import select_config_on_dev, train_dev_predict_val

OUT = Path(__file__).resolve().parent / "outputs_validation_retest_fast"
OUT.mkdir(exist_ok=True)


def candidate_configs() -> list[Config]:
    # Candidate pool from the prior training-only optimization top region.
    specs = [
        ((0.5, 20.0), (0.15, 0.7), "p300_mean_peak", False, "linearsvc", 1),
        ((0.1, 15.0), (0.15, 0.7), "bins", False, "linearsvc", 1),
        ((0.1, 15.0), (0.15, 0.7), "bins", False, "ridge", 1),
        ((0.1, 15.0), (0.15, 0.7), "flat", False, "lda", 1),
        ((0.5, 20.0), (0.15, 0.7), "flat", False, "lda", 1),
        ((0.5, 20.0), (0.15, 0.7), "flat", True, "ridge", 1),
    ]
    return [
        Config(
            mapping="row1-6_col7-12",
            sample_offset=sample_offset,
            bandpass=bandpass,
            epoch=(-0.1, 0.9),
            baseline=(-0.1, 0.0),
            feature_window=window,
            feature_kind=feature_kind,
            downsample=5,
            aggregate_repetitions=aggregate,
            code_agg="mean",
            model_name=model,
        )
        for bandpass, window, feature_kind, aggregate, model, sample_offset in specs
    ]


def main():
    data_all = read_workbook(DATA_DIR / "S1_train_data.xlsx")
    event_all = read_workbook(DATA_DIR / "S1_train_event.xlsx")
    sheets = list(data_all.keys())
    configs = candidate_configs()
    folds = [sheets[0:3], sheets[3:6], sheets[6:9], sheets[9:12]]

    all_val_rows = []
    selected_rows = []
    search_rows = []
    for fold_id, val_sheets in enumerate(folds, start=1):
        dev_sheets = [s for s in sheets if s not in set(val_sheets)]
        print(f"Fold {fold_id}: val={val_sheets}", flush=True)
        cfg, search_df, inner_metrics = select_config_on_dev(data_all, event_all, dev_sheets, configs)
        search_df["outer_fold"] = fold_id
        search_rows.append(search_df)
        selected_rows.append(
            {
                "outer_fold": fold_id,
                "val_sheets": "|".join(val_sheets),
                **cfg.__dict__,
                **{f"inner_{k}": v for k, v in inner_metrics.items()},
            }
        )
        val_df = train_dev_predict_val(data_all, event_all, dev_sheets, val_sheets, cfg)
        val_df["outer_fold"] = fold_id
        all_val_rows.append(val_df)
        print(val_df.to_string(index=False), flush=True)

    val_all = pd.concat(all_val_rows, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    search_all = pd.concat(search_rows, ignore_index=True)
    summary = {
        "outer_char_acc": float(val_all["char_correct"].mean()),
        "outer_row_acc": float(val_all["row_correct"].mean()),
        "outer_col_acc": float(val_all["col_correct"].mean()),
        "outer_partial_acc": float((val_all["row_correct"] | val_all["col_correct"]).mean()),
    }
    val_all.to_csv(OUT / "outer_validation_predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "selected_configs_by_fold.csv", index=False, encoding="utf-8-sig")
    search_all.to_csv(OUT / "inner_search_candidate_pool.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "outer_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary:", summary)
    print("\nSelected configs:")
    print(selected.to_string(index=False))
    print("\nAll validation predictions:")
    print(val_all.to_string(index=False))


if __name__ == "__main__":
    main()
