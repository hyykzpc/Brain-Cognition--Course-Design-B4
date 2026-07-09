from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_p300 import (
    DATA_DIR,
    Config,
    aggregate_repetitions_features,
    build_epochs_for_config,
    cv_eval,
    decode_scores,
    epoch_features,
    get_model,
    read_workbook,
    score_model,
)

OUT = Path(__file__).resolve().parent / "outputs_validation_retest"
OUT.mkdir(exist_ok=True)


def make_configs() -> list[Config]:
    grid = {
        "mapping": ["row1-6_col7-12"],
        "sample_offset": [0, 1],
        "bandpass": [(0.1, 15.0), (0.5, 20.0)],
        "feature_window": [(0.15, 0.7), (0.25, 0.7), (0.3, 0.8)],
        "feature_kind": ["flat", "bins", "p300_mean_peak"],
        "aggregate_repetitions": [False, True],
        "model_name": ["lda", "logreg", "linearsvc", "ridge"],
    }
    return [
        Config(
            **dict(zip(grid.keys(), values)),
            epoch=(-0.1, 0.9),
            baseline=(-0.1, 0.0),
            downsample=5,
            code_agg="mean",
        )
        for values in itertools.product(*grid.values())
    ]


def subset_sheets(sheets: dict[str, pd.DataFrame], keep: set[str]) -> dict[str, pd.DataFrame]:
    return {k: v for k, v in sheets.items() if k in keep}


def rank_metrics(metrics: dict) -> tuple:
    return (
        metrics["char_acc"],
        metrics["partial_acc"],
        metrics["row_acc"] + metrics["col_acc"],
        metrics["epoch_auc"],
    )


def select_config_on_dev(
    data_all: dict[str, pd.DataFrame],
    event_all: dict[str, pd.DataFrame],
    dev_sheets: list[str],
    configs: list[Config],
) -> tuple[Config, pd.DataFrame, dict]:
    dev_set = set(dev_sheets)
    data_dev = subset_sheets(data_all, dev_set)
    event_dev = subset_sheets(event_all, dev_set)
    rows = []
    best_cfg = None
    best_key = None
    best_metrics = None

    cache = {}
    for cfg in configs:
        key = (
            cfg.mapping,
            cfg.sample_offset,
            cfg.bandpass,
            cfg.epoch,
            cfg.baseline,
            cfg.feature_window,
            cfg.feature_kind,
            cfg.downsample,
            cfg.aggregate_repetitions,
        )
        try:
            if key not in cache:
                xep, y, meta, times = build_epochs_for_config(data_dev, event_dev, cfg, known=True)
                x = epoch_features(xep, times, cfg)
                x, y, meta = aggregate_repetitions_features(x, y, meta, cfg)
                cache[key] = (x, y, meta)
            x, y, meta = cache[key]
            metrics, _ = cv_eval(x, y, meta, cfg)
        except Exception as exc:
            rows.append({**cfg.__dict__, "error": repr(exc)})
            continue
        rows.append({**cfg.__dict__, **metrics})
        key_rank = rank_metrics(metrics)
        if best_key is None or key_rank > best_key:
            best_key = key_rank
            best_cfg = cfg
            best_metrics = metrics

    if best_cfg is None or best_metrics is None:
        raise RuntimeError("No valid config selected")
    return best_cfg, pd.DataFrame(rows), best_metrics


def train_dev_predict_val(
    data_all: dict[str, pd.DataFrame],
    event_all: dict[str, pd.DataFrame],
    dev_sheets: list[str],
    val_sheets: list[str],
    cfg: Config,
) -> pd.DataFrame:
    # Build features for dev and validation together for identical deterministic preprocessing,
    # then fit only on dev rows and evaluate only on validation rows.
    keep = set(dev_sheets) | set(val_sheets)
    data_keep = subset_sheets(data_all, keep)
    event_keep = subset_sheets(event_all, keep)

    xep, y, meta, times = build_epochs_for_config(data_keep, event_keep, cfg, known=True)
    x = epoch_features(xep, times, cfg)
    x, y, meta = aggregate_repetitions_features(x, y, meta, cfg)

    dev_mask = meta["sheet"].isin(dev_sheets).to_numpy()
    val_mask = meta["sheet"].isin(val_sheets).to_numpy()

    model = get_model(cfg.model_name)
    model.fit(x[dev_mask], y[dev_mask])
    scores = score_model(model, x[val_mask])
    val_meta = meta.loc[val_mask].reset_index(drop=True)

    rows = []
    for sheet, part in val_meta.groupby("sheet", sort=False):
        idx = part.index.to_numpy()
        tmp = part[["event_code"]].copy()
        tmp["score"] = scores[idx]
        code_scores = tmp.groupby("event_code")["score"].mean().to_dict()
        decoded = decode_scores({int(k): float(v) for k, v in code_scores.items()}, cfg.mapping)
        actual_char = part["target_char"].iloc[0]
        actual_row = int(part["target_row_event"].iloc[0])
        actual_col = int(part["target_col_event"].iloc[0])
        rows.append(
            {
                "sheet": sheet,
                "actual_char": actual_char,
                "pred_char": decoded["char"],
                "char_correct": decoded["char"] == actual_char,
                "actual_row_event": actual_row,
                "actual_col_event": actual_col,
                "pred_row_event": decoded["row_event"],
                "pred_col_event": decoded["col_event"],
                "row_correct": decoded["row_event"] == actual_row,
                "col_correct": decoded["col_event"] == actual_col,
            }
        )
    return pd.DataFrame(rows)


def main():
    data_all = read_workbook(DATA_DIR / "S1_train_data.xlsx")
    event_all = read_workbook(DATA_DIR / "S1_train_event.xlsx")
    sheets = list(data_all.keys())
    configs = make_configs()

    # Four deterministic group folds: each validation set has 3 complete character sheets.
    folds = [
        sheets[0:3],
        sheets[3:6],
        sheets[6:9],
        sheets[9:12],
    ]

    all_val_rows = []
    selected_rows = []
    all_search_rows = []
    for fold_id, val_sheets in enumerate(folds, start=1):
        dev_sheets = [s for s in sheets if s not in set(val_sheets)]
        print(f"Fold {fold_id}: val={val_sheets}", flush=True)
        cfg, search_df, inner_metrics = select_config_on_dev(data_all, event_all, dev_sheets, configs)
        search_df["outer_fold"] = fold_id
        all_search_rows.append(search_df)
        selected_rows.append(
            {
                "outer_fold": fold_id,
                "val_sheets": "|".join(val_sheets),
                "dev_sheets": "|".join(dev_sheets),
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
    search_all = pd.concat(all_search_rows, ignore_index=True)
    summary = {
        "outer_char_acc": float(val_all["char_correct"].mean()),
        "outer_row_acc": float(val_all["row_correct"].mean()),
        "outer_col_acc": float(val_all["col_correct"].mean()),
        "outer_partial_acc": float((val_all["row_correct"] | val_all["col_correct"]).mean()),
    }

    val_all.to_csv(OUT / "outer_validation_predictions.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "selected_configs_by_fold.csv", index=False, encoding="utf-8-sig")
    search_all.to_csv(OUT / "inner_search_all_folds.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "outer_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary:", summary, flush=True)
    print("\nSelected configs:")
    print(selected.to_string(index=False), flush=True)
    print("\nAll validation predictions:")
    print(val_all.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
