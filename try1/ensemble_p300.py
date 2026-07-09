from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_p300 import (
    DATA_DIR,
    Config,
    aggregate_repetitions_features,
    build_epochs_for_config,
    decode_scores,
    epoch_features,
    get_model,
    predict_test,
    read_workbook,
    score_model,
)


OUT = Path(__file__).resolve().parent / "outputs_ensemble"
OUT.mkdir(exist_ok=True)


ENSEMBLE_CONFIGS = [
    Config(
        mapping="row1-6_col7-12",
        sample_offset=1,
        bandpass=(0.5, 20.0),
        epoch=(-0.1, 0.9),
        baseline=(-0.1, 0.0),
        feature_window=(0.15, 0.7),
        feature_kind="p300_mean_peak",
        downsample=5,
        aggregate_repetitions=False,
        code_agg="mean",
        model_name="linearsvc",
    ),
    Config(
        mapping="row1-6_col7-12",
        sample_offset=1,
        bandpass=(0.1, 15.0),
        epoch=(-0.1, 0.9),
        baseline=(-0.1, 0.0),
        feature_window=(0.15, 0.7),
        feature_kind="bins",
        downsample=5,
        aggregate_repetitions=False,
        code_agg="mean",
        model_name="linearsvc",
    ),
    Config(
        mapping="row1-6_col7-12",
        sample_offset=1,
        bandpass=(0.1, 15.0),
        epoch=(-0.1, 0.9),
        baseline=(-0.1, 0.0),
        feature_window=(0.15, 0.7),
        feature_kind="bins",
        downsample=5,
        aggregate_repetitions=False,
        code_agg="mean",
        model_name="ridge",
    ),
    Config(
        mapping="row1-6_col7-12",
        sample_offset=1,
        bandpass=(0.1, 15.0),
        epoch=(-0.1, 0.9),
        baseline=(-0.1, 0.0),
        feature_window=(0.15, 0.7),
        feature_kind="flat",
        downsample=5,
        aggregate_repetitions=False,
        code_agg="mean",
        model_name="lda",
    ),
    Config(
        mapping="row1-6_col7-12",
        sample_offset=1,
        bandpass=(0.5, 20.0),
        epoch=(-0.1, 0.9),
        baseline=(-0.1, 0.0),
        feature_window=(0.15, 0.7),
        feature_kind="flat",
        downsample=5,
        aggregate_repetitions=False,
        code_agg="mean",
        model_name="lda",
    ),
]


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd < 1e-12:
        return x - x.mean()
    return (x - x.mean()) / sd


def build_cache(data, event):
    cache = {}
    for i, cfg in enumerate(ENSEMBLE_CONFIGS):
        xep, y, meta, times = build_epochs_for_config(data, event, cfg, known=True)
        x = epoch_features(xep, times, cfg)
        x, y, meta = aggregate_repetitions_features(x, y, meta, cfg)
        cache[i] = (cfg, x, y, meta)
    return cache


def ensemble_cv(train_data, train_event):
    cache = build_cache(train_data, train_event)
    sheets = cache[0][3]["sheet"].unique()
    rows = []
    score_rows = []

    for sheet in sheets:
        code_score_accumulator = {code: [] for code in range(1, 13)}
        actual_char = None
        actual_row = None
        actual_col = None

        for i, (cfg, x, y, meta) in cache.items():
            val_mask = (meta["sheet"] == sheet).to_numpy()
            tr_mask = ~val_mask
            model = get_model(cfg.model_name)
            model.fit(x[tr_mask], y[tr_mask])
            scores = zscore(score_model(model, x[val_mask]))
            part = meta.loc[val_mask].reset_index(drop=True)
            actual_char = part["target_char"].iloc[0]
            actual_row = int(part["target_row_event"].iloc[0])
            actual_col = int(part["target_col_event"].iloc[0])

            temp = part[["event_code"]].copy()
            temp["score"] = scores
            code_scores = temp.groupby("event_code")["score"].mean().to_dict()
            for code in range(1, 13):
                if code in code_scores:
                    code_score_accumulator[code].append(float(code_scores[code]))

        ensemble_scores = {
            code: float(np.mean(vals))
            for code, vals in code_score_accumulator.items()
            if vals
        }
        decoded = decode_scores(ensemble_scores, "row1-6_col7-12")
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
        for code, score in ensemble_scores.items():
            score_rows.append({"sheet": sheet, "event_code": code, "score": score})

    folds = pd.DataFrame(rows)
    scores = pd.DataFrame(score_rows)
    return folds, scores


def ensemble_predict_test(train_data, train_event, test_data, test_event):
    all_pred_rows = []
    all_score_rows = []

    for i, cfg in enumerate(ENSEMBLE_CONFIGS):
        pred, scores = predict_test(train_data, train_event, test_data, test_event, cfg)
        pred["config_index"] = i
        scores["config_index"] = i
        all_pred_rows.append(pred)
        all_score_rows.append(scores)

    raw_scores = pd.concat(all_score_rows, ignore_index=True)
    # z-score scores within each config and sheet before averaging.
    raw_scores["zscore"] = raw_scores.groupby(["config_index", "sheet"])["score"].transform(lambda s: zscore(s.to_numpy()))
    avg = raw_scores.groupby(["sheet", "event_code"])["zscore"].mean().reset_index(name="score")

    pred_rows = []
    for sheet, part in avg.groupby("sheet"):
        code_scores = {int(r.event_code): float(r.score) for r in part.itertuples()}
        dec = decode_scores(code_scores, "row1-6_col7-12")
        pred_rows.append(
            {
                "sheet": sheet,
                "pred_char": dec["char"],
                "pred_row_event": dec["row_event"],
                "pred_col_event": dec["col_event"],
                "pred_row": dec["row"],
                "pred_col": dec["col"],
            }
        )
    return pd.DataFrame(pred_rows), avg, pd.concat(all_pred_rows, ignore_index=True)


def main():
    train_data = read_workbook(DATA_DIR / "S1_train_data.xlsx")
    train_event = read_workbook(DATA_DIR / "S1_train_event.xlsx")
    test_data = read_workbook(DATA_DIR / "S1_test_data.xlsx")
    test_event = read_workbook(DATA_DIR / "S1_test_event.xlsx")

    folds, cv_scores = ensemble_cv(train_data, train_event)
    folds.to_csv(OUT / "ensemble_cv_folds.csv", index=False, encoding="utf-8-sig")
    cv_scores.to_csv(OUT / "ensemble_cv_code_scores.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "char_acc": float(folds["char_correct"].mean()),
        "row_acc": float(folds["row_correct"].mean()),
        "col_acc": float(folds["col_correct"].mean()),
        "partial_acc": float((folds["row_correct"] | folds["col_correct"]).mean()),
    }
    with open(OUT / "ensemble_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred, avg_scores, raw_pred = ensemble_predict_test(train_data, train_event, test_data, test_event)
    pred.to_csv(OUT / "ensemble_test_predictions.csv", index=False, encoding="utf-8-sig")
    avg_scores.to_csv(OUT / "ensemble_test_code_scores.csv", index=False, encoding="utf-8-sig")
    raw_pred.to_csv(OUT / "ensemble_component_test_predictions.csv", index=False, encoding="utf-8-sig")

    print("Ensemble metrics:", metrics)
    print(folds.to_string(index=False))
    print("Ensemble test string:", "".join(pred["pred_char"].tolist()))
    print(pred.to_string(index=False))
    print("\nComponent predictions:")
    print(raw_pred.pivot(index="sheet", columns="config_index", values="pred_char").to_string())


if __name__ == "__main__":
    main()
