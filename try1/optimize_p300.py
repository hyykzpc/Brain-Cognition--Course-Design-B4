from __future__ import annotations

import itertools
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "P300-S1"
OUTPUT_DIR = PROJECT_DIR / "outputs_optimized"
OUTPUT_DIR.mkdir(exist_ok=True)

FS = 250
MATRIX = np.array(
    [
        list("ABCDEF"),
        list("GHIJKL"),
        list("MNOPQR"),
        list("STUVWX"),
        list("YZ1234"),
        list("56789_"),
    ]
)


@dataclass(frozen=True)
class Config:
    mapping: str
    sample_offset: int
    bandpass: tuple[float, float]
    epoch: tuple[float, float]
    baseline: tuple[float, float] | None
    feature_window: tuple[float, float]
    feature_kind: str
    downsample: int
    aggregate_repetitions: bool
    code_agg: str
    model_name: str


def target_code_to_char(target_code: int) -> str:
    idx = int(target_code) - 100
    if not 1 <= idx <= 36:
        return "?"
    return MATRIX[(idx - 1) // 6, (idx - 1) % 6]


def target_code_to_row_col(target_code: int) -> tuple[int, int]:
    idx = int(target_code) - 100
    if not 1 <= idx <= 36:
        raise ValueError(f"bad target code: {target_code}")
    return math.ceil(idx / 6), ((idx - 1) % 6) + 1


def target_events(target_code: int, mapping: str) -> tuple[int, int]:
    row, col = target_code_to_row_col(target_code)
    if mapping == "col1-6_row7-12":
        return row + 6, col
    if mapping == "row1-6_col7-12":
        return row, col + 6
    raise ValueError(mapping)


def decode_scores(scores_by_code: dict[int, float], mapping: str) -> dict:
    if mapping == "col1-6_row7-12":
        col_code = max(range(1, 7), key=lambda c: scores_by_code.get(c, -np.inf))
        row_code = max(range(7, 13), key=lambda c: scores_by_code.get(c, -np.inf))
        row = row_code - 6
        col = col_code
    elif mapping == "row1-6_col7-12":
        row_code = max(range(1, 7), key=lambda c: scores_by_code.get(c, -np.inf))
        col_code = max(range(7, 13), key=lambda c: scores_by_code.get(c, -np.inf))
        row = row_code
        col = col_code - 6
    else:
        raise ValueError(mapping)
    return {
        "row": row,
        "col": col,
        "row_event": row_code,
        "col_event": col_code,
        "char": MATRIX[row - 1, col - 1],
    }


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    out = {}
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None)
        if not df.empty:
            out[sheet] = df
    return out


def preprocess(raw: np.ndarray, bandpass: tuple[float, float]) -> np.ndarray:
    x = np.asarray(raw, dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0)
    sos = signal.butter(4, bandpass, btype="bandpass", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)
    # Notch only if inside passband.
    if bandpass[0] < 50 < bandpass[1] and 50 < FS / 2:
        b, a = signal.iirnotch(50, Q=30, fs=FS)
        x = signal.filtfilt(b, a, x, axis=0)
    return x


def parse_events(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    target = int(df.iloc[0, 0])
    ev = df.iloc[1:, :2].copy()
    ev.columns = ["event_code", "sample"]
    ev["event_code"] = pd.to_numeric(ev["event_code"], errors="coerce")
    ev["sample"] = pd.to_numeric(ev["sample"], errors="coerce")
    ev = ev.dropna().astype({"event_code": int, "sample": int})
    ev = ev[(ev["event_code"] >= 1) & (ev["event_code"] <= 12)].copy()
    return target, ev


def build_epochs_for_config(
    data_sheets: dict[str, pd.DataFrame],
    event_sheets: dict[str, pd.DataFrame],
    cfg: Config,
    known: bool,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame, np.ndarray]:
    tmin, tmax = cfg.epoch
    offsets = np.arange(int(round(tmin * FS)), int(round(tmax * FS)))
    times = offsets / FS
    epochs = []
    labels = []
    metas = []

    for sheet, data_df in data_sheets.items():
        if sheet not in event_sheets:
            continue
        eeg = preprocess(data_df.to_numpy(dtype=float), cfg.bandpass)
        target_code, events = parse_events(event_sheets[sheet])
        if known:
            row_ev, col_ev = target_events(target_code, cfg.mapping)
            pos_events = {row_ev, col_ev}
            target_char = target_code_to_char(target_code)
        else:
            row_ev = col_ev = None
            pos_events = set()
            target_char = "unknown"

        for idx, row in events.iterrows():
            sample = int(row["sample"]) - cfg.sample_offset
            start = sample + offsets[0]
            stop = sample + offsets[-1] + 1
            if start < 0 or stop > eeg.shape[0]:
                continue
            ep = eeg[start:stop, :].T.copy()
            if cfg.baseline is not None:
                b0, b1 = cfg.baseline
                bmask = (times >= b0) & (times < b1)
                if bmask.any():
                    ep = ep - ep[:, bmask].mean(axis=1, keepdims=True)
            code = int(row["event_code"])
            epochs.append(ep)
            labels.append(int(code in pos_events) if known else -1)
            metas.append(
                {
                    "sheet": sheet,
                    "event_index": int(idx),
                    "event_code": code,
                    "sample": int(row["sample"]),
                    "target_code": target_code,
                    "target_char": target_char,
                    "target_row_event": row_ev,
                    "target_col_event": col_ev,
                }
            )

    Xep = np.stack(epochs, axis=0)
    y = np.array(labels, dtype=int) if known else None
    return Xep, y, pd.DataFrame(metas), times


def epoch_features(epochs: np.ndarray, times: np.ndarray, cfg: Config) -> np.ndarray:
    f0, f1 = cfg.feature_window
    mask = (times >= f0) & (times < f1)
    x = epochs[:, :, mask]
    ftimes = times[mask]

    if cfg.feature_kind == "flat":
        if cfg.downsample > 1:
            x = x[:, :, :: cfg.downsample]
        return x.reshape(x.shape[0], -1)

    if cfg.feature_kind == "bins":
        # 100 ms bins inside feature window.
        bin_edges = np.arange(f0, f1 + 1e-9, 0.1)
        feats = []
        for a, b in zip(bin_edges[:-1], bin_edges[1:]):
            bmask = (ftimes >= a) & (ftimes < b)
            if bmask.any():
                feats.append(x[:, :, bmask].mean(axis=2))
        return np.concatenate(feats, axis=1)

    if cfg.feature_kind == "p300_mean_peak":
        windows = [(0.18, 0.28), (0.28, 0.42), (0.42, 0.60), (0.60, 0.80)]
        feats = []
        for a, b in windows:
            bmask = (times >= a) & (times < b)
            if not bmask.any():
                continue
            seg = epochs[:, :, bmask]
            feats.append(seg.mean(axis=2))
            feats.append(seg.max(axis=2))
            feats.append(seg.min(axis=2))
        return np.concatenate(feats, axis=1)

    raise ValueError(cfg.feature_kind)


def aggregate_repetitions_features(
    X: np.ndarray,
    y: np.ndarray | None,
    meta: pd.DataFrame,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    if not cfg.aggregate_repetitions:
        return X, y, meta.copy()

    rows = []
    feats = []
    labels = []
    for (sheet, code), idxs in meta.groupby(["sheet", "event_code"]).groups.items():
        idxs = np.array(list(idxs), dtype=int)
        feats.append(X[idxs].mean(axis=0))
        if y is not None:
            labels.append(int(np.max(y[idxs])))
        first = meta.iloc[idxs[0]].to_dict()
        first["n_repetitions"] = len(idxs)
        rows.append(first)
    return np.vstack(feats), (np.array(labels, dtype=int) if y is not None else None), pd.DataFrame(rows)


def get_model(name: str):
    if name == "lda":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", C=0.2, max_iter=5000)),
            ]
        )
    if name == "linearsvc":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(class_weight="balanced", C=0.05, max_iter=10000)),
            ]
        )
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier(class_weight="balanced", alpha=10.0))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=4, class_weight="balanced", random_state=7)
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=400, max_depth=4, class_weight="balanced", random_state=7)
    raise ValueError(name)


def score_model(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return np.asarray(s, dtype=float)
    return model.predict(X).astype(float)


def cv_eval(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, cfg: Config) -> tuple[dict, pd.DataFrame]:
    fold_rows = []
    y_all = []
    pred_all = []
    score_all = []
    for sheet in meta["sheet"].unique():
        val_mask = (meta["sheet"] == sheet).to_numpy()
        tr_mask = ~val_mask
        model = get_model(cfg.model_name)
        model.fit(X[tr_mask], y[tr_mask])
        scores = score_model(model, X[val_mask])
        # Threshold for epoch metrics only. Decoding uses rank, not threshold.
        preds = (scores >= np.median(score_model(model, X[tr_mask]))).astype(int)
        val_meta = meta.loc[val_mask].reset_index(drop=True)
        tmp = val_meta[["event_code"]].copy()
        tmp["score"] = scores
        if cfg.code_agg == "sum":
            code_scores = tmp.groupby("event_code")["score"].sum().to_dict()
        else:
            code_scores = tmp.groupby("event_code")["score"].mean().to_dict()
        decoded = decode_scores({int(k): float(v) for k, v in code_scores.items()}, cfg.mapping)
        actual_char = val_meta["target_char"].iloc[0]
        fold_rows.append(
            {
                "sheet": sheet,
                "actual_char": actual_char,
                "pred_char": decoded["char"],
                "char_correct": decoded["char"] == actual_char,
                "actual_row_event": int(val_meta["target_row_event"].iloc[0]),
                "actual_col_event": int(val_meta["target_col_event"].iloc[0]),
                "pred_row_event": decoded["row_event"],
                "pred_col_event": decoded["col_event"],
                "row_correct": decoded["row_event"] == int(val_meta["target_row_event"].iloc[0]),
                "col_correct": decoded["col_event"] == int(val_meta["target_col_event"].iloc[0]),
                "balanced_accuracy": balanced_accuracy_score(y[val_mask], preds),
                "f1": f1_score(y[val_mask], preds, zero_division=0),
                "auc": roc_auc_score(y[val_mask], scores) if len(np.unique(y[val_mask])) == 2 else np.nan,
            }
        )
        y_all.extend(y[val_mask].tolist())
        pred_all.extend(preds.tolist())
        score_all.extend(scores.tolist())

    folds = pd.DataFrame(fold_rows)
    metrics = {
        "char_acc": float(folds["char_correct"].mean()),
        "row_acc": float(folds["row_correct"].mean()),
        "col_acc": float(folds["col_correct"].mean()),
        "partial_acc": float((folds["row_correct"] | folds["col_correct"]).mean()),
        "epoch_bal_acc": float(balanced_accuracy_score(y_all, pred_all)),
        "epoch_f1": float(f1_score(y_all, pred_all, zero_division=0)),
        "epoch_auc": float(roc_auc_score(y_all, score_all)),
    }
    return metrics, folds


def predict_test(
    train_data,
    train_event,
    test_data,
    test_event,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xep, y, meta, times = build_epochs_for_config(train_data, train_event, cfg, known=True)
    X = epoch_features(Xep, times, cfg)
    X, y, meta = aggregate_repetitions_features(X, y, meta, cfg)

    Xtep, _, tmeta, ttimes = build_epochs_for_config(test_data, test_event, cfg, known=False)
    Xt = epoch_features(Xtep, ttimes, cfg)
    Xt, _, tmeta = aggregate_repetitions_features(Xt, None, tmeta, cfg)

    model = get_model(cfg.model_name)
    model.fit(X, y)
    scores = score_model(model, Xt)
    rows = []
    score_rows = []
    for sheet in tmeta["sheet"].unique():
        mask = (tmeta["sheet"] == sheet).to_numpy()
        part = tmeta.loc[mask].reset_index(drop=True)
        tmp = part[["event_code"]].copy()
        tmp["score"] = scores[mask]
        series = tmp.groupby("event_code")["score"].sum() if cfg.code_agg == "sum" else tmp.groupby("event_code")["score"].mean()
        code_scores = {int(k): float(v) for k, v in series.to_dict().items()}
        dec = decode_scores(code_scores, cfg.mapping)
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
        for code, score in code_scores.items():
            score_rows.append({"sheet": sheet, "event_code": code, "score": score})
    return pd.DataFrame(rows), pd.DataFrame(score_rows)


def main():
    train_data = read_workbook(DATA_DIR / "S1_train_data.xlsx")
    train_event = read_workbook(DATA_DIR / "S1_train_event.xlsx")
    test_data = read_workbook(DATA_DIR / "S1_test_data.xlsx")
    test_event = read_workbook(DATA_DIR / "S1_test_event.xlsx")

    configs = []
    grid = {
        "mapping": ["col1-6_row7-12", "row1-6_col7-12"],
        "sample_offset": [0, 1],
        "bandpass": [(0.1, 15.0), (0.5, 20.0), (0.5, 35.0), (1.0, 15.0)],
        "feature_window": [(0.0, 0.8), (0.15, 0.7), (0.25, 0.7), (0.3, 0.8)],
        "feature_kind": ["flat", "bins", "p300_mean_peak"],
        "aggregate_repetitions": [False, True],
        "model_name": ["lda", "logreg", "linearsvc", "ridge", "rf", "extratrees"],
    }
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        # Tree models on very high-dimensional flat single-epoch features are mostly noise and slow.
        if params["feature_kind"] == "flat" and not params["aggregate_repetitions"] and params["model_name"] in {"rf", "extratrees"}:
            continue
        configs.append(
            Config(
                **params,
                epoch=(-0.1, 0.9),
                baseline=(-0.1, 0.0),
                downsample=5,
                code_agg="mean",
            )
        )

    rows = []
    best_folds = None
    best_cfg = None
    best_key = None
    cache = {}
    for i, cfg in enumerate(configs, 1):
        cache_key = (cfg.mapping, cfg.sample_offset, cfg.bandpass, cfg.epoch, cfg.baseline, cfg.feature_window, cfg.feature_kind, cfg.downsample, cfg.aggregate_repetitions)
        try:
            if cache_key not in cache:
                Xep, y, meta, times = build_epochs_for_config(train_data, train_event, cfg, known=True)
                X = epoch_features(Xep, times, cfg)
                X, y, meta = aggregate_repetitions_features(X, y, meta, cfg)
                cache[cache_key] = (X, y, meta)
            X, y, meta = cache[cache_key]
            metrics, folds = cv_eval(X, y, meta, cfg)
        except Exception as exc:
            rows.append({"error": repr(exc), **cfg.__dict__})
            continue
        row = {**cfg.__dict__, **metrics}
        rows.append(row)
        key = (metrics["char_acc"], metrics["partial_acc"], metrics["epoch_auc"], metrics["row_acc"] + metrics["col_acc"])
        if best_key is None or key > best_key:
            best_key = key
            best_cfg = cfg
            best_folds = folds
            print(f"new best {i}/{len(configs)}: {key} {cfg}")

    results = pd.DataFrame(rows)
    results_sorted = results.sort_values(
        ["char_acc", "partial_acc", "epoch_auc", "row_acc", "col_acc"],
        ascending=False,
        na_position="last",
    )
    results_sorted.to_csv(OUTPUT_DIR / "optimization_results.csv", index=False, encoding="utf-8-sig")
    results_sorted.head(50).to_csv(OUTPUT_DIR / "optimization_top50.csv", index=False, encoding="utf-8-sig")
    print("\nTop 20:")
    print(results_sorted.head(20).to_string())

    if best_cfg is None:
        raise RuntimeError("no valid config")
    assert best_folds is not None
    best_folds.to_csv(OUTPUT_DIR / "best_cv_folds.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_cfg.__dict__, f, ensure_ascii=False, indent=2)

    pred, score_table = predict_test(train_data, train_event, test_data, test_event, best_cfg)
    pred.to_csv(OUTPUT_DIR / "best_test_predictions.csv", index=False, encoding="utf-8-sig")
    score_table.to_csv(OUTPUT_DIR / "best_test_code_scores.csv", index=False, encoding="utf-8-sig")
    print("\nBest config:", best_cfg)
    print("\nBest CV folds:")
    print(best_folds.to_string())
    print("\nBest test prediction:", "".join(pred["pred_char"].tolist()))
    print(pred.to_string(index=False))


if __name__ == "__main__":
    main()
