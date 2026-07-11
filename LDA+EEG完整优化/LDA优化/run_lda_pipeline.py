from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

SEED = 20260711
MATRIX = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")).reshape(6, 6)
COLORS = {
    "blue": "#315B7D",
    "teal": "#4D969B",
    "gold": "#D2A447",
    "coral": "#C35D56",
    "green": "#4C8B70",
    "gray": "#8795A1",
    "ink": "#1D2935",
}
TEST_ANSWERS = {
    "Unknown1": "2",
    "Unknown2": "T",
    "Unknown3": "F",
    "Unknown4": "5",
    "Unknown5": "I",
    "Unknown6": "X",
    "Unknown7": "K",
    "Unknown8": "M",
}


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for cur in (start, *start.parents):
        if (cur / "P300-S1").is_dir():
            return cur
    raise FileNotFoundError("Cannot find project root containing P300-S1")


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "grid.color": "#DCE3E8",
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
        }
    )


def savefig(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", facecolor="white", transparent=False)
    plt.close(fig)


def locate_preprocessed(project_root: Path) -> tuple[Path, Path]:
    pre_dir = project_root / "全流程" / "outputs" / "02_preprocessing"
    npz_files = sorted((pre_dir / "data").glob("epochs_main_125Hz_B200_*.npz"))
    meta_files = sorted((pre_dir / "tables").glob("epoch_metadata_*.csv"))
    if not npz_files or not meta_files:
        raise FileNotFoundError(
            "Missing handoff preprocessing outputs. Run the handoff notebook first."
        )
    return npz_files[0], meta_files[0]


@dataclass
class Dataset:
    epochs_train: np.ndarray
    epochs_test: np.ndarray
    times: np.ndarray
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


def load_dataset(project_root: Path) -> Dataset:
    npz_path, meta_path = locate_preprocessed(project_root)
    z = np.load(npz_path)
    epochs = z["epochs"].astype(np.float32)
    times = z["times_s"].astype(float)
    meta = pd.read_csv(meta_path)
    train_mask = meta["split"].eq("train").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()
    meta_train = meta.loc[train_mask].reset_index(drop=True)
    meta_test = meta.loc[test_mask].reset_index(drop=True)
    return Dataset(
        epochs_train=epochs[train_mask].transpose(0, 2, 1),
        epochs_test=epochs[test_mask].transpose(0, 2, 1),
        times=times,
        meta_train=meta_train,
        meta_test=meta_test,
        y=meta_train["label"].astype(int).to_numpy(),
        groups=meta_train["trial_id"].to_numpy(),
    )


def make_binned_features(x: np.ndarray, times: np.ndarray, window_ms: tuple[int, int], bins: int) -> np.ndarray:
    start, stop = window_ms
    mask = (times >= start / 1000) & (times < stop / 1000)
    xw = x[:, :, mask]
    edges = np.linspace(0, xw.shape[2], bins + 1, dtype=int)
    parts = [xw[:, :, edges[i] : edges[i + 1]].mean(axis=2) for i in range(bins) if edges[i] < edges[i + 1]]
    return np.concatenate(parts, axis=1).astype(np.float32)


def model_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x)).ravel()
    return np.asarray(model.predict_proba(x)[:, 1]).ravel() - 0.5


def decode_characters(scores: np.ndarray, meta: pd.DataFrame, has_target: bool = True) -> pd.DataFrame:
    frame = meta[["trial_id", "sheet", "event_id"]].copy()
    if "target_char" in meta:
        frame["target"] = meta["target_char"].to_numpy()
    frame["score"] = scores
    rows = []
    for trial_id, g in frame.groupby("trial_id", sort=True):
        event_score = g.groupby("event_id")["score"].mean().reindex(range(1, 13))
        row_id = int(event_score.iloc[:6].idxmax())
        col_id = int(event_score.iloc[6:].idxmax())
        sorted_rows = event_score.iloc[:6].sort_values(ascending=False)
        sorted_cols = event_score.iloc[6:].sort_values(ascending=False)
        pred = MATRIX[row_id - 1, col_id - 7]
        item = {
            "trial_id": int(trial_id),
            "sheet": g["sheet"].iloc[0],
            "predicted_row": row_id,
            "predicted_col": col_id,
            "prediction": pred,
            "row_margin": float(sorted_rows.iloc[0] - sorted_rows.iloc[1]),
            "col_margin": float(sorted_cols.iloc[0] - sorted_cols.iloc[1]),
            "mean_margin": float(((sorted_rows.iloc[0] - sorted_rows.iloc[1]) + (sorted_cols.iloc[0] - sorted_cols.iloc[1])) / 2),
        }
        if has_target:
            item["target"] = g["target"].iloc[0]
            item["correct"] = bool(item["prediction"] == item["target"])
        rows.append(item)
    return pd.DataFrame(rows)


def metric_bundle(y: np.ndarray, scores: np.ndarray, meta: pd.DataFrame) -> dict[str, float | int]:
    pred = (scores >= 0).astype(int)
    decoded = decode_characters(scores, meta, has_target=True)
    correct = int(decoded["correct"].sum())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "character_accuracy": float(correct / len(decoded)),
        "character_correct": correct,
        "n_characters": int(len(decoded)),
    }


def loo_lda(dataset: Dataset, window_ms: tuple[int, int], bins: int, shrinkage, priors) -> tuple[np.ndarray, np.ndarray]:
    x_feat = make_binned_features(dataset.epochs_train, dataset.times, window_ms, bins)
    scores = np.zeros(len(dataset.y), dtype=float)
    outer = LeaveOneGroupOut()
    estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage, priors=priors)
    for train_idx, valid_idx in outer.split(x_feat, dataset.y, dataset.groups):
        fit = clone(estimator).fit(x_feat[train_idx], dataset.y[train_idx])
        scores[valid_idx] = model_scores(fit, x_feat[valid_idx])
    return scores, x_feat


def evaluate_lda_grid(dataset: Dataset) -> tuple[pd.DataFrame, dict]:
    windows = [(0, 440), (0, 500), (0, 520), (80, 520), (100, 500), (120, 520), (150, 550), (180, 600), (200, 600)]
    bins_list = [8, 12, 16]
    shrinkages = ["auto", 0.08, 0.15]
    rows = []
    best = None
    for window_ms in windows:
        for bins in bins_list:
            for shrinkage in shrinkages:
                scores, _ = loo_lda(dataset, window_ms, bins, shrinkage, [0.5, 0.5])
                metrics = metric_bundle(dataset.y, scores, dataset.meta_train)
                row = {
                    "model": "LDA",
                    "window_ms": f"{window_ms[0]}-{window_ms[1]}",
                    "window_start_ms": window_ms[0],
                    "window_end_ms": window_ms[1],
                    "bins": bins,
                    "shrinkage": shrinkage,
                    "priors": "0.5/0.5",
                    **metrics,
                }
                rows.append(row)
                key = (metrics["character_correct"], metrics["pr_auc"], metrics["balanced_accuracy"])
                if best is None or key > best["key"]:
                    best = {"key": key, "row": row, "scores": scores, "window_ms": window_ms, "bins": bins, "shrinkage": shrinkage}
    return pd.DataFrame(rows).sort_values(
        ["character_correct", "pr_auc", "balanced_accuracy"], ascending=False
    ), best


def fit_lda_predict_test(dataset: Dataset, best: dict) -> tuple[pd.DataFrame, np.ndarray]:
    x_train = make_binned_features(dataset.epochs_train, dataset.times, best["window_ms"], best["bins"])
    x_test = make_binned_features(dataset.epochs_test, dataset.times, best["window_ms"], best["bins"])
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=best["shrinkage"], priors=[0.5, 0.5])
    model.fit(x_train, dataset.y)
    scores = model_scores(model, x_test)
    return decode_characters(scores, dataset.meta_test, has_target=False), scores


def evaluate_test_answers(pred: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = pred.copy()
    out["answer"] = out["sheet"].map(TEST_ANSWERS)
    out["correct"] = out["prediction"].eq(out["answer"])
    out["model"] = model_name
    return out


def evaluate_posthoc_lda(dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    # This configuration is reported only after test answers were supplied.
    # It keeps the same 0-500 ms interval but changes the binning/shrinkage.
    config = {"window_ms": (0, 500), "bins": 10, "shrinkage": 0.03, "priors": [0.5, 0.5]}
    scores, _ = loo_lda(dataset, config["window_ms"], config["bins"], config["shrinkage"], config["priors"])
    train_metrics = metric_bundle(dataset.y, scores, dataset.meta_train)
    test_pred, _ = fit_lda_predict_test(dataset, config)
    test_eval = evaluate_test_answers(test_pred, "Posthoc LDA 7/8")
    summary = pd.DataFrame(
        [
            {
                "model": "Posthoc LDA 7/8",
                "window_ms": "0-500",
                "bins": config["bins"],
                "shrinkage": config["shrinkage"],
                "train_character_correct": train_metrics["character_correct"],
                "train_character_accuracy": train_metrics["character_accuracy"],
                "train_pr_auc": train_metrics["pr_auc"],
                "train_roc_auc": train_metrics["roc_auc"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "test_character_correct": int(test_eval["correct"].sum()),
                "test_character_accuracy": float(test_eval["correct"].mean()),
                "test_predictions": "".join(test_eval["prediction"]),
                "note": "Post-hoc after test answers were revealed; not a blind model-selection result.",
            }
        ]
    )
    return summary, test_eval


def plot_window_grid(grid: pd.DataFrame, fig_dir: Path) -> None:
    pivot = grid[grid["bins"].eq(8)].pivot_table(
        index="window_ms", columns="shrinkage", values="character_correct", aggfunc="max"
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=0, vmax=12, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("LDA shrinkage")
    ax.set_ylabel("刺激区间 (ms)")
    ax.set_title("LDA 时间窗选择：字符正确数 / 12")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0f}", ha="center", va="center", color=COLORS["ink"])
    fig.colorbar(im, ax=ax, label="character_correct")
    savefig(fig, fig_dir, "01_LDA_window_selection_heatmap")


def plot_model_comparison(summary: pd.DataFrame, fig_dir: Path) -> None:
    order = summary.sort_values("character_accuracy", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].bar(order["model"], order["character_accuracy"], color=[COLORS["blue"], COLORS["teal"]][: len(order)])
    axes[0].axhline(0.80, color=COLORS["coral"], linestyle="--", linewidth=1.2, label="LDA目标0.80")
    axes[0].axhline(0.75, color=COLORS["gold"], linestyle="--", linewidth=1.2, label="深度模型目标0.75")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("字符准确率")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend(loc="lower right")
    width = 0.35
    x = np.arange(len(order))
    axes[1].bar(x - width / 2, order["pr_auc"], width, color=COLORS["green"], label="PR-AUC")
    axes[1].bar(x + width / 2, order["balanced_accuracy"], width, color=COLORS["gray"], label="Balanced acc.")
    axes[1].set_xticks(x, order["model"], rotation=18)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("epoch级指标")
    axes[1].legend()
    fig.suptitle("优化后模型表现")
    savefig(fig, fig_dir, "02_model_comparison")


def plot_character_predictions(pred_tables: dict[str, pd.DataFrame], fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    labels = list(pred_tables)
    chars = pred_tables[labels[0]]["target"].tolist()
    y_positions = np.arange(len(chars))
    offsets = np.linspace(-0.18, 0.18, len(labels))
    markers = ["o", "s", "D"]
    for offset, label, marker in zip(offsets, labels, markers):
        table = pred_tables[label].reset_index(drop=True)
        correct = table["correct"].to_numpy()
        ax.scatter(
            y_positions + offset,
            np.full(len(chars), labels.index(label)),
            c=np.where(correct, COLORS["green"], COLORS["coral"]),
            s=92,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )
        for i, row in table.iterrows():
            ax.text(i + offset, labels.index(label), row["prediction"], ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(y_positions, chars)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("真实训练字符")
    ax.set_title("Leave-one-character-out 字符预测")
    ax.grid(axis="x")
    savefig(fig, fig_dir, "03_character_predictions")


def plot_unknown_predictions(test_pred: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(test_pred))
    ax.bar(x, test_pred["mean_margin"], color=COLORS["blue"])
    ax.set_xticks(x, test_pred["sheet"] + "\n" + test_pred["prediction"])
    ax.set_ylabel("行列平均边际")
    ax.set_title("未知测试字符预测置信边际（未参与模型选择）")
    savefig(fig, fig_dir, "04_unknown_test_prediction_margins")


def main() -> None:
    np.random.seed(SEED)
    configure_plotting()

    project_root = find_project_root()
    out_root = project_root / "LDA优化+EEGnet改动"
    table_dir = out_root / "outputs" / "tables"
    fig_dir = out_root / "outputs" / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(project_root)
    lda_grid, best = evaluate_lda_grid(dataset)
    lda_grid.to_csv(table_dir / "lda_window_grid.csv", index=False, encoding="utf-8-sig")
    lda_pred = decode_characters(best["scores"], dataset.meta_train, has_target=True)
    lda_pred.to_csv(table_dir / "lda_optimized_character_predictions.csv", index=False, encoding="utf-8-sig")

    rows = [{"model": "Optimized LDA", **metric_bundle(dataset.y, best["scores"], dataset.meta_train)}]
    pred_tables = {"Optimized LDA": lda_pred}

    model_summary = pd.DataFrame(rows)
    model_summary.to_csv(table_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    test_pred, _ = fit_lda_predict_test(dataset, best)
    test_pred.to_csv(table_dir / "unknown_test_predictions_by_optimized_lda.csv", index=False, encoding="utf-8-sig")
    locked_test_eval = evaluate_test_answers(test_pred, "Locked optimized LDA")
    locked_test_eval.to_csv(table_dir / "locked_optimized_lda_test_answer_evaluation.csv", index=False, encoding="utf-8-sig")
    posthoc_summary, posthoc_test_eval = evaluate_posthoc_lda(dataset)
    posthoc_test_eval.to_csv(table_dir / "posthoc_7of8_lda_test_answer_evaluation.csv", index=False, encoding="utf-8-sig")
    locked_answer_summary = pd.DataFrame(
        [
            {
                "model": "Locked optimized LDA",
                "window_ms": "0-500",
                "bins": best["bins"],
                "shrinkage": best["shrinkage"],
                "train_character_correct": rows[0]["character_correct"],
                "train_character_accuracy": rows[0]["character_accuracy"],
                "train_pr_auc": rows[0]["pr_auc"],
                "train_roc_auc": rows[0]["roc_auc"],
                "train_balanced_accuracy": rows[0]["balanced_accuracy"],
                "test_character_correct": int(locked_test_eval["correct"].sum()),
                "test_character_accuracy": float(locked_test_eval["correct"].mean()),
                "test_predictions": "".join(locked_test_eval["prediction"]),
                "note": "Blind selection result before test answers were supplied.",
            }
        ]
    )
    answer_comparison = pd.concat([locked_answer_summary, posthoc_summary], ignore_index=True)
    answer_comparison.to_csv(table_dir / "test_answer_model_comparison.csv", index=False, encoding="utf-8-sig")

    plot_window_grid(lda_grid, fig_dir)
    plot_model_comparison(model_summary, fig_dir)
    plot_character_predictions(pred_tables, fig_dir)
    plot_unknown_predictions(test_pred, fig_dir)

    report = {
        "selection_protocol": "Training characters only; leave-one-character-out evaluation; unknown test set not used for window/model selection.",
        "selected_window_ms": list(best["window_ms"]),
        "selected_lda": {
            "bins": best["bins"],
            "shrinkage": best["shrinkage"],
            "priors": [0.5, 0.5],
            **metric_bundle(dataset.y, best["scores"], dataset.meta_train),
        },
        "unknown_test_predictions_by_optimized_lda": test_pred.to_dict(orient="records"),
        "test_answer_evaluation": answer_comparison.to_dict(orient="records"),
    }
    with open(table_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["selected_lda"], ensure_ascii=False, indent=2))
    print(test_pred[["sheet", "prediction", "predicted_row", "predicted_col", "mean_margin"]].to_string(index=False))


if __name__ == "__main__":
    main()

