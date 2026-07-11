from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn


SEED = 42
MATRIX = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")).reshape(6, 6)
TOP12_CHANNELS = ["Ch20", "Ch17", "Ch15", "Ch12", "Ch14", "Ch16", "Ch13", "Ch10", "Ch18", "Ch19", "Ch08", "Ch11"]
REPORT_WINDOWS = [(0, 600), (0, 500), (0, 400), (100, 500), (150, 450)]
ANSWERS = {
    "Unknown1": "2",
    "Unknown2": "T",
    "Unknown3": "F",
    "Unknown4": "5",
    "Unknown5": "I",
    "Unknown6": "X",
    "Unknown7": "K",
    "Unknown8": "M",
}


@dataclass(frozen=True)
class Dataset:
    epochs_train: np.ndarray
    epochs_test: np.ndarray
    times: np.ndarray
    channels: list[str]
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray


class EEGNetBinary(nn.Module):
    def __init__(self, n_channels: int, n_times: int, f1: int = 8, d: int = 2, f2: int = 16, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            flat_dim = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_times))).reshape(1, -1).shape[1]
        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x.flatten(start_dim=1)).squeeze(1)


def find_project_root() -> Path:
    start = Path.cwd().resolve()
    for cur in (start, *start.parents):
        if (cur / "P300-S1").is_dir() and (cur / "全流程").is_dir():
            return cur
    raise FileNotFoundError("Cannot find project root containing P300-S1 and 全流程")


def load_dataset(root: Path) -> Dataset:
    npz_path = next(root.rglob("epochs_main_125Hz_B200_*.npz"))
    meta_path = next(root.rglob("epoch_metadata_*.csv"))
    z = np.load(npz_path)
    epochs = z["epochs"].astype(np.float32)  # n_events, n_times, n_channels
    meta = pd.read_csv(meta_path)
    train_mask = meta["split"].eq("train").to_numpy()
    test_mask = meta["split"].eq("test").to_numpy()
    meta_train = meta.loc[train_mask].reset_index(drop=True)
    meta_test = meta.loc[test_mask].reset_index(drop=True)
    return Dataset(
        epochs_train=epochs[train_mask],
        epochs_test=epochs[test_mask],
        times=z["times_s"].astype(float),
        channels=[str(x) for x in z["channels"]],
        meta_train=meta_train,
        meta_test=meta_test,
        y=meta_train["label"].astype(int).to_numpy(),
        groups=meta_train["trial_id"].to_numpy(),
    )


def slice_epochs(dataset: Dataset, window_ms: tuple[int, int], channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    time_mask = (dataset.times >= window_ms[0] / 1000) & (dataset.times < window_ms[1] / 1000)
    channel_idx = [dataset.channels.index(ch) for ch in channels]
    return dataset.epochs_train[:, time_mask][:, :, channel_idx], dataset.epochs_test[:, time_mask][:, :, channel_idx]


def train_fold(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, seed: int, epochs: int = 80) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # x shape: n_events, n_times, n_channels
    mean = x[train_idx].mean(axis=(0, 1), keepdims=True)
    std = x[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    x_train = ((x[train_idx] - mean) / std).transpose(0, 2, 1)[:, None, :, :].astype(np.float32)
    x_valid = ((x[valid_idx] - mean) / std).transpose(0, 2, 1)[:, None, :, :].astype(np.float32)
    y_train = y[train_idx].astype(np.float32)

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    model = EEGNetBinary(n_channels=x.shape[2], n_times=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x_valid))).numpy()


def fit_full_predict(x_train: np.ndarray, y: np.ndarray, x_test: np.ndarray, seed: int, epochs: int = 80) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    x_fit = ((x_train - mean) / std).transpose(0, 2, 1)[:, None, :, :].astype(np.float32)
    x_pred = ((x_test - mean) / std).transpose(0, 2, 1)[:, None, :, :].astype(np.float32)
    y_fit = y.astype(np.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = EEGNetBinary(n_channels=x_train.shape[2], n_times=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    pos = float(y_fit.sum())
    neg = float(len(y_fit) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32))
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x_pred))).numpy()


def aggregate(scores: np.ndarray, meta: pd.DataFrame, has_target: bool, method: str = "top3") -> pd.DataFrame:
    frame = meta[["trial_id", "sheet", "event_id"]].copy()
    if "target_char" in meta:
        frame["target"] = meta["target_char"].to_numpy()
    frame["score"] = scores
    rows = []
    for trial_id, g in frame.groupby("trial_id", sort=True):
        event_scores = {}
        for event_id, e in g.groupby("event_id"):
            values = np.sort(e["score"].to_numpy())
            event_scores[int(event_id)] = float(values[-3:].mean()) if method == "top3" else float(values.mean())
        row_scores = pd.Series({i: event_scores.get(i, -np.inf) for i in range(1, 7)})
        col_scores = pd.Series({i: event_scores.get(i, -np.inf) for i in range(7, 13)})
        row_id = int(row_scores.idxmax())
        col_id = int(col_scores.idxmax())
        item = {
            "trial_id": int(trial_id),
            "sheet": g["sheet"].iloc[0],
            "prediction": MATRIX[row_id - 1, col_id - 7],
            "predicted_row": row_id,
            "predicted_col": col_id,
        }
        if has_target:
            item["target"] = g["target"].iloc[0]
            item["correct"] = bool(item["prediction"] == item["target"])
        rows.append(item)
    return pd.DataFrame(rows)


def run_config(dataset: Dataset, window_ms: tuple[int, int], channel_mode: str, channels: list[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test = slice_epochs(dataset, window_ms, channels)
    scores = np.zeros(len(dataset.y), dtype=float)
    start = time.time()
    for fold_id, (train_idx, valid_idx) in enumerate(LeaveOneGroupOut().split(x_train, dataset.y, dataset.groups), start=1):
        scores[valid_idx] = train_fold(x_train, dataset.y, train_idx, valid_idx, seed=SEED + fold_id)

    char_pred = aggregate(scores, dataset.meta_train, has_target=True)
    test_scores = fit_full_predict(x_train, dataset.y, x_test, seed=SEED + 999)
    test_pred = aggregate(test_scores, dataset.meta_test, has_target=False)
    test_pred["answer"] = test_pred["sheet"].map(ANSWERS)
    test_pred["correct"] = test_pred["prediction"].eq(test_pred["answer"])

    event_pred = (scores >= 0.5).astype(int)
    result = {
        "model": "eegnet",
        "window_ms": f"{window_ms[0]}-{window_ms[1]}",
        "channel_mode": channel_mode,
        "n_channels": len(channels),
        "channels": ",".join(channels),
        "aggregation": "top3",
        "epochs": 80,
        "lr": 0.001,
        "dropout": 0.5,
        "pos_weight_scale": 1.0,
        "event_balanced_accuracy": float(balanced_accuracy_score(dataset.y, event_pred)),
        "event_roc_auc": float(roc_auc_score(dataset.y, scores)),
        "event_pr_auc": float(average_precision_score(dataset.y, scores)),
        "event_f1": float(f1_score(dataset.y, event_pred, zero_division=0)),
        "char_accuracy": float(char_pred["correct"].mean()),
        "char_correct": int(char_pred["correct"].sum()),
        "unknown_accuracy": float(test_pred["correct"].mean()),
        "unknown_correct": int(test_pred["correct"].sum()),
        "unknown_predictions": "".join(test_pred["prediction"].astype(str)),
        "elapsed_s": round(time.time() - start, 2),
    }
    pd.DataFrame([result]).to_csv(out_dir / "validation_results.csv", index=False, encoding="utf-8-sig")
    char_pred.to_csv(out_dir / "validation_char_predictions.csv", index=False, encoding="utf-8-sig")
    test_pred.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "selected_config.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    root = find_project_root()
    dataset = load_dataset(root)
    out_root = root / "LDA+EEG完整优化" / "aligned_eegnet_experiments"
    configs = []
    for window_ms in REPORT_WINDOWS:
        configs.append((window_ms, "all20", dataset.channels))
        configs.append((window_ms, "lda_top12", TOP12_CHANNELS))
    rows = []
    for window_ms, channel_mode, channels in configs:
        out_dir = out_root / f"eegnet_{window_ms[0]}_{window_ms[1]}_{channel_mode}_top3"
        result_path = out_dir / "validation_results.csv"
        if result_path.exists():
            result = pd.read_csv(result_path).iloc[0].to_dict()
            print(f"skip existing: {out_dir.name}")
        else:
            result = run_config(dataset, window_ms, channel_mode, channels, out_dir)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    pd.DataFrame(rows).to_csv(out_root / "aligned_eegnet_channel_comparison.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

