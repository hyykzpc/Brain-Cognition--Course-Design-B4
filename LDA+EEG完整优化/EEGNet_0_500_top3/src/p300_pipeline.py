from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, decimate, sosfiltfilt
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MATRIX = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("567890"),
]

DATA_CACHE: dict[tuple[str, str], np.ndarray] = {}
EVENT_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


@dataclass(frozen=True)
class Config:
    raw_fs: int = 250
    downsample_factor: int = 2
    low_hz: float = 0.5
    high_hz: float = 30.0
    baseline_sec: float = 0.2
    random_state: int = 42
    n_channels: int = 20
    eegnet_epochs: int = 80
    eegnet_batch_size: int = 64
    eegnet_lr: float = 1e-3
    eegnet_weight_decay: float = 1e-3
    eegnet_pos_weight_scale: float = 1.0
    eegnet_f1: int = 8
    eegnet_d: int = 2
    eegnet_f2: int = 16
    eegnet_dropout: float = 0.5
    eegnet_scheduler: bool = False
    xdawn_filters: int = 10
    aggregate_method: str = "mean"

    @property
    def fs(self) -> float:
        return self.raw_fs / self.downsample_factor

    @property
    def baseline_samples(self) -> int:
        return int(round(self.baseline_sec * self.fs))


@dataclass(frozen=True)
class EpochWindow:
    start_ms: float
    end_ms: float

    @property
    def label(self) -> str:
        return f"{self.start_ms:g}-{self.end_ms:g}ms"

    def start_samples(self, cfg: Config) -> int:
        return int(round(self.start_ms / 1000.0 * cfg.fs))

    def end_samples(self, cfg: Config) -> int:
        return int(round(self.end_ms / 1000.0 * cfg.fs))

    def n_samples(self, cfg: Config) -> int:
        return self.end_samples(cfg) - self.start_samples(cfg)


def char_to_events(ch: str) -> tuple[int, int]:
    for r, row in enumerate(MATRIX, start=1):
        for c, value in enumerate(row, start=1):
            if value == ch:
                return r, 6 + c
    raise ValueError(f"Unknown target character: {ch}")


def events_to_char(row_event: int, col_event: int) -> str:
    row = row_event
    col = col_event - 6
    return MATRIX[row - 1][col - 1]


def parse_target_from_sheet(sheet_name: str) -> str:
    match = re.search(r"\((.)\)", sheet_name)
    if not match:
        raise ValueError(f"Cannot parse target character from sheet name: {sheet_name}")
    return match.group(1)


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, header=None)


def get_preprocessed_data(path: Path, sheet_name: str, cfg: Config) -> np.ndarray:
    key = (str(path.resolve()), sheet_name)
    if key not in DATA_CACHE:
        DATA_CACHE[key] = preprocess_continuous(read_sheet(path, sheet_name).to_numpy(), cfg)
    return DATA_CACHE[key]


def get_valid_events(path: Path, sheet_name: str) -> pd.DataFrame:
    key = (str(path.resolve()), sheet_name)
    if key not in EVENT_CACHE:
        EVENT_CACHE[key] = valid_event_rows(read_sheet(path, sheet_name))
    return EVENT_CACHE[key]


def preprocess_continuous(data: np.ndarray, cfg: Config) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = data - np.nanmean(data, axis=0, keepdims=True)
    nyq = cfg.fs / 2.0
    sos = butter(
        N=4,
        Wn=[cfg.low_hz / nyq, cfg.high_hz / nyq],
        btype="bandpass",
        output="sos",
    )
    downsampled = decimate(data, cfg.downsample_factor, axis=0, ftype="fir", zero_phase=True)
    return sosfiltfilt(sos, downsampled, axis=0)


def valid_event_rows(event_df: pd.DataFrame) -> pd.DataFrame:
    clean = event_df.iloc[:, :2].dropna().copy()
    clean.columns = ["event_code", "row_index"]
    clean["event_code"] = clean["event_code"].astype(int)
    clean["row_index"] = clean["row_index"].astype(int)
    return clean[clean["event_code"].between(1, 12)].reset_index(drop=True)


def adjusted_event_index(row_index: int, cfg: Config) -> int:
    return int(np.floor((row_index - 1) / cfg.downsample_factor))


def extract_epoch(data: np.ndarray, row_index: int, epoch: EpochWindow, cfg: Config) -> np.ndarray | None:
    event_start = adjusted_event_index(row_index, cfg)
    baseline_start = event_start - cfg.baseline_samples
    start = event_start + epoch.start_samples(cfg)
    end = event_start + epoch.end_samples(cfg)
    if baseline_start < 0 or end > data.shape[0]:
        return None
    baseline = data[baseline_start:event_start].mean(axis=0, keepdims=True)
    return data[start:end] - baseline


def build_train_dataset(
    data_path: Path,
    event_path: Path,
    epoch: EpochWindow,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    data_book = pd.ExcelFile(data_path)
    event_book = pd.ExcelFile(event_path)
    X, y, groups = [], [], []
    meta: list[dict] = []
    for sheet_idx, sheet in enumerate(data_book.sheet_names):
        if sheet not in event_book.sheet_names:
            raise ValueError(f"Missing event sheet for {sheet}")
        target = parse_target_from_sheet(sheet)
        row_event, col_event = char_to_events(target)
        data = get_preprocessed_data(data_path, sheet, cfg)
        events = get_valid_events(event_path, sheet)
        for _, row in events.iterrows():
            sample_epoch = extract_epoch(data, int(row.row_index), epoch, cfg)
            if sample_epoch is None:
                continue
            code = int(row.event_code)
            label = int(code in (row_event, col_event))
            X.append(sample_epoch.reshape(-1))
            y.append(label)
            groups.append(sheet_idx)
            meta.append({"sheet": sheet, "target": target, "event_code": code, "row_index": int(row.row_index)})
    return np.asarray(X), np.asarray(y), np.asarray(groups), meta


def build_test_dataset(
    data_path: Path,
    event_path: Path,
    epoch: EpochWindow,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    data_book = pd.ExcelFile(data_path)
    event_book = pd.ExcelFile(event_path)
    X, meta = [], []
    for sheet in data_book.sheet_names:
        if sheet not in event_book.sheet_names:
            raise ValueError(f"Missing event sheet for {sheet}")
        data = get_preprocessed_data(data_path, sheet, cfg)
        events = get_valid_events(event_path, sheet)
        for _, row in events.iterrows():
            sample_epoch = extract_epoch(data, int(row.row_index), epoch, cfg)
            if sample_epoch is None:
                continue
            X.append(sample_epoch.reshape(-1))
            meta.append({"sample": sheet, "event_code": int(row.event_code), "row_index": int(row.row_index)})
    return np.asarray(X), meta


class EEGNetBinary(nn.Module):
    """Compact EEGNet-style binary classifier for small P300 epochs."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        dropout: float = 0.5,
    ) -> None:
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
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = self.block2(self.block1(dummy)).reshape(1, -1).shape[1]
        self.classifier = nn.Linear(flat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x).squeeze(1)


class EEGNetGlobalBinary(nn.Module):
    """EEGNet variant with global average pooling to reduce classifier parameters."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(f2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x).squeeze(1)


class P300CNNBinary(nn.Module):
    """Small P300-specific CNN with temporal filters, spatial mixing, and global pooling."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        temporal_filters: int = 12,
        spatial_filters: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(spatial_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x).squeeze(1)


class TorchEEGClassifier:
    """Minimal sklearn-like wrapper so the existing pipeline can call fit/predict_proba."""

    def __init__(self, cfg: Config, architecture: str = "eegnet") -> None:
        self.cfg = cfg
        self.architecture = architecture
        self.model: nn.Module | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_times_: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _epochs_from_flat(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.n_times_ = X.shape[1] // self.cfg.n_channels
            self.mean_ = X.mean(axis=0, keepdims=True)
            self.std_ = X.std(axis=0, keepdims=True) + 1e-6
        if self.n_times_ is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("TorchEEGClassifier must be fitted before prediction.")
        X = (X - self.mean_) / self.std_
        X = X.reshape(X.shape[0], self.n_times_, self.cfg.n_channels)
        X = np.transpose(X, (0, 2, 1))
        return X[:, None, :, :]

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.cfg.random_state)
        np.random.seed(self.cfg.random_state)
        X_epoch = self._epochs_from_flat(X, fit=True)
        y = np.asarray(y, dtype=np.float32)
        assert self.n_times_ is not None
        if self.architecture == "eegnet":
            self.model = EEGNetBinary(
                self.cfg.n_channels,
                self.n_times_,
                f1=self.cfg.eegnet_f1,
                d=self.cfg.eegnet_d,
                f2=self.cfg.eegnet_f2,
                dropout=self.cfg.eegnet_dropout,
            ).to(self.device)
        elif self.architecture == "eegnet_global":
            self.model = EEGNetGlobalBinary(
                self.cfg.n_channels,
                self.n_times_,
                f1=self.cfg.eegnet_f1,
                d=self.cfg.eegnet_d,
                f2=self.cfg.eegnet_f2,
                dropout=self.cfg.eegnet_dropout,
            ).to(self.device)
        elif self.architecture == "p300cnn":
            self.model = P300CNNBinary(
                self.cfg.n_channels,
                self.n_times_,
                temporal_filters=self.cfg.eegnet_f1,
                spatial_filters=self.cfg.eegnet_f2,
                dropout=self.cfg.eegnet_dropout,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown torch EEG architecture: {self.architecture}")

        positives = float(y.sum())
        negatives = float(len(y) - positives)
        pos_weight = torch.tensor(
            [self.cfg.eegnet_pos_weight_scale * negatives / max(positives, 1.0)],
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.eegnet_lr,
            weight_decay=self.cfg.eegnet_weight_decay,
        )

        dataset = TensorDataset(torch.from_numpy(X_epoch), torch.from_numpy(y))
        generator = torch.Generator().manual_seed(self.cfg.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.eegnet_batch_size,
            shuffle=True,
            generator=generator,
        )

        scheduler = None
        if self.cfg.eegnet_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cfg.eegnet_epochs, eta_min=self.cfg.eegnet_lr * 0.01
            )

        self.model.train()
        for _ in range(self.cfg.eegnet_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TorchEEGClassifier must be fitted before prediction.")
        X_epoch = self._epochs_from_flat(X, fit=False)
        loader = DataLoader(TensorDataset(torch.from_numpy(X_epoch)), batch_size=256, shuffle=False)
        probs = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                logits = self.model(xb.to(self.device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
        p1 = np.concatenate(probs)
        return np.column_stack([1.0 - p1, p1])


class XDawnLDAClassifier:
    """xDAWN-style supervised spatial filtering followed by a linear classifier.

    Supports LDA (default) and LogisticRegression via ``classifier_type``.
    """

    def __init__(
        self,
        cfg: Config,
        solver: str = "lsqr",
        shrinkage: str | float | None = "auto",
        classifier_type: str = "lda",
    ) -> None:
        self.cfg = cfg
        self.solver = solver
        self.shrinkage = shrinkage
        self.classifier_type = classifier_type
        self.filters_: np.ndarray | None = None
        self.n_times_: int | None = None

        if classifier_type == "lda":
            clf = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
        elif classifier_type == "logreg":
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=cfg.random_state,
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")

        self.model = Pipeline([("scale", StandardScaler()), ("clf", clf)])

    def _epochs_from_flat(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] % self.cfg.n_channels != 0:
            raise ValueError("Epoch feature length is not divisible by the configured channel count.")
        n_times = X.shape[1] // self.cfg.n_channels
        if self.n_times_ is None:
            self.n_times_ = n_times
        elif self.n_times_ != n_times:
            raise ValueError("Input epoch length differs from the fitted xDAWN epoch length.")
        return X.reshape(X.shape[0], n_times, self.cfg.n_channels)

    def _fit_filters(self, epochs: np.ndarray, y: np.ndarray) -> np.ndarray:
        centered = epochs - epochs.mean(axis=1, keepdims=True)
        signal_cov = np.einsum("ntc,ntd->cd", centered, centered)
        signal_cov /= max(centered.shape[0] * centered.shape[1], 1)

        target = epochs[y == 1].mean(axis=0)
        nontarget = epochs[y == 0].mean(axis=0)
        evoked_diff = target - nontarget
        evoked_cov = evoked_diff.T @ evoked_diff / max(evoked_diff.shape[0], 1)

        reg = 1e-3 * np.trace(signal_cov) / self.cfg.n_channels
        eigvals, eigvecs = eigh(evoked_cov, signal_cov + reg * np.eye(self.cfg.n_channels))
        order = np.argsort(eigvals)[::-1]
        n_filters = min(self.cfg.xdawn_filters, self.cfg.n_channels)
        return eigvecs[:, order[:n_filters]]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("XDawnLDAClassifier must be fitted before prediction.")
        epochs = self._epochs_from_flat(X)
        return (epochs @ self.filters_).reshape(epochs.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y, dtype=int)
        epochs = self._epochs_from_flat(X)
        self.filters_ = self._fit_filters(epochs, y)
        self.model.fit(self._transform(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._transform(X))


def make_model(name: str, cfg: Config, lda_solver: str = "lsqr", lda_shrinkage="auto", no_pca: bool = False):
    if name == "lda":
        clf = LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)
    elif name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=cfg.random_state,
        )
    elif name == "eegnet":
        return TorchEEGClassifier(cfg, architecture="eegnet")
    elif name == "eegnet_global":
        return TorchEEGClassifier(cfg, architecture="eegnet_global")
    elif name == "p300cnn":
        return TorchEEGClassifier(cfg, architecture="p300cnn")
    elif name in {"xdawn_lda", "xdawn"}:
        return XDawnLDAClassifier(cfg, solver=lda_solver, shrinkage=lda_shrinkage, classifier_type="lda")
    elif name == "xdawn_logreg":
        return XDawnLDAClassifier(cfg, solver=lda_solver, shrinkage=lda_shrinkage, classifier_type="logreg")
    else:
        raise ValueError(f"Unknown model: {name}")
    steps = [("scale", StandardScaler())]
    if not no_pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full", random_state=cfg.random_state)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def score_positive(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def aggregate_character(rows: list[dict], method: str = "mean") -> dict:
    scores_by_event = {event: [] for event in range(1, 13)}
    for row in rows:
        scores_by_event[int(row["event_code"])].append(float(row["score"]))

    event_scores = {}
    for event, scores in scores_by_event.items():
        if not scores:
            event_scores[event] = -np.inf
            continue
        values = np.asarray(scores, dtype=float)
        if method == "mean":
            event_scores[event] = float(values.mean())
        elif method.startswith("top"):
            k = int(method.removeprefix("top"))
            event_scores[event] = float(np.sort(values)[-k:].mean())
        else:
            raise ValueError(f"Unknown aggregate method: {method}")

    row_event = max(range(1, 7), key=lambda event: event_scores[event])
    col_event = max(range(7, 13), key=lambda event: event_scores[event])
    row_scores = np.asarray([event_scores[event] for event in range(1, 7)], dtype=float)
    col_scores = np.asarray([event_scores[event] for event in range(7, 13)], dtype=float)
    row_probs = row_scores / max(row_scores.sum(), 1e-12)
    col_probs = col_scores / max(col_scores.sum(), 1e-12)
    row_order = np.argsort(row_scores)[::-1]
    col_order = np.argsort(col_scores)[::-1]
    row_prob = float(row_probs[row_event - 1])
    col_prob = float(col_probs[col_event - 7])
    return {
        "pred_char": events_to_char(row_event, col_event),
        "pred_row_event": row_event,
        "pred_col_event": col_event,
        "pred_row": row_event,
        "pred_col": col_event - 6,
        "row_score": event_scores[row_event],
        "col_score": event_scores[col_event],
        "row_prob": row_prob,
        "col_prob": col_prob,
        "char_confidence": row_prob * col_prob,
        "row_margin": float(row_scores[row_order[0]] - row_scores[row_order[1]]),
        "col_margin": float(col_scores[col_order[0]] - col_scores[col_order[1]]),
        **{f"score_event_{event:02d}": event_scores[event] for event in range(1, 13)},
    }


def evaluate_candidate(
    model_name: str, epoch: EpochWindow, paths: dict[str, Path], cfg: Config,
    lda_solver: str = "lsqr", lda_shrinkage="auto", no_pca: bool = False,
) -> tuple[dict, list[dict]]:
    X, y, groups, meta = build_train_dataset(paths["train_data"], paths["train_event"], epoch, cfg)
    logo = LeaveOneGroupOut()
    event_rows: list[dict] = []
    char_rows: list[dict] = []

    for train_idx, val_idx in logo.split(X, y, groups):
        model = make_model(model_name, cfg, lda_solver=lda_solver, lda_shrinkage=lda_shrinkage, no_pca=no_pca)
        model.fit(X[train_idx], y[train_idx])
        scores = score_positive(model, X[val_idx])
        pred = (scores >= 0.5).astype(int)
        val_meta = [meta[i] for i in val_idx]
        for m, score, pred_label, true_label in zip(val_meta, scores, pred, y[val_idx]):
            event_rows.append({**m, "score": float(score), "pred_label": int(pred_label), "true_label": int(true_label)})

        sheet = val_meta[0]["sheet"]
        true_char = val_meta[0]["target"]
        agg = aggregate_character([row for row in event_rows if row["sheet"] == sheet], cfg.aggregate_method)
        char_rows.append({"sheet": sheet, "true_char": true_char, **agg, "correct": int(agg["pred_char"] == true_char)})

    event_df = pd.DataFrame(event_rows)
    char_df = pd.DataFrame(char_rows)
    metric = {
        "model": model_name,
        "epoch_ms": epoch.label,
        "epoch_start_ms": epoch.start_ms,
        "epoch_end_ms": epoch.end_ms,
        "response_samples": epoch.n_samples(cfg),
        "response_ms": round((epoch.end_ms - epoch.start_ms), 1),
        "aggregate_method": cfg.aggregate_method,
        "n_samples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "event_accuracy": accuracy_score(event_df["true_label"], event_df["pred_label"]),
        "event_balanced_accuracy": balanced_accuracy_score(event_df["true_label"], event_df["pred_label"]),
        "event_precision": precision_score(event_df["true_label"], event_df["pred_label"], zero_division=0),
        "event_recall": recall_score(event_df["true_label"], event_df["pred_label"], zero_division=0),
        "event_f1": f1_score(event_df["true_label"], event_df["pred_label"], zero_division=0),
        "event_roc_auc": roc_auc_score(event_df["true_label"], event_df["score"]),
        "char_accuracy": float(char_df["correct"].mean()),
    }
    return metric, char_rows


def predict_test(model, epoch: EpochWindow, paths: dict[str, Path], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_test, test_meta = build_test_dataset(paths["test_data"], paths["test_event"], epoch, cfg)
    scores = score_positive(model, X_test)
    event_df = pd.DataFrame([{**m, "score": float(s)} for m, s in zip(test_meta, scores)])

    predictions = []
    for sample, sample_df in event_df.groupby("sample", sort=False):
        agg = aggregate_character(sample_df.to_dict("records"), cfg.aggregate_method)
        predictions.append({"sample": sample, **agg})
    return pd.DataFrame(predictions), event_df


def write_report(
    output_dir: Path,
    cfg: Config,
    metrics: pd.DataFrame,
    selected: dict,
    validation_chars: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    if selected["model"] in {"eegnet", "p300cnn"}:
        feature_line = (
            f"- 特征/模型: {selected['model']} 直接学习 通道 x 时间 的卷积特征"
            f" (epochs={cfg.eegnet_epochs}, lr={cfg.eegnet_lr:g}, dropout={cfg.eegnet_dropout:g}, "
            f"pos_weight_scale={cfg.eegnet_pos_weight_scale:g})"
        )
    elif str(selected["model"]).startswith("xdawn"):
        feature_line = f"- 特征/模型: xDAWN 空间滤波({cfg.xdawn_filters} 个滤波器) + StandardScaler + shrinkage LDA"
    else:
        feature_line = "- 特征: 事件后窗口展开后进入 StandardScaler + PCA(95% 方差)"

    lines = [
        "# P300 字符识别实验结果",
        "",
        "## 预处理与建模",
        "",
        f"- 原始采样率: {cfg.raw_fs} Hz",
        f"- 降采样: {cfg.downsample_factor} 倍，建模采样率 {cfg.fs:g} Hz",
        f"- 带通滤波: {cfg.low_hz:g}-{cfg.high_hz:g} Hz",
        f"- 基线校正: 事件前 {cfg.baseline_sec:g} s",
        feature_line,
        f"- 字符聚合: {cfg.aggregate_method}",
        "- 验证: 12 个训练字符做 Leave-One-Character-Out 交叉验证",
        "",
        "## 最优配置",
        "",
        f"- 模型: {selected['model']}",
        f"- 响应窗口: {selected['epoch_ms']} ({selected['response_samples']} samples)",
        f"- 字符级验证准确率: {selected['char_accuracy']:.4f}",
        f"- 事件级 Balanced Accuracy: {selected['event_balanced_accuracy']:.4f}",
        f"- 事件级 ROC-AUC: {selected['event_roc_auc']:.4f}",
        "",
        "## Unknown 预测",
        "",
    ]
    for _, row in predictions.iterrows():
        lines.append(
            f"- {row['sample']} = {row['pred_char']} "
            f"(字符置信度={row['char_confidence']:.3f}, "
            f"行概率={row['row_prob']:.3f}, 列概率={row['col_prob']:.3f}, "
            f"行得分={row['row_score']:.3f}, 列得分={row['col_score']:.3f}, "
            f"行margin={row['row_margin']:.3f}, 列margin={row['col_margin']:.3f})"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- validation_results.csv: 不同模型和窗口的交叉验证指标",
            "- validation_char_predictions.csv: 留一字符验证的字符级预测明细",
            "- test_predictions.csv: 8 个 Unknown 的最终预测",
            "- test_event_scores.csv: 测试集每个闪烁事件的模型得分",
            "- model.pkl: 使用全部训练字符重训后的最终模型",
            "",
        ]
    )
    output_dir.joinpath("experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P300 character recognition pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("P300-S1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--models",
        default="lda,logreg,xdawn_lda",
        help="Comma-separated model names. Available: lda, logreg, xdawn_lda, eegnet, eegnet_global, p300cnn",
    )
    parser.add_argument(
        "--windows",
        default="60,80,100,120,150",
        help="Comma-separated response window lengths after downsampling. Used as 0-N ms when --epoch-ms is not set.",
    )
    parser.add_argument(
        "--epoch-ms",
        default=None,
        help="Comma-separated epoch ranges in milliseconds, for example: 200-600,250-650.",
    )
    parser.add_argument(
        "--aggregate-method",
        default="mean",
        help="How repeated flashes are aggregated per row/column. Available: mean, top2, top3.",
    )
    parser.add_argument("--eegnet-epochs", type=int, default=80)
    parser.add_argument("--eegnet-lr", type=float, default=1e-3)
    parser.add_argument("--eegnet-weight-decay", type=float, default=1e-3)
    parser.add_argument("--eegnet-pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--eegnet-f1", type=int, default=8)
    parser.add_argument("--eegnet-d", type=int, default=2)
    parser.add_argument("--eegnet-f2", type=int, default=16)
    parser.add_argument("--eegnet-dropout", type=float, default=0.5)
    parser.add_argument("--eegnet-scheduler", action="store_true", help="Use CosineAnnealingLR scheduler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--xdawn-filters", type=int, default=10, help="Number of xDAWN spatial filters.")
    parser.add_argument(
        "--lda-solver",
        default="lsqr",
        choices=["svd", "lsqr", "eigen"],
        help="LDA solver for both pure LDA and xDAWN+LDA pipelines.",
    )
    parser.add_argument(
        "--lda-shrinkage",
        default="auto",
        help="Shrinkage for LDA: 'auto', None (no shrinkage), or a float value.",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA in the pure LDA/LogReg pipeline.",
    )
    args = parser.parse_args()

    cfg = replace(
        Config(),
        aggregate_method=args.aggregate_method,
        eegnet_epochs=args.eegnet_epochs,
        eegnet_lr=args.eegnet_lr,
        eegnet_weight_decay=args.eegnet_weight_decay,
        eegnet_pos_weight_scale=args.eegnet_pos_weight_scale,
        eegnet_f1=args.eegnet_f1,
        eegnet_d=args.eegnet_d,
        eegnet_f2=args.eegnet_f2,
        eegnet_dropout=args.eegnet_dropout,
        eegnet_scheduler=args.eegnet_scheduler,
        xdawn_filters=args.xdawn_filters,
    )
    # Resolve LDA shrinkage: "auto" → "auto", "none" → None, else float
    lda_shrinkage = args.lda_shrinkage
    if lda_shrinkage.lower() == "none":
        lda_shrinkage = None
    elif lda_shrinkage.lower() != "auto":
        try:
            lda_shrinkage = float(lda_shrinkage)
        except ValueError:
            pass  # keep as string, e.g. "auto"
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_data": data_dir / "S1_train_data.xlsx",
        "train_event": data_dir / "S1_train_event.xlsx",
        "test_data": data_dir / "S1_test_data.xlsx",
        "test_event": data_dir / "S1_test_event.xlsx",
    }

    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if args.epoch_ms:
        epoch_windows = []
        for item in args.epoch_ms.split(","):
            start, end = item.strip().split("-", 1)
            epoch_windows.append(EpochWindow(float(start), float(end)))
    else:
        epoch_windows = [
            EpochWindow(0.0, int(value.strip()) / cfg.fs * 1000.0)
            for value in args.windows.split(",")
            if value.strip()
        ]

    metrics, char_predictions = [], {}
    for epoch in epoch_windows:
        for model_name in model_names:
            metric, char_rows = evaluate_candidate(
                model_name, epoch, paths, cfg,
                lda_solver=args.lda_solver, lda_shrinkage=lda_shrinkage, no_pca=args.no_pca,
            )
            metrics.append(metric)
            char_predictions[(model_name, epoch.label)] = char_rows
            print(
                f"{model_name:6s} epoch={epoch.label:>9s} "
                f"char_acc={metric['char_accuracy']:.3f} "
                f"bal_acc={metric['event_balanced_accuracy']:.3f} "
                f"auc={metric['event_roc_auc']:.3f}"
            )

    metrics_df = pd.DataFrame(metrics).sort_values(
        ["char_accuracy", "event_roc_auc", "event_balanced_accuracy"],
        ascending=[False, False, False],
    )
    selected = metrics_df.iloc[0].to_dict()
    selected_epoch = EpochWindow(float(selected["epoch_start_ms"]), float(selected["epoch_end_ms"]))
    best_key = (str(selected["model"]), str(selected["epoch_ms"]))
    validation_chars = pd.DataFrame(char_predictions[best_key])

    X_train, y_train, _groups, _meta = build_train_dataset(
        paths["train_data"], paths["train_event"], selected_epoch, cfg
    )
    model = make_model(str(selected["model"]), cfg, lda_solver=args.lda_solver, lda_shrinkage=lda_shrinkage, no_pca=args.no_pca)
    model.fit(X_train, y_train)
    predictions, test_events = predict_test(model, selected_epoch, paths, cfg)

    metrics_df.to_csv(output_dir / "validation_results.csv", index=False, encoding="utf-8-sig")
    validation_chars.to_csv(output_dir / "validation_char_predictions.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    test_events.to_csv(output_dir / "test_event_scores.csv", index=False, encoding="utf-8-sig")
    joblib.dump({"config": cfg, "selected": selected, "model": model}, output_dir / "model.pkl")
    output_dir.joinpath("selected_config.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir, cfg, metrics_df, selected, validation_chars, predictions)

    print("\nBest configuration:")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print("\nPredictions:")
    for _, row in predictions.iterrows():
        print(f"{row['sample']} = {row['pred_char']}")
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

