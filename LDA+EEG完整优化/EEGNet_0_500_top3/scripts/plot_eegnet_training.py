from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC_DIR))

from p300_pipeline import Config, EEGNetBinary, EpochWindow, build_train_dataset


def train_with_history(X: np.ndarray, y: np.ndarray, cfg: Config) -> pd.DataFrame:
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n_times = X.shape[1] // cfg.n_channels
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X_epoch = ((X - mean) / std).reshape(X.shape[0], n_times, cfg.n_channels)
    X_epoch = np.transpose(X_epoch, (0, 2, 1))[:, None, :, :]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetBinary(
        cfg.n_channels,
        n_times,
        f1=cfg.eegnet_f1,
        d=cfg.eegnet_d,
        f2=cfg.eegnet_f2,
        dropout=cfg.eegnet_dropout,
    ).to(device)

    positives = float(y.sum())
    negatives = float(len(y) - positives)
    pos_weight = torch.tensor(
        [cfg.eegnet_pos_weight_scale * negatives / max(positives, 1.0)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.eegnet_lr,
        weight_decay=cfg.eegnet_weight_decay,
    )

    dataset = TensorDataset(torch.from_numpy(X_epoch), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(cfg.random_state)
    loader = DataLoader(
        dataset,
        batch_size=cfg.eegnet_batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.eegnet_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            seen += len(yb)

        model.eval()
        probs: list[np.ndarray] = []
        with torch.no_grad():
            for xb, _yb in DataLoader(dataset, batch_size=256, shuffle=False):
                logits = model(xb.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
        prob = np.concatenate(probs)
        pred = (prob >= 0.5).astype(int)
        y_int = y.astype(int)

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(seen, 1),
                "accuracy": accuracy_score(y_int, pred),
                "balanced_accuracy": balanced_accuracy_score(y_int, pred),
                "roc_auc": roc_auc_score(y_int, prob),
            }
        )

    return pd.DataFrame(history)


def setup_chinese_font() -> None:
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_loss(history: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(history["epoch"], history["loss"], color="#2F5597", linewidth=2)
    ax.set_title("最优 EEGNet 训练损失曲线（0-500 ms，top3）")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("BCE 损失")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(history: pd.DataFrame, output_path: Path) -> None:
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(history["epoch"], history["accuracy"], label="准确率", color="#70AD47", linewidth=2)
    ax.plot(
        history["epoch"],
        history["balanced_accuracy"],
        label="平衡准确率",
        color="#ED7D31",
        linewidth=2,
    )
    ax.plot(history["epoch"], history["roc_auc"], label="ROC-AUC", color="#8064A2", linewidth=2)
    ax.set_title("最优 EEGNet 训练指标曲线（0-500 ms，top3）")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("指标值")
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the best EEGNet training process.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"F:\university\大三课程\脑与认知\大作业\P300-S1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR,
    )
    args = parser.parse_args()

    selected_path = args.output_dir / "selected_config.json"
    with selected_path.open("r", encoding="utf-8-sig") as f:
        selected = json.load(f)

    cfg = replace(Config(), aggregate_method=selected.get("aggregate_method", "top3"))
    epoch = EpochWindow(float(selected["epoch_start_ms"]), float(selected["epoch_end_ms"]))
    paths = {
        "train_data": args.data_dir / "S1_train_data.xlsx",
        "train_event": args.data_dir / "S1_train_event.xlsx",
    }

    X_train, y_train, _groups, _meta = build_train_dataset(
        paths["train_data"], paths["train_event"], epoch, cfg
    )
    history = train_with_history(X_train, y_train, cfg)

    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "eegnet_training_history.csv"
    loss_png_path = figures_dir / "eegnet_training_loss.png"
    metrics_png_path = figures_dir / "eegnet_training_metrics.png"
    history.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_loss(history, loss_png_path)
    plot_metrics(history, metrics_png_path)

    print(f"Saved {csv_path}")
    print(f"Saved {loss_png_path}")
    print(f"Saved {metrics_png_path}")


if __name__ == "__main__":
    main()
