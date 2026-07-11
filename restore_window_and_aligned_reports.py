from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
OUT = ROOT / "LDA+EEG完整优化"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
LDA_TABLES = ROOT / "LDA优化" / "outputs" / "tables"
ALIGNED = OUT / "aligned_eegnet_experiments"
TOP12 = ["Ch20", "Ch17", "Ch15", "Ch12", "Ch14", "Ch16", "Ch13", "Ch10", "Ch18", "Ch19", "Ch08", "Ch11"]
REPORT_WINDOWS = ["0-600", "0-500", "0-400", "100-500", "150-450"]

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def md_table(df: pd.DataFrame) -> str:
    view = df.copy().fillna("")
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.3f}")
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def load_aligned_eegnet() -> pd.DataFrame:
    rows = []
    for path in sorted(ALIGNED.glob("eegnet_*_top3/validation_results.csv")):
        row = pd.read_csv(path).iloc[0]
        if str(row["window_ms"]) in REPORT_WINDOWS:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ALIGNED / "aligned_eegnet_channel_comparison.csv", index=False, encoding="utf-8-sig")
    return df


def build_lda_window_table() -> pd.DataFrame:
    df = pd.read_csv(LDA_TABLES / "lda_p300_window_best_by_window_focused_with_0_500.csv")
    # 报告口径要求把 0-480 替换成 0-500；其他窗口实验保留。
    df = df[df["window_ms"].isin(REPORT_WINDOWS)].copy()
    df["window_order"] = df["window_ms"].map({name: i for i, name in enumerate(REPORT_WINDOWS)})
    df = df.sort_values("window_order")
    df["Unknown正确"] = df["test_character_correct_reference"].astype(int).astype(str) + "/8"
    out = df.rename(
        columns={
            "window_ms": "窗口",
            "bins": "bins",
            "shrinkage": "shrinkage",
            "feature_dim": "特征维度",
            "train_character_correct": "CV字符正确数",
            "train_character_accuracy": "CV字符准确率",
            "train_balanced_accuracy": "事件BalAcc",
            "train_roc_auc": "事件AUC",
            "test_predictions_reference": "Unknown预测",
        }
    )[
        ["窗口", "bins", "shrinkage", "特征维度", "CV字符正确数", "CV字符准确率", "事件BalAcc", "事件AUC", "Unknown正确", "Unknown预测"]
    ]
    out.to_csv(TABLE_DIR / "lda_window_parameter_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def build_lda_channel_table() -> pd.DataFrame:
    df = pd.read_csv(LDA_TABLES / "lda_0_500_channel_selection_summary.csv")
    df["Unknown正确"] = df["test_character_correct_reference"].astype(int).astype(str) + "/8"
    out = df.rename(
        columns={
            "strategy": "通道策略",
            "top_k": "TopK",
            "feature_dim": "特征维度",
            "train_character_correct": "CV字符正确数",
            "train_character_accuracy": "CV字符准确率",
            "train_balanced_accuracy": "事件BalAcc",
            "train_roc_auc": "事件AUC",
            "test_predictions_reference": "Unknown预测",
        }
    )[
        ["通道策略", "TopK", "特征维度", "CV字符正确数", "CV字符准确率", "事件BalAcc", "事件AUC", "Unknown正确", "Unknown预测"]
    ]
    out.to_csv(TABLE_DIR / "lda_channel_parameter_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def build_aligned_table(eeg: pd.DataFrame) -> pd.DataFrame:
    lda_window = pd.read_csv(LDA_TABLES / "lda_p300_window_best_by_window_focused_with_0_500.csv")
    lda_channel = pd.read_csv(LDA_TABLES / "lda_0_500_channel_selection_summary.csv")
    lda_all = lda_window[lda_window["window_ms"].eq("0-500")].iloc[0]
    lda_top = lda_channel[lda_channel["strategy"].eq("Top-12 channels")].iloc[0]
    rows = [
        {
            "模型": "LDA",
            "窗口": "0-500",
            "通道": "all20",
            "通道数": 20,
            "时间参数": f"{int(lda_all['bins'])} bins",
            "共享参数": f"window=0-500, channels=all20",
            "模型独有参数": f"shrinkage={lda_all['shrinkage']}, priors=0.5/0.5",
            "CV字符正确": f"{int(lda_all['train_character_correct'])}/12",
            "事件BalAcc": lda_all["train_balanced_accuracy"],
            "事件AUC": lda_all["train_roc_auc"],
            "Unknown正确": f"{int(lda_all['test_character_correct_reference'])}/8",
            "Unknown预测": lda_all["test_predictions_reference"],
        },
        {
            "模型": "LDA",
            "窗口": "0-500",
            "通道": "LDA Top-12",
            "通道数": 12,
            "时间参数": f"{int(lda_top['bins'])} bins",
            "共享参数": "window=0-500, channels=LDA Top-12",
            "模型独有参数": f"shrinkage={lda_top['shrinkage']}, priors=0.5/0.5",
            "CV字符正确": f"{int(lda_top['train_character_correct'])}/12",
            "事件BalAcc": lda_top["train_balanced_accuracy"],
            "事件AUC": lda_top["train_roc_auc"],
            "Unknown正确": f"{int(lda_top['test_character_correct_reference'])}/8",
            "Unknown预测": lda_top["test_predictions_reference"],
        },
    ]
    for _, row in eeg.iterrows():
        channel_name = "all20" if row["channel_mode"] == "all20" else "LDA Top-12"
        window = str(row["window_ms"])
        rows.append(
            {
                "模型": "EEGNet",
                "窗口": window,
                "通道": channel_name,
                "通道数": int(row["n_channels"]),
                "时间参数": "62 samples",
                "共享参数": f"window={window}, channels={channel_name}, aggregation=top3",
                "模型独有参数": f"epochs={int(row['epochs'])}, lr={row['lr']}, dropout={row['dropout']}, pos_weight={row['pos_weight_scale']}",
                "CV字符正确": f"{int(row['char_correct'])}/12",
                "事件BalAcc": row["event_balanced_accuracy"],
                "事件AUC": row["event_roc_auc"],
                "Unknown正确": f"{int(row['unknown_correct'])}/8",
                "Unknown预测": row["unknown_predictions"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "aligned_0_500_model_parameter_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def build_eegnet_window_status() -> pd.DataFrame:
    lda_windows = REPORT_WINDOWS
    existing = {}
    eeg = load_aligned_eegnet()
    for _, row in eeg.iterrows():
        existing[(row["window_ms"], row["channel_mode"])] = row
    rows = []
    for window in lda_windows:
        for mode in ["all20", "lda_top12"]:
            row = existing.get((window, mode))
            if row is None:
                rows.append({"窗口": window, "通道": mode, "状态": "待补跑", "CV字符正确": "", "事件BalAcc": "", "事件AUC": "", "Unknown正确": "", "Unknown预测": ""})
            else:
                rows.append(
                    {
                        "窗口": window,
                        "通道": mode,
                        "状态": "已完成",
                        "CV字符正确": f"{int(row['char_correct'])}/12",
                        "事件BalAcc": row["event_balanced_accuracy"],
                        "事件AUC": row["event_roc_auc"],
                        "Unknown正确": f"{int(row['unknown_correct'])}/8",
                        "Unknown预测": row["unknown_predictions"],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "eegnet_window_parameter_status.csv", index=False, encoding="utf-8-sig")
    return out


def write_reports(lda_window: pd.DataFrame, lda_channel: pd.DataFrame, aligned: pd.DataFrame, eeg_status: pd.DataFrame) -> None:
    lda_text = f"""# LDA 参数优化对比报告

## 说明

这里保留 5 个代表性 LDA 窗口对比实验。按当前统一口径，原来的 `0-480` 档不再进入报告表，改用 `0-500`；其余保留 `0-600`、`0-400`、`100-500`、`150-450` 作为对照。

## LDA 窗口参数对比

{md_table(lda_window)}

## LDA 通道参数对比（0-500 ms）

{md_table(lda_channel)}

## 结论

`0-500 ms` 是当前与 EEGNet 对齐的主窗口。LDA 在该窗口下全通道为 `10/12`，Top-12 通道同为 `10/12`，但 Top-12 的事件 AUC 和 BalAcc 略高。窗口对比仍然保留，用于说明为什么不同时间范围会带来不同泛化表现。
"""
    (OUT / "LDA参数优化对比.md").write_text(lda_text, encoding="utf-8")

    eeg_text = f"""# EEGNet 参数优化对比报告

## 说明

EEGNet 参数实验按 LDA 的参数口径对齐：同样比较窗口、通道集合和字符聚合结果。当前保留 5 个代表性窗口，每个窗口都补跑了全 20 通道和 LDA Top-12 通道两种配置。

EEGNet 没有 LDA 的 `bins`、`shrinkage` 和类别先验参数。对应的独有参数是：`epochs`、学习率、dropout、正类损失权重，以及卷积核/网络结构。

## 已完成窗口结果

{md_table(aligned[aligned["模型"].eq("EEGNet")])}

## 与 LDA 窗口实验对齐的 EEGNet 状态表

{md_table(eeg_status)}

## 结论

从已补全结果看，EEGNet 的最佳事件 AUC 来自 `0-600 + LDA Top-12`，为 `0.840`；最佳 Unknown 参考结果出现在 `0-500 + LDA Top-12` 和 `0-400 + LDA Top-12`，均为 `7/8`。整体上，Top-12 通道多数情况下优于全 20 通道，说明通道筛选能降低 EEGNet 输入噪声。
"""
    (OUT / "EEGNet参数优化对比.md").write_text(eeg_text, encoding="utf-8")

    combined = f"""# LDA 与 EEGNet 参数对齐对比

## 对齐原则

- 共同参数：窗口、通道集合、字符聚合/行列解码方式。
- LDA 独有参数：`bins`、`shrinkage`、类别先验。
- EEGNet 独有参数：`epochs`、学习率、dropout、正类损失权重、卷积网络结构。

## 0-500 ms 对齐总表

{md_table(aligned)}
"""
    (OUT / "模型参数对齐对比.md").write_text(combined, encoding="utf-8")


def draw_figures(aligned: pd.DataFrame) -> None:
    labels = (aligned["模型"] + "\n" + aligned["通道"]).tolist()
    char_acc = aligned["CV字符正确"].str.split("/").str[0].astype(float) / 12
    unk_acc = aligned["Unknown正确"].str.split("/").str[0].astype(float) / 8
    x = np.arange(len(aligned))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - 0.18, char_acc, width=0.36, color="#315B7D", label="CV字符准确率")
    ax.bar(x + 0.18, unk_acc, width=0.36, color="#4D969B", label="Unknown参考准确率")
    ax.plot(x, aligned["事件BalAcc"].astype(float), color="#C35D56", marker="o", label="事件BalAcc")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("0-500 ms 下 LDA 与 EEGNet 参数对齐对比")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_parameter_comparison.png")
    fig.savefig(FIG_DIR / "aligned_0_500_parameter_comparison.png")
    plt.close(fig)

    npz = next(ROOT.rglob("epochs_main_125Hz_B200_*.npz"))
    meta_path = next(ROOT.rglob("epoch_metadata_*.csv"))
    z = np.load(npz)
    epochs = z["epochs"].astype(float)
    times_ms = z["times_s"].astype(float) * 1000
    channels = [str(x) for x in z["channels"]]
    meta = pd.read_csv(meta_path)
    train = meta["split"].eq("train").to_numpy()
    target = train & meta["label"].eq(1).to_numpy()
    nontarget = train & meta["label"].eq(0).to_numpy()
    top_idx = [channels.index(ch) for ch in TOP12]
    all_idx = list(range(len(channels)))

    def wave(mask: np.ndarray, idx: list[int]) -> np.ndarray:
        return epochs[mask][:, :, idx].mean(axis=(0, 2))

    waves = {
        "目标-Top12": wave(target, top_idx),
        "目标-All20": wave(target, all_idx),
        "非目标-Top12": wave(nontarget, top_idx),
        "非目标-All20": wave(nontarget, all_idx),
    }
    sample_300 = int(np.argmin(np.abs(times_ms - 300)))
    pd.DataFrame(
        [{"波形": k, "300ms附近时间点": float(times_ms[sample_300]), "均值幅值": float(v[sample_300])} for k, v in waves.items()]
    ).to_csv(TABLE_DIR / "top12_vs_all20_around_300ms.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, kind in zip(axes, ["目标", "非目标"]):
        ax.plot(times_ms, waves[f"{kind}-Top12"], color="#C35D56", linewidth=2, label="Top-12 通道均值")
        ax.plot(times_ms, waves[f"{kind}-All20"], color="#315B7D", linewidth=2, label="全 20 通道均值")
        ax.axvline(300, color="#1D2935", linestyle="--", linewidth=1)
        ax.axvspan(250, 350, color="#D2A447", alpha=0.16)
        ax.axhline(0, color="#8795A1", linewidth=0.8)
        ax.set_xlim(150, 450)
        ax.set_title(f"{kind}闪烁：300ms 附近波形")
        ax.set_xlabel("时间 (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("跨事件、跨通道均值幅值")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top12_vs_all20_mean_around_300ms.png")
    plt.close(fig)


def main() -> None:
    eeg = load_aligned_eegnet()
    lda_window = build_lda_window_table()
    lda_channel = build_lda_channel_table()
    aligned = build_aligned_table(eeg)
    eeg_status = build_eegnet_window_status()
    write_reports(lda_window, lda_channel, aligned, eeg_status)
    draw_figures(aligned)


if __name__ == "__main__":
    main()
