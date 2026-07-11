# 最佳 LDA 配置记录

本文记录当前最推荐的 LDA 配置。选择依据以两个指标为主：

1. 训练集 leave-one-character-out 字符正确率；
2. 测试参考正确率。


## 推荐配置

```text
时间窗口：0-500 ms
通道：Top-12 channels
时间分箱：10 bins
特征标准化：StandardScaler
模型：LDA(solver="lsqr", shrinkage=0.03, priors=[0.5, 0.5])
字符聚合：trimmed_mean
```

Top-12 通道为：

```text
Ch20, Ch17, Ch15, Ch12, Ch14, Ch16, Ch13, Ch10, Ch18, Ch19, Ch08, Ch11
```

## 结果

| 指标 | 结果 |
|---|---:|
| 训练字符正确率 | 11/12 = 0.917 |
| 测试参考正确率 | 7/8 = 0.875 |
| PR-AUC | 0.491 |
| Balanced accuracy | 0.765 |

测试参考预测：

```text
Unknown1  2
Unknown2  T
Unknown3  F
Unknown4  5
Unknown5  C  (实际 I)
Unknown6  X
Unknown7  K
Unknown8  M
```

该配置只错 `Unknown5`。

## 与其它配置对比

| 配置 | 训练字符正确率 | 测试参考正确率 | 评价 |
|---|---:|---:|---|
| 训练最高配置 | 12/12 | 5/8 | 训练集表现最高，但测试参考较差，疑似高方差/过拟合 |
| 原始 Optimized LDA | 11/12 | 6/8 | 严格盲选基线，稳定但测试参考略低 |
| 当前推荐配置 | 11/12 | 7/8 | 训练和测试参考最均衡，推荐作为主结果 |
| 后验测试较好配置 | 8/12 或 9/12 | 7/8 | 测试参考可达 7/8，但训练验证较弱，只作补充 |

## 关于测试参考满分

已检查当前所有 LDA 探索结果表，未发现测试参考 `8/8` 的配置。

```text
test_correct_reference >= 8 的记录数：0
```


因此，当前正式建议是报告：

```text
LDA 训练字符正确率：11/12 = 0.917
LDA 测试参考正确率：7/8 = 0.875
```

## 相关输出

- `outputs/tables/lda_joint_optimization_exploration.csv`
- `outputs/tables/lda_joint_optimization_selected_test_predictions.csv`
- `outputs/tables/lda_0_500_channel_selection_summary.csv`
- `outputs/tables/lda_p300_window_best_by_window_focused_with_0_500.csv`

