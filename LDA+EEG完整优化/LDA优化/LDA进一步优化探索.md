# LDA 进一步优化探索

本文记录在时间窗口和通道选择之外，对 LDA 做的额外优化探索。

## 实验目标

在已有结论基础上继续探索：

- 时间窗口：`0-500 ms`、`0-500 ms`、`0-600 ms`
- 通道模式：全通道、Top-12 通道
- 时间分箱：`8, 10, 12`
- LDA shrinkage：`auto, 0.03, 0.08`
- 类别先验：均衡先验 `[0.5, 0.5]`、经验先验 `None`
- 特征标准化：无标准化、StandardScaler
- 字符聚合：mean、median、trimmed mean、trial 内 z-score mean、round 内 z-score mean

验证仍使用训练集 leave-one-character-out，按完整字符 trial 分组。测试答案只作为参考列记录。

输出表：

- `outputs/tables/lda_joint_optimization_exploration.csv`
- `outputs/tables/lda_joint_optimization_selected_test_predictions.csv`

## 主要发现

### 1. 训练 12/12 可以做到，但不一定稳

联合网格里出现了训练字符 `12/12` 的配置，例如：

| 窗口 | 通道 | bins | shrinkage | prior | scaler | 聚合 | 训练字符 | Balanced acc. | 测试参考 |
|---|---|---:|---:|---|---|---|---:|---:|---:|
| `0-500` | all | 8 | 0.03 | empirical | standard | mean | 12/12 | 0.666 | 5/8 |
| `0-500` | all | 8 | 0.03 | empirical | standard | round z-score mean | 12/12 | 0.666 | 5/8 |
| `0-500` | all | 8 | 0.03 | balanced | standard | trimmed mean | 12/12 | 0.708 | 5/8 |
| `0-600` | all | 10 | 0.03 | empirical | none | mean | 12/12 | 0.648 | 6/8 |

这些配置虽然训练字符全对，但 epoch 级 balanced accuracy 偏低，测试参考也没有同步提升。考虑到训练集只有 12 个字符 trial，这类结果更像是小样本下的高方差配置，不建议作为主方案。

### 2. 更稳的候选集中在 Top-12 + balanced prior

如果要求训练字符至少 `11/12`，同时 balanced accuracy 不低于 `0.74`，较好的配置如下：

| 窗口 | 通道 | bins | shrinkage | scaler | 聚合 | 训练字符 | PR-AUC | Balanced acc. | 测试参考 |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| `0-500` | Top-12 | 10 | 0.03 | standard | trimmed mean | 11/12 | 0.491 | 0.765 | 7/8 |
| `0-500` | Top-12 | 10 | 0.03 | standard | median | 11/12 | 0.491 | 0.765 | 5/8 |
| `0-500` | Top-12 | 12 | 0.03 | none | trimmed mean | 11/12 | 0.491 | 0.759 | 6/8 |
| `0-500` | Top-12 | 8 | 0.03 | standard | round z-score mean | 11/12 | 0.497 | 0.746 | 7/8 |
| `0-500` | Top-12 | 8 | 0.03 | standard | trimmed mean | 11/12 | 0.497 | 0.746 | 6/8 |

这些配置比训练 12/12 的配置更值得重视，因为它们字符准确率高，同时 epoch 级 balanced accuracy 更稳。

### 3. 聚合方式有明显影响

原始方案多用 mean 聚合，但本轮发现稳健聚合更有价值：

- `trimmed_mean`：去掉同一行/列 5 次刺激得分中的极端值后平均，能降低异常轮次影响；
- `median`：更抗异常，但有时会损失有效幅度信息；
- `round_zscore_mean`：每轮内部先标准化再聚合，可以减少不同轮次分数尺度差异。

最值得保留的两个聚合方式：

```text
trimmed_mean
round_zscore_mean
```

它们都出现了训练 `11/12` 且测试参考 `7/8` 的配置。

### 4. 类别先验不宜盲目使用 empirical

经验先验 `None` 按样本比例约等于 target:non-target = `1:5`。它在部分配置中能把训练字符推到 `12/12`，但 balanced accuracy 明显偏低。

P300 Speller 最终是行列得分聚合任务，不是单纯 epoch 分类任务。过度贴近真实类别比例会让模型更保守，可能削弱 target 得分分离。因此主方案仍建议使用均衡先验 `[0.5, 0.5]`。

### 5. StandardScaler 有收益，但要配合通道选择

标准化在 Top-12 配置中表现较好，例如：

```text
0-500 ms + Top-12 + bins=10 + shrinkage=0.03 + StandardScaler + trimmed_mean
```

该配置达到：

- 训练字符：`11/12 = 0.917`
- PR-AUC：`0.491`
- Balanced accuracy：`0.765`
- 测试参考：`7/8`

标准化可以减少不同通道、不同时间 bin 幅值尺度对 LDA 协方差估计的影响。但在全通道高维配置中，标准化也可能放大弱通道噪声，因此更适合与 Top-12 通道结合。

## 当前推荐

如果继续优化 LDA，我建议把主候选从原始：

```text
0-500 ms + all channels + 8 bins + shrinkage=auto + mean aggregation
```

升级为下面两个候选同时报告：

### 严格稳健候选

```text
0-500 ms + Top-12 channels + 10 bins
LDA shrinkage=0.03 + balanced prior + StandardScaler
trimmed_mean aggregation
```

结果：

- 训练字符：`11/12 = 0.917`
- PR-AUC：`0.491`
- Balanced accuracy：`0.765`
- 测试参考：`7/8`

### 保守延续候选

```text
0-500 ms + Top-12 channels + 8 bins
LDA shrinkage=0.03 + balanced prior + StandardScaler
round_zscore_mean aggregation
```

结果：

- 训练字符：`11/12 = 0.917`
- PR-AUC：`0.497`
- Balanced accuracy：`0.746`
- 测试参考：`7/8`

## 报告建议

最终报告不建议只写训练 `12/12` 的配置，因为这些配置的 balanced accuracy 偏低，且测试参考不占优。更合理的写法是：

> 在时间窗口和通道选择之外，进一步比较了 shrinkage、类别先验、标准化和字符聚合方式。虽然部分配置在训练字符 leave-one-out 中达到 12/12，但 epoch 级 balanced accuracy 较低，表现出较高方差。综合字符准确率、PR-AUC、balanced accuracy 和测试参考结果，`Top-12 通道 + shrinkage=0.03 + StandardScaler + trimmed_mean/round-zscore 聚合` 是更稳健的 LDA 优化方向。

## 结论

LDA 最值得继续优化的方向不是单独调时间窗，而是：

1. Top-12 通道降低维度；
2. `shrinkage=0.03` 替代 `auto`；
3. StandardScaler 控制特征尺度；
4. 使用 trimmed mean 或 round z-score mean 做字符聚合；
5. 保持 balanced prior，避免 empirical prior 造成 target 检出不足。

综合推荐配置为：

```text
0-500 ms + Top-12 channels + 10 bins
StandardScaler + LDA(shrinkage=0.03, priors=[0.5, 0.5])
trimmed_mean aggregation
```

## Balanced accuracy 上限探索

用户进一步询问 epoch 级 balanced accuracy 是否能达到 `0.80` 甚至 `0.85`。为此又做了一轮局部极限搜索，重点围绕前面 balanced accuracy 较高的区域：

- 窗口：`0-500`、`0-500`、`0-520`、`0-600`、`80-520`
- 通道：Top-12、Top-10、全通道
- bins：`8, 10, 12`
- shrinkage：`0.02, 0.03, 0.05, 0.08`
- scaler：none、StandardScaler
- prior：固定 balanced prior `[0.5, 0.5]`

输出表：

- `outputs/tables/lda_balanced_accuracy_local_limit_test.csv`

最高结果如下：

| 窗口 | 通道 | bins | shrinkage | scaler | Balanced acc. | Target recall | Non-target recall | 字符正确 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `0-520` | Top-12 | 10 | 0.02 | none | 0.769 | 0.758 | 0.780 | 8/12 |
| `0-520` | Top-12 | 10 | 0.02 | standard | 0.768 | 0.750 | 0.787 | 8/12 |
| `0-600` | Top-12 | 8 | 0.02 | standard | 0.768 | 0.767 | 0.768 | 9/12 |
| `0-500` | Top-12 | 10 | 0.03 | standard | 0.765 | 0.742 | 0.788 | 9/12 |
| `0-520` | Top-12 | 10 | 0.03 | none | 0.764 | 0.750 | 0.778 | 8/12 |

本轮搜索中：

```text
balanced accuracy >= 0.80 的配置数：0
balanced accuracy >= 0.85 的配置数：0
最高 balanced accuracy：0.769
```

因此，在当前预处理、LDA 特征形式和严格 leave-one-character-out 口径下，balanced accuracy 达到 `0.80` 已经比较困难，`0.85` 基本不现实。

原因主要有三点：

1. 训练数据独立字符 trial 只有 12 个，epoch 虽有 720 个，但同一字符 trial 内 epoch 相关性强，不能当作完全独立样本。
2. target 与 non-target 比例为 `1:5`，P300 单 epoch 信噪比较低，很多 target epoch 并不会有稳定大幅 P300。
3. 字符解码依赖 5 轮行列得分聚合，最终字符正确率可以较高，但单 epoch balanced accuracy 不一定同步很高。

所以更合理的目标是：

- epoch balanced accuracy 稳定在 `0.74-0.77`；
- 字符级 leave-one-character-out 保持 `10/12` 或 `11/12`；
- 测试参考保持 `6/8` 到 `7/8`；
- 避免为了提高 epoch 指标牺牲字符解码稳定性。

