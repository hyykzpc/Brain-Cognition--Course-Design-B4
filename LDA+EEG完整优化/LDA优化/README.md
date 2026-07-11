# LDA 优化实验

本目录保留 P300 字符识别任务中的 LDA 优化实验代码、记录和结果。

## 运行

从仓库根目录执行：

```powershell
python ".\LDA优化\run_lda_pipeline.py"
```

脚本读取预处理产物：

- `全流程/outputs/02_preprocessing/data/epochs_main_125Hz_B200_*.npz`
- `全流程/outputs/02_preprocessing/tables/epoch_metadata_*.csv`

主要输出：

- `LDA优化/outputs/tables/`
- `LDA优化/outputs/figures/`

## 当前 LDA 配置

- 刺激区间：`0-500 ms`
- 特征：20 通道 x 8 个时间分箱
- LDA：`solver="lsqr"`，`shrinkage="auto"`，类别先验 `[0.5, 0.5]`
- 验证：leave-one-character-out，按完整字符 trial 分组

## 当前结果

- 训练字符交叉验证：`11/12 = 0.917`
- Unknown 测试集盲选结果：`6/8 = 0.750`
- 后验备用方案：`0-500 ms`、`10` 分箱、`shrinkage=0.03`，测试集 `7/8 = 0.875`

## 结果记录

- 时间窗口实验记录：`时间窗口实验记录.md`
- 通道选择实验记录：`通道选择实验记录.md`
- LDA 进一步优化探索：`LDA进一步优化探索.md`
- 最佳 LDA 配置记录：`最佳LDA配置记录.md`


