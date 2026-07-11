# LDA + EEGNet 完整优化整理版

本目录汇总当前 P300 字符识别任务中已经优化过的 LDA 与 EEGNet 结果，按模型分开存放。

## 目录结构

- `LDA优化/`：优化后的 LDA 代码、实验记录、图表和表格结果。
- `EEGNet_0_500_top3/`：优化后的 EEGNet 0-500 ms、top3 字符聚合代码、最终模型与预测文件。
- `LDA参数优化对比.md`：同一 LDA 模型在不同窗口、分箱、shrinkage、通道策略和联合参数下的结果对比。
- `EEGNet参数优化对比.md`：同一 EEGNet 模型在不同窗口、聚合方式、正类权重和 scheduler 下的结果对比。
- `figures/`：参数对比图和单字符 12 通道响应示例图。
- `tables/`：从现有实验结果整理出的参数对比 CSV。

## LDA 结果入口

- 运行脚本：`LDA优化/run_lda_pipeline.py`
- 总结文件：`LDA优化/README.md`
- 最佳配置记录：`LDA优化/最佳LDA配置记录.md`
- 结果汇总：`LDA优化/outputs/tables/summary.json`
- 模型汇总：`LDA优化/outputs/tables/model_summary.csv`

当前 LDA 盲选配置：

- 时间窗口：`0-500 ms`
- 分箱：`8`
- shrinkage：`auto`
- 训练字符交叉验证：`11/12 = 0.917`
- Unknown 测试集盲选结果：`6/8 = 0.750`
- 后验备用方案：`10` 分箱、`shrinkage=0.03`，Unknown 测试集 `7/8 = 0.875`

## EEGNet 结果入口

- 运行脚本：`EEGNet_0_500_top3/src/p300_pipeline.py`
- 辅助脚本：`EEGNet_0_500_top3/src/ensemble_predict.py`、`EEGNet_0_500_top3/src/ensemble_validate.py`
- 实验报告：`EEGNet_0_500_top3/experiment_report.md`
- 最佳配置：`EEGNet_0_500_top3/selected_config.json`
- 最终模型：`EEGNet_0_500_top3/model.pkl`
- 验证结果：`EEGNet_0_500_top3/validation_results.csv`
- Unknown 预测：`EEGNet_0_500_top3/test_predictions.csv`
- 测试事件得分：`EEGNet_0_500_top3/test_event_scores.csv`

从本目录复现 EEGNet 当前方案：

```powershell
python ".\EEGNet_0_500_top3\src\p300_pipeline.py" --data-dir "..\..\..\大作业\P300-S1" --output-dir ".\EEGNet_0_500_top3" --models eegnet --epoch-ms 0-500 --aggregate-method top3
```

当前 EEGNet 配置：

- 时间窗口：`0-500 ms`
- 字符聚合：`top3`
- 训练字符交叉验证：`10/12 = 0.833`
- 事件级 balanced accuracy：`0.760`
- 事件级 ROC-AUC：`0.824`

## 新增对比图

- 参数对比图：`figures/model_parameter_comparison.png`
- 单字符 Top-12 EEG 通道响应示例：`figures/example_char_top12_channels.png`

## 来源

- LDA 来源：`F:\university\大三课程\脑与认知\zky\Brain-Cognition--Course-Design-B4\LDA优化`
- EEGNet 结果来源：`F:\university\大三课程\脑与认知\大作业\outputs_eegnet_0_500_top3`
- EEGNet 代码来源：`F:\university\大三课程\脑与认知\大作业\src`


## 0-500 ms 参数对齐

- 总表：`模型参数对齐对比.md`
- 对齐 CSV：`tables/aligned_0_500_model_parameter_comparison.csv`
- 300ms 附近通道均值对比图：`figures/top12_vs_all20_mean_around_300ms.png`

