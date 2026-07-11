# P300实验主线

## 文件

- `P300完整实验主线交接版_对话1.ipynb`：完整可执行主线；
- `完整说明README_对话1.md`：详细依据、结果和后续建议；
- 本文件：快速交接说明。

## 流程

```text
原始Excel
→ 0.5–30 Hz零相位带通
→ [-200,800) ms epoch
→ 250→125 Hz多相降采样
→ [-200,0) ms基线校正
→ 100–600 ms特征区间
→ 约25 ms通道均值特征
→ 按字符trial嵌套分组验证
→ Shrinkage LDA
→ 行列得分聚合
→ 字符解码
```

## 当前选择

- 主预处理：`0.5–30 Hz + 125 Hz + [-200,0) ms baseline`；
- P300识别区间：`100–600 ms`；
- 基础模型：Shrinkage LDA；
- 外层验证：Leave-One-Character-Out；
- 主要选择指标：字符准确率；
- 未知测试集：未参与选择。

Shrinkage LDA与Linear SVM均为8/12字符正确；LDA的PR-AUC和平衡准确率更高，因此作为基础模型。时间窗实验中`100–600 ms`为9/12，是下一阶段应在完整嵌套管线中继续验证的候选区间。

## 运行

从仓库根目录执行：

```powershell
pip install -r 全流程/requirements_对话1.txt
jupyter notebook 交接版_对话1/P300完整实验主线交接版_对话1.ipynb
```

后续优化必须继续按完整字符trial分组，不能随机拆分epoch，也不能使用未知测试答案选择方案。
