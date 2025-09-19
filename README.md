# P2P 逾期预测课程作业 / P2P Overdue Prediction Coursework

本仓库为课程作业项目，目标是在“申请时点可得”信息下预测是否会逾期；项目重点在于正确的数据处理流程、避免数据泄露、以及用合理指标评估模型。

This repo contains a coursework project for predicting loan overdue using only application-time features. The focus is on leakage-free data processing and sound evaluation.

## 快速开始 / Quick Start

1) 环境 / Environment
- 建议 Anaconda，Python 3.9+；示例：`conda activate ML_p2p`
- 需安装：pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

2) 数据准备 / Data
- 原始数据位于 `P2P_PPDAI_DATA/`，清洗后样本在 `cleaned_LC.csv`。

3) 生成干净特征 / Build clean features
```
python feature_engineering_clean.py
```
输出 `engineered_df_clean.csv`（仅包含申请时点可用字段，自动剔除含“逾期/还款/结清/回款”等关键词的泄露列）。

4) 模型评估 / Evaluation
```
python evaluation.py
```
脚本会优先读取 `engineered_df_clean.csv`；若缺少标签，会从 `engineered_df.csv` 通过 `ListingId` 合并。输出：
- 主要指标：Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- 图表：`summaries/roc.png`, `summaries/confusion_matrix.png`, `summaries/pr_curve.png`, `summaries/threshold_curve.png`
- 泄露诊断：`summaries/leakage_report.txt`

5) GUI（推荐入口） / Recommended Entry
```
python gui.py
```
零代码演示：一键加载数据、训练示例模型、运行评估并在界面查看输出与图表，适合助教/评审快速复现。

## 项目结构 / Project Structure
- `cleaned_LC.csv` 清洗后的基础数据 / cleaned base data
- `feature_engineering_clean.py` 生成无泄露特征 / leakage-free feature builder
- `engineered_df_clean.csv` 干净特征结果 / clean engineered dataset
- `evaluation.py` 评估与可视化（含泄露诊断与阈值扫描）/ evaluation with diagnostics
- `gui.py` 演示用界面 / demo GUI
- `summaries/` 评估与摘要输出 / figures and reports
- `log.md` 重要变更与排查记录 / change log and investigations

## 脚本说明（含开发过程）/ Scripts Overview (incl. dev process)
- `exploratory.py` 开发探索脚本：用于原始字段理解、基本统计与可视化（非评估用）。
  - Reflects early EDA of raw fields; kept for provenance only.
- `cleaning_merge.py` 清洗合并流程的早期脚本（已由 clean 流程替代，仅供参考）。
- `data_load.py` 数据载入与基础预处理示例（演示用途）。
- `feature_engineering_clean.py` 生产提交所用“申请时点安全”特征（正式使用）。
- `evaluation.py` 正式评估脚本（ROC/PR、阈值扫描、泄露诊断、时间/分组切分）。
- `summarize.py` 生成文字版摘要到 `summaries/`，用于报告附录。
- `gui.py` 交互演示界面（推荐优先使用，适合零代码复现）。
- `full_script_verification.md`/`_cn.md` 全流程核对清单，用于复现。

## 交付物与评分关注点 / Deliverables & Grading Focus
- 代码可运行：`feature_engineering_clean.py` 与 `evaluation.py` 可直接复现结果。
- 输出材料：`summaries/` 内的 ROC、PR、混淆矩阵、阈值曲线与 `leakage_report.txt`。
- 文档说明：`README.md` 与 `log.md` 记录泄露修复前后对比与关键决策。
- 可选演示：`gui.py`。

## 重要说明 / Notes
- 本项目已修复“事后信息”导致的过高指标（例如 `还款日期` 等）。修复后 AUC≈0.57（示例），更符合实际难度与类别不平衡情况。
- 评估中提供 PR‑AUC 与阈值扫描，用于选择更贴合业务目标的阈值（而非固定 0.5）。

This project explicitly removes post‑event information to avoid target leakage. After fixes, metrics reflect realistic difficulty and class imbalance. PR‑AUC and threshold sweep are provided to support business‑oriented thresholding.

## 引用与致谢 / Acknowledgements
- 数据字段参考 `LC/LP/LCIS` 表结构与课程提供资料。感谢课程助教与同学的反馈。

