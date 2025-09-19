import pandas as pd
import datetime
import os

# Load engineered data for summary stats
import pandas as pd

# Compute some simple stats for summary (reuse from earlier logic)
try:
    df = pd.read_csv('engineered_df.csv', encoding='utf-8')
except Exception:
    # Fallback to cleaned file if engineered missing
    df = pd.read_csv('cleaned_LC.csv', encoding='utf-8')

label_col = 'Label' if 'Label' in df.columns else ('还款状态' if '还款状态' in df.columns else None)
if label_col is None:
    label_col = '还款状态'

# 计算总体逾期率（若为 0/1 标签则直接均值；否则按>0 视为逾期）
label_series = df[label_col]
if label_series.dropna().isin([0,1]).all():
    overdue_rate = label_series.mean()
else:
    overdue_rate = (label_series > 0).mean()

# 计算和标签的相关性（仅数值列）
numeric_df = df.select_dtypes(include=['number'])
if label_col in numeric_df.columns:
    corr_with_label = numeric_df.corr()[label_col].sort_values(ascending=False).head(5)
else:
    # 若数值列里没有标签列，则跳过
    corr_with_label = pd.Series(dtype='float64')

# 输出目录与时间戳
os.makedirs('summaries', exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
md_path = f"summaries/summary_{timestamp}.md"
txt_path = f"summaries/summary_{timestamp}.txt"

# 将相关性格式化为表格字符串（去掉 dtype 行）
if not corr_with_label.empty:
    corr_df = corr_with_label.reset_index()
    corr_df.columns = ['特征 (Feature)', '与标签相关性 (Correlation w/ Label)']
    corr_str = corr_df.to_string(index=False)
else:
    corr_str = '（无可计算的数值相关性 / No numeric correlation available）'

# 写入 Markdown（中英双语）
with open(md_path, 'w', encoding='utf-8') as f:
    f.write(f"# 摘要报告 (Summary Report)\n生成于 (Generated on): {timestamp}\n\n")
    f.write("# P2P 借贷分析摘要 (P2P Lending Analysis Summary)\n\n")
    f.write("## 主要发现 (Key Findings)\n")
    f.write(f"- 总体逾期率 (Overall Overdue Rate): {overdue_rate:.4f}\n")
    f.write("- 与标签相关性最高的特征（前五） (Top Correlations with Label - Top 5):\n")
    f.write("```\n" + corr_str + "\n```\n\n")
    f.write("## 经验教训 (Lessons Learned)\n")
    f.write("- 使用类权重等方式处理数据不平衡。 (Handled class imbalance via class weights)\n")
    f.write("- 特征工程（如逾期比率）对提升预测有效。 (Feature engineering like overdue ratios helps)\n")
    f.write("- 与常见基准相比（ROC-AUC 0.7-0.8），本项目具有改进空间。 (Compared to typical benchmarks)\n\n")
    f.write("## 潜在改进 (Potential Improvements)\n")
    f.write("- 进一步的超参数调优与模型对比。 (More hyperparameter tuning and model comparison)\n")
    f.write("- 更严格地避免特征泄露并完善特征选择。 (Stronger leakage prevention and feature selection)\n")

# 写入纯文本（中英双语）
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(f"摘要报告 (Summary Report)\n生成于 (Generated on): {timestamp}\n\n")
    f.write("P2P 借贷分析摘要 (P2P Lending Analysis Summary)\n\n")
    f.write("主要发现 (Key Findings)\n")
    f.write(f"- 总体逾期率 (Overall Overdue Rate): {overdue_rate:.4f}\n")
    f.write("- 与标签相关性最高的特征（前五） (Top Correlations with Label - Top 5):\n")
    f.write(corr_str + "\n\n")
    f.write("经验教训 (Lessons Learned)\n")
    f.write("- 使用类权重等方式处理数据不平衡。 (Handled class imbalance via class weights)\n")
    f.write("- 特征工程（如逾期比率）对提升预测有效。 (Feature engineering like overdue ratios helps)\n")
    f.write("- 与常见基准相比（ROC-AUC 0.7-0.8），本项目具有改进空间。 (Compared to typical benchmarks)\n\n")
    f.write("潜在改进 (Potential Improvements)\n")
    f.write("- 进一步的超参数调优与模型对比。 (More hyperparameter tuning and model comparison)\n")
    f.write("- 更严格地避免特征泄露并完善特征选择。 (Stronger leakage prevention and feature selection)\n")

print(f"摘要已保存至 (Saved to): {md_path} 和 {txt_path}")
