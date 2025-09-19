import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
lc = pd.read_csv('cleaned_LC.csv', encoding='utf-8')
lp = pd.read_csv('../P2P_PPDAI_DATA/LP.csv', encoding='utf-8')
print("LP Head:\n", lp.head())
print("LP Columns:", lp.columns.tolist())

# Load and clean LCIS (added for consistency)
lcis = pd.read_csv('../P2P_PPDAI_DATA/LCIS.csv', encoding='utf-8')
lcis = lcis.dropna(subset=['ListingId']).fillna(0)
lcis['借款成功日期'] = pd.to_datetime(lcis['借款成功日期'], errors='coerce')
lcis['上次还款日期'] = pd.to_datetime(lcis['上次还款日期'], errors='coerce')
lcis['下次计划还款日期'] = pd.to_datetime(lcis['下次计划还款日期'], errors='coerce')
lcis['recorddate'] = pd.to_datetime(lcis['recorddate'], errors='coerce')

# Aggregate LCIS per ListingId
lcis_agg = lcis.groupby('ListingId').agg(
    num_investments=('ListingId', 'count'),
    total_invested=('我的投资金额', 'sum'),
    avg_overdue_days=('标当前逾期天数', 'mean')
).reset_index()

# Quick clean LP (subset)
lp = lp.dropna(subset=['ListingId']).fillna(0)
lp['到期日期'] = pd.to_datetime(lp['到期日期'], errors='coerce')
lp['还款日期'] = pd.to_datetime(lp['还款日期'], errors='coerce')

# Merge with cleaned LC
df = pd.merge(lc, lp, on='ListingId', how='inner')
df = pd.merge(df, lcis_agg, on='ListingId', how='left').fillna(0)
print("DF Columns:", df.columns.tolist())

label_col = 'Label' if 'Label' in df.columns else '还款状态'

# Overdue stats
print("Overdue Stats:\n", df[label_col].value_counts(normalize=True))

# Plot distributions (e.g., loan amounts)
sns.histplot(df['金额'])
plt.title('Loan Amount Distribution')
plt.savefig('distribution.png')
plt.close()

# Overdue breakdown
sns.countplot(x=label_col, data=df)
plt.title('Overdue Breakdown')
plt.savefig('overdue_breakdown.png')
plt.close()

# Correlations (numeric only)
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=False)
plt.title('Correlation Heatmap')
plt.savefig('correlations.png')
plt.close()

# Grouped analysis (e.g., mean amount by overdue)
print("Mean Amount by Label:\n", df.groupby('还款状态')['金额'].mean())

# Additional stats for LCIS features
print("Num Investments Stats:\n", df['num_investments'].describe())
sns.histplot(df['num_investments'])
plt.title('Number of Investments Distribution')
plt.savefig('num_investments_dist.png')
plt.close()
