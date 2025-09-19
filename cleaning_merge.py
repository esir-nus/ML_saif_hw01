import pandas as pd

# Load data (adapt from data_load.py)
lc = pd.read_csv('../P2P_PPDAI_DATA/LC.csv', encoding='utf-8')
lp = pd.read_csv('../P2P_PPDAI_DATA/LP.csv', encoding='utf-8')

# Define required columns
required_cols = ['ListingId', '借款金额', '借款期限', '借款利率', '借款成功日期', '初始评级', '借款类型', '是否首标', '年龄', '性别', '手机认证', '户口认证', '视频认证', '学历认证', '征信认证', '淘宝认证', '历史成功借款次数', '历史成功借款金额', '总待还本金', '历史正常还款期数', '历史逾期还款期数']

# After loading lc
lc = lc[required_cols]

# Convert date
lc['借款成功日期'] = pd.to_datetime(lc['借款成功日期'], errors='coerce')

# Cleaning: Handle missing values
lc = lc.dropna(subset=['ListingId'])  # Drop rows missing key ID
# After dropna
numeric_cols = lc.select_dtypes(include=['number']).columns
lc[numeric_cols] = lc[numeric_cols].fillna(lc[numeric_cols].median())  # Median fill for better imputation

# Filter dates
lc = lc[(lc['借款成功日期'] >= pd.to_datetime('2015-01-01')) & (lc['借款成功日期'] <= pd.to_datetime('2017-01-30'))]

# Enhanced outlier handling
lc = lc[(lc['年龄'] >= 18) & (lc['年龄'] <= 120)]
lc = lc[(lc['借款金额'] > 0) & (lc['借款金额'] <= 1000000)]  # Assuming max 1M
lc = lc[(lc['借款利率'] >= 0) & (lc['借款利率'] <= 50)]  # Reasonable rate range
lc = lc[lc['借款期限'] > 0]  # Positive terms

# After all filters
print("Cleaned LC Columns:", lc.columns.tolist())
print("Cleaned LC shape:", lc.shape)
if lc.shape[0] != 328553:
    print(f"Warning: Expected 328553 rows, got {lc.shape[0]}")

# Export cleaned LC
lc.to_csv('cleaned_LC.csv', index=False, encoding='utf-8')

lp = lp.dropna(subset=['ListingId'])
lp = lp.fillna(0)

# Convert types: Dates in LP
lp['到期日期'] = pd.to_datetime(lp['到期日期'], errors='coerce')
lp['还款日期'] = pd.to_datetime(lp['还款日期'], errors='coerce')

# Load and clean LCIS
lcis = pd.read_csv('../P2P_PPDAI_DATA/LCIS.csv', encoding='utf-8')
lcis = lcis.dropna(subset=['ListingId'])
lcis = lcis.fillna(0)
# Convert dates
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

print("LCIS Agg Shape:", lcis_agg.shape)

# Merge LC and LP first, then with LCIS_agg
df = pd.merge(lc, lp, on='ListingId', how='inner')
df = pd.merge(df, lcis_agg, on='ListingId', how='left').fillna(0)  # Left join to keep all loans, fill missing with 0

# Verify
print("Merged Head:\n", df.head())
print("Merged Shape:", df.shape)
