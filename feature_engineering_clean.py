import os
import re
import pandas as pd


def is_leaky(column_name: str) -> bool:
    patterns = [
        r"逾期", r"overdue", r"已还", r"实还", r"剩余", r"结清", r"已结清",
        r"应还", r"回款", r"还款", r"repay", r"paid", r"remaining", r"outstanding",
        r"writeoff", r"chargeoff", r"bad[_-]?debt", r"最后还款日期", r"actual.*(pay|settle)",
    ]
    for pat in patterns:
        if re.search(pat, str(column_name), flags=re.IGNORECASE):
            return True
    return False


def build_clean_features(input_csv: str = 'cleaned_LC.csv', output_csv: str = 'engineered_df_clean.csv') -> str:
    df = pd.read_csv(input_csv, encoding='utf-8')

    # Identify label column (keep it if present for later evaluation)
    label_col = 'Label' if 'Label' in df.columns else ('还款状态' if '还款状态' in df.columns else None)

    # Remove leaky columns by keyword (except label)
    drop_cols = []
    for c in df.columns:
        if c == label_col:
            continue
        if is_leaky(c):
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Application-time safe whitelist (keep if exists)
    safe_candidates = [
        'ListingId', '借款金额', '借款期限', '借款利率', '借款成功日期', '初始评级', '借款类型', '是否首标',
        '年龄', '性别', '手机认证', '户口认证', '视频认证', '学历认证', '征信认证', '淘宝认证',
        '历史成功借款次数', '历史成功借款金额'
    ]
    existing_safe = [c for c in safe_candidates if c in df.columns]
    keep_cols = set(existing_safe + ([label_col] if label_col and label_col in df.columns else []))
    if keep_cols:
        df = df[[c for c in df.columns if c in keep_cols]]

    # Basic cleaning: coerce date, impute numerics with median
    if '借款成功日期' in df.columns:
        df['借款成功日期'] = pd.to_datetime(df['借款成功日期'], errors='coerce')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Leave categoricals as is for evaluation pipeline to one-hot
    df.to_csv(output_csv, index=False, encoding='utf-8')
    return os.path.abspath(output_csv)


if __name__ == '__main__':
    path = build_clean_features()
    print(f"Clean engineered file saved to: {path}")


