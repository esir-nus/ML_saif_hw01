import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import seaborn as sns
import os
import re
from typing import List, Tuple

# Load engineered data (prefer clean version if present)
data_path = 'engineered_df_clean.csv' if os.path.exists('engineered_df_clean.csv') else 'engineered_df.csv'
df = pd.read_csv(data_path, encoding='utf-8')

# Sample for memory efficiency on large data
if len(df) > 100000:
    df = df.sample(100000, random_state=42)
    print("已采样 100k 行以优化内存")

# Decide label column first
label_col = 'Label' if 'Label' in df.columns else ('还款状态' if '还款状态' in df.columns else None)
if label_col is None:
    # Try to merge label from engineered_df.csv by ListingId
    if 'ListingId' in df.columns and os.path.exists('engineered_df.csv'):
        try:
            df_lab = pd.read_csv('engineered_df.csv', encoding='utf-8', usecols=['ListingId', 'Label'])
            label_col = 'Label'
        except Exception:
            try:
                df_lab = pd.read_csv('engineered_df.csv', encoding='utf-8', usecols=['ListingId', '还款状态'])
                label_col = '还款状态'
            except Exception:
                df_lab = None
        if df_lab is not None:
            before = df.shape[0]
            df = df.merge(df_lab, on='ListingId', how='inner')
            after = df.shape[0]
            print(f"已从 engineered_df.csv 合并标签列 '{label_col}'，样本数: {before} -> {after}")
        else:
            raise KeyError("未找到标签列：应包含 'Label' 或 '还款状态'")
    else:
        raise KeyError("未找到标签列：应包含 'Label' 或 '还款状态'")

# Normalize/binarize label
df[label_col] = df[label_col].apply(lambda x: 1 if x > 0 else 0)

# Leakage filtering by multilingual keyword patterns (never drop the label)
base_leaky_cols = [
    'payment_ratio', 'paid_principal', 'total_overdue_periods',
    '应还本金', '已还本金', '剩余本金', '应还利息', '已还利息', '剩余利息', '还款状态'
]

leaky_patterns = [
    r"逾期", r"overdue", r"已还", r"实还", r"剩余", r"结清", r"已结清",
    r"应还", r"回款", r"还款", r"repay", r"paid", r"remaining", r"outstanding",
    r"writeoff", r"chargeoff", r"bad[_-]?debt"
]

candidate_cols = set(base_leaky_cols)
for c in df.columns:
    if c == label_col:
        continue
    for pat in leaky_patterns:
        if re.search(pat, str(c), flags=re.IGNORECASE):
            candidate_cols.add(c)
            break

to_drop = [c for c in candidate_cols if c in df.columns and c != label_col]
if len(to_drop) > 0:
    df = df.drop(columns=to_drop)
    print(f"已移除潜在泄露特征以确保公平评估: {len(to_drop)} 列")
else:
    print("未发现需要移除的潜在泄露特征（基于关键词）")

# Identify possible time and group columns
time_cols_candidates = [
    '借款成功日期', 'issue_date', 'IssueDate', 'ListingDate', '放款日期', '放款时间'
]
group_cols_candidates = [
    'user_id', '用户ID', 'BorrowerId', '借款人ID', 'CustomerId'
]

time_col = next((c for c in time_cols_candidates if c in df.columns), None)
group_col = next((c for c in group_cols_candidates if c in df.columns), None)

# Parse datetime if available
if time_col is not None:
    try:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    except Exception:
        time_col = None

X = df.drop(columns=[label_col])
y = df[label_col]

# Train/validation/test split: prefer time-based, else group-based, else stratified random
def split_data(X_df: pd.DataFrame, y_s: pd.Series):
    if time_col is not None and X_df[time_col].notna().sum() > 0:
        cutoff = X_df[time_col].quantile(0.8)
        train_idx = X_df[time_col] <= cutoff
        test_idx = ~train_idx
        return X_df[train_idx], X_df[test_idx], y_s[train_idx], y_s[test_idx], 'time'
    if group_col is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        groups = X_df[group_col]
        train_idx, test_idx = next(splitter.split(X_df, y_s, groups))
        return X_df.iloc[train_idx], X_df.iloc[test_idx], y_s.iloc[train_idx], y_s.iloc[test_idx], 'group'
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.3, random_state=42, stratify=y_s
    )
    return X_train, X_test, y_train, y_test, 'random'

X_train, X_test, y_train, y_test, split_mode = split_data(X, y)
print(f"使用切分方式: {split_mode}")

# Enforce/inspect group exclusivity if group column exists
if group_col is not None:
    try:
        train_groups = set(X_train[group_col].dropna().astype(str))
        test_groups = set(X_test[group_col].dropna().astype(str))
        overlap = train_groups.intersection(test_groups)
        if len(overlap) > 0:
            print(f"警告: 发现 {len(overlap)} 个主体在训练与测试集同时出现，可能存在信息泄露。")
        else:
            print("主体分割检查通过：训练/测试主体无重叠。")
    except Exception:
        pass

# Build preprocessing pipeline (fit on train only)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(with_mean=False))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

class_counts = np.bincount(y_train)
pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1.0

model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    ))
])

model.fit(X_train, y_train)

# Predictions and probabilities
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# Compute metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)

print("模型评估结果（采样后） (Evaluation on Sampled Data)")
print("----------------------")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数 (F1-Score): {f1:.4f}")
print(f"ROC 曲线下面积 (ROC-AUC): {roc_auc:.4f}")

# PR-AUC and threshold sweep
pr_auc = average_precision_score(y_test, probs)
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

# Threshold sweep to report best-F1 and a recall-oriented point
def metrics_at_threshold(p: np.ndarray, y_true: np.ndarray, thr: float):
    pred = (p >= thr).astype(int)
    return (
        precision_score(y_true, pred, zero_division=0),
        recall_score(y_true, pred, zero_division=0),
        f1_score(y_true, pred, zero_division=0),
        accuracy_score(y_true, pred),
    )

candidate_thresholds = np.unique(np.quantile(probs, np.linspace(0.01, 0.99, 99)))
best_f1, best_thr, best_tuple = -1.0, 0.5, (precision, recall, f1, accuracy)
for thr in candidate_thresholds:
    prc, rcl, f1c, accc = metrics_at_threshold(probs, y_test, thr)
    if f1c > best_f1:
        best_f1, best_thr, best_tuple = f1c, thr, (prc, rcl, f1c, accc)
print(f"最佳F1阈值 (Best-F1 Threshold): {best_thr:.4f}, P/R/F1/Acc = {best_tuple}")

# Plot PR curve and threshold curve
os.makedirs('summaries', exist_ok=True)
plt.figure()
plt.plot(recalls, precisions, label=f'PR (AP={pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('summaries/pr_curve.png')
plt.close()

plt.figure()
thr_list = candidate_thresholds
f1_list, r_list, p_list = [], [], []
for thr in thr_list:
    p_, r_, f1_, _ = metrics_at_threshold(probs, y_test, thr)
    f1_list.append(f1_)
    r_list.append(r_)
    p_list.append(p_)
plt.plot(thr_list, f1_list, label='F1')
plt.plot(thr_list, p_list, label='Precision')
plt.plot(thr_list, r_list, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold vs Metrics')
plt.legend()
plt.savefig('summaries/threshold_curve.png')
plt.close()

# ================== Leakage Diagnostics ==================
def leakage_diagnostics(X_df: pd.DataFrame, y_series: pd.Series) -> Tuple[List[str], List[str]]:
    flags_corr: List[str] = []
    flags_auc: List[str] = []

    # 1) Extreme correlation/equality checks
    try:
        numeric = X_df.select_dtypes(include=[np.number])
        if not numeric.empty:
            corr_with_y = numeric.corrwith(y_series)
            corr_sorted = corr_with_y.abs().sort_values(ascending=False)
            for col, val in corr_sorted.items():
                if np.isfinite(val) and val >= 0.99:
                    flags_corr.append(f"[CORR>=0.99] {col}: corr={val:.4f}")
    except Exception:
        pass

    for col in X_df.columns:
        try:
            same_ratio = (pd.Series(X_df[col]).astype(str) == y_series.astype(str)).mean()
            if same_ratio >= 0.99:
                flags_corr.append(f"[EQUAL>=0.99] {col}: same={same_ratio:.4f}")
        except Exception:
            continue

    # 2) Single-feature AUC sweep (factorize objects)
    from sklearn.metrics import roc_auc_score
    for col in X_df.columns:
        try:
            series = X_df[col]
            if series.nunique(dropna=True) <= 1:
                continue
            if series.dtype == 'object':
                series = pd.Series(pd.factorize(series)[0], index=series.index)
            auc = roc_auc_score(y_series, series)
            if auc >= 0.99 or (1 - auc) >= 0.99:
                flags_auc.append(f"[SINGLE_AUC>=0.99] {col}: AUC={auc:.4f}")
        except Exception:
            continue

    return flags_corr, flags_auc

flags_corr, flags_auc = leakage_diagnostics(X_train, y_train)

os.makedirs('summaries', exist_ok=True)
report_path = os.path.join('summaries', 'leakage_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("潜在泄露诊断报告 (Leakage Diagnostics)\n")
    f.write("====================================\n\n")
    f.write(f"切分方式: {split_mode}\n")
    if group_col is not None:
        f.write(f"分组字段: {group_col}\n")
    if time_col is not None:
        f.write(f"时间字段: {time_col}\n")
    f.write("\n[1] 极端相关/相等检查 (训练集)\n")
    if flags_corr:
        for line in flags_corr:
            f.write(line + "\n")
    else:
        f.write("无明显极端相关/相等特征。\n")
    f.write("\n[2] 单特征AUC扫描 (训练集)\n")
    if flags_auc:
        for line in flags_auc:
            f.write(line + "\n")
    else:
        f.write("未发现单特征即可近乎完美预测的列。\n")

print(f"泄露诊断报告已保存: {report_path}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, probs)
os.makedirs('summaries', exist_ok=True)
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('summaries/roc.png')
plt.close()

# Add confusion matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('summaries/confusion_matrix.png')
plt.close()

# Cross-validation (5-fold) consistent with split mode
if split_mode == 'group' and group_col is not None:
    cv = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', groups=df[group_col])
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"交叉验证准确率均值 (CV Accuracy Mean): {cv_scores.mean():.4f} (标准差 Std: {cv_scores.std():.4f})")
print("图表已保存 (Saved): summaries/roc.png, summaries/confusion_matrix.png, summaries/pr_curve.png, summaries/threshold_curve.png")
