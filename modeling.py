import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Function to get model by type
def get_model(model_type='xgboost', class_weights=None):
    if model_type == 'xgboost':
        return XGBClassifier(scale_pos_weight=class_weights[1] if class_weights is not None else 1, random_state=42)
    elif model_type == 'decision_tree':
        return DecisionTreeClassifier(class_weight='balanced', random_state=42)
    elif model_type == 'random_forest':
        return RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Load engineered data
print("Loading data...")
df = pd.read_csv('engineered_df.csv', encoding='utf-8')
print(f"Data loaded: {df.shape} rows/columns")

# Decide label column first
label_col = 'Label' if 'Label' in df.columns else ('还款状态' if '还款状态' in df.columns else None)
if label_col is None:
    raise KeyError("未找到标签列：应包含 'Label' 或 '还款状态'")

# Drop potential leaky features (never drop label)
leaky_cols = ['payment_ratio', 'paid_principal', 'total_overdue_periods', '应还本金', '已还本金', '剩余本金', '应还利息', '已还利息', '剩余利息', '还款状态']
leaky_cols = [c for c in leaky_cols if c != label_col]
df = df.drop(columns=[col for col in leaky_cols if col in df.columns])
print("已移除潜在泄露特征")

# Prepare data: Assume 'Label' is the target (0/1 for non-overdue/overdue); adjust if named differently
print("Preparing features and labels...")
df[label_col] = df[label_col].apply(lambda x: 1 if x > 0 else 0)  # Binarize if needed

# Drop non-numeric/categorical handling: One-hot encode categoricals
categoricals = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categoricals, drop_first=True)

X = df.drop(columns=[label_col])
y = df[label_col]
print(f"Features: {X.shape}, Labels: {y.shape}")

# Handle class imbalance: Compute weights
class_weights = len(y) / (2 * np.bincount(y))  # Balanced weights for binary classes
print(f"Class weights: {class_weights}")

# Split data (70/15/15: train/val/test)
print("Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)  # 50% of 30% = 15% overall for test/val
print(f"Splits: Train {X_train.shape[0]}, Val {X_val.shape[0]}, Test {X_test.shape[0]}")

# Train XGBoost (with class weights and validation set for early stopping)
try:
    print("Initializing and training model...")
    model = get_model('xgboost', class_weights)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)  # Removed early_stopping_rounds
    print("Model trained successfully")

    # Basic evaluation on test set
    print("Evaluating on test set...")
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Cross-validation for robustness (5-fold) on full data
    print("Running cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy Mean: {cv_scores.mean():.4f} (Std: {cv_scores.std():.4f})")
except Exception as e:
    import traceback
    error_msg = f"Modeling failed: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    raise
