#Step 4 — Model Training, Tuning & Evaluation


import warnings
warnings.filterwarnings("ignore")  

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score,
)

df = pd.read_csv("data/features_readable.csv")
print(f"Loaded: {df.shape}")

categorical_cols = [
    "city", "gender", "plan_type", "complaint_resolved", "network_type",
    "device_brand", "device_price_range", "contract_type",
    "international_roaming_used", "streaming_service_used", "payment_method",
]
numeric_cols = [
    "age", "monthly_income", "tenure_months", "monthly_recharge_amount",
    "data_usage_gb", "call_minutes_used", "sms_count", "number_of_complaints",
    "avg_monthly_bill", "late_payment_count", "customer_service_calls",
    "value_added_services_count", "satisfaction_score",
    "app_usage_hours_per_week", "social_media_usage_hours",
    "number_of_family_lines", "tenure_days_from_signup",
]

skewed_cols = [
    "data_usage_gb", "monthly_income", "social_media_usage_hours",
    "app_usage_hours_per_week", "tenure_months", "late_payment_count",
    "avg_monthly_bill", "monthly_recharge_amount", "value_added_services_count",
    "number_of_family_lines", "number_of_complaints", "customer_service_calls",
]
normal_cols = [c for c in numeric_cols if c not in skewed_cols]


def build_preprocessor():
    """Fresh, UNFITTED preprocessor — must stay unfitted so GridSearchCV
    can fit it separately inside every training fold."""
    skewed_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    normal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("skewed_num", skewed_pipe, skewed_cols),
        ("normal_num", normal_pipe, normal_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


# 1. Train/test split

X = df[numeric_cols + categorical_cols]
y = df["churn_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

# 2. Full pipeline + grid search + stratified CV

pipeline = Pipeline([
    ("preprocess", build_preprocessor()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print(f"\nBest params: {grid.best_params_}")
print(f"Best CV F1 (mean across 5 folds): {grid.best_score_:.3f}")

best_model = grid.best_estimator_

# 3. Evaluate on the held-out test set

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
auc = roc_auc_score(y_test, y_proba)

print("\nConfusion matrix:")
print(cm)
print("\nClassification report:")
print(report)
print(f"ROC-AUC: {auc:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion matrix (test set)")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="#C44E52")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC curve (test set)")
axes[1].legend()
fig.tight_layout()
fig.savefig("plots/model_evaluation.png", dpi=110)
plt.close(fig)

baseline_acc = (y_test == 0).mean()
model_acc = (y_pred == y_test).mean()
print(f"\nBaseline accuracy (always predict 'No Churn'): {baseline_acc:.3f}")
print(f"Model accuracy: {model_acc:.3f}  <- barely different, which is exactly "
      f"why precision/recall/F1/ROC-AUC matter more than accuracy here")


# 4. logistic regression coefficients

feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
coefs = best_model.named_steps["clf"].coef_[0]
coef_series = pd.Series(coefs, index=feature_names).sort_values()
top_coefs = pd.concat([coef_series.head(8), coef_series.tail(8)])

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#4C72B0" if v < 0 else "#C44E52" for v in top_coefs.values]
top_coefs.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Logistic regression coefficients\n(red = pushes toward churn, blue = pushes toward staying)")
ax.set_xlabel("Coefficient (on scaled/encoded features)")
fig.tight_layout()
fig.savefig("plots/logreg_coefficients.png", dpi=110)
plt.close(fig)

joblib.dump(best_model, "artifacts/churn_model_pipeline.joblib")
print("\nSaved full pipeline -> artifacts/churn_model_pipeline.joblib")

# 5. Bonus regression side-task -> a genuine RMSE
#    Predicting avg_monthly_bill from usage behavior (not from the
#    other billing column, to avoid a trivial/leaky prediction).

reg_features = [
    "tenure_months", "data_usage_gb", "call_minutes_used", "sms_count",
    "app_usage_hours_per_week", "social_media_usage_hours",
    "value_added_services_count", "number_of_family_lines",
]
reg_df = df.dropna(subset=["avg_monthly_bill"] + reg_features)
Xr = reg_df[reg_features]
yr = reg_df["avg_monthly_bill"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("reg", LinearRegression()),
])
reg_pipeline.fit(Xr_train, yr_train)
yr_pred = reg_pipeline.predict(Xr_test)

rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
print(f"\nBonus regression task (predicting avg_monthly_bill):")
print(f"RMSE: {rmse:.2f}  |  R2: {r2:.3f}  |  target mean: {yr.mean():.2f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(yr_test, yr_pred, alpha=0.4, s=15, color="#55A868")
lims = [min(yr_test.min(), yr_pred.min()), max(yr_test.max(), yr_pred.max())]
ax.plot(lims, lims, linestyle="--", color="gray")
ax.set_xlabel("Actual avg_monthly_bill")
ax.set_ylabel("Predicted avg_monthly_bill")
ax.set_title(f"Bonus regression task — RMSE={rmse:.1f}, R2={r2:.3f}")
fig.tight_layout()
fig.savefig("plots/bonus_regression.png", dpi=110)
plt.close(fig)

joblib.dump(reg_pipeline, "artifacts/bonus_regression_pipeline.joblib")
print("\nDone. All artifacts saved under artifacts/, plots under plots/.")
