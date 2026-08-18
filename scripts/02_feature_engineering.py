#Step 2 — Feature Engineering


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

os.makedirs("plots", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

df = pd.read_csv("data/cleaned.csv")
print(f"Loaded cleaned data: {df.shape}")

# 1. Drop columns that aren't model features

df = df.drop(columns=["customer_id", "signup_date"])

# 2. One more outlier fix that Step 1 left for here: tenure_months

before = df["tenure_months"].notna().sum()
df.loc[(df["tenure_months"] < 0) | (df["tenure_months"] > 96), "tenure_months"] = np.nan
after = df["tenure_months"].notna().sum()
print(f"tenure_months: {before - after} out-of-range values (<0 or >96) set to NaN")

# 3. Drop rows with no target

before_rows = len(df)
df = df.dropna(subset=["churn"]).reset_index(drop=True)
print(f"Dropped {before_rows - len(df)} row(s) with missing churn label -> {len(df)} rows remain")

df["churn_binary"] = (df["churn"] == "Yes").astype(int)

# 4. Column groups

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

# 5. Check skewness

skew_before = df[numeric_cols].skew(numeric_only=True).sort_values(ascending=False)
print("\nSkewness per numeric column (before any transform):")
print(skew_before)

SKEW_THRESHOLD = 0.75
skewed_cols = skew_before[skew_before.abs() > SKEW_THRESHOLD].index.tolist()
normal_cols = [c for c in numeric_cols if c not in skewed_cols]
print(f"\n{len(skewed_cols)} columns flagged as skewed (|skew| > {SKEW_THRESHOLD}): {skewed_cols}")

demo_cols = skew_before.index[:3].tolist()
fig, axes = plt.subplots(2, len(demo_cols), figsize=(4 * len(demo_cols), 6))
for i, col in enumerate(demo_cols):
    vals = df[col].dropna()
    axes[0, i].hist(vals, bins=40, color="#4C72B0")
    axes[0, i].set_title(f"{col}\nbefore (skew={vals.skew():.2f})")
    log_vals = np.log1p(vals)
    axes[1, i].hist(log_vals, bins=40, color="#55A868")
    axes[1, i].set_title(f"{col}\nafter log1p (skew={log_vals.skew():.2f})")
fig.tight_layout()
fig.savefig("plots/skew_before_after.png", dpi=110)
plt.close(fig)

# 6. Build the preprocessing pipeline

skewed_numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale", StandardScaler()),
])

normal_numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("skewed_num", skewed_numeric_pipeline, skewed_cols),
    ("normal_num", normal_numeric_pipeline, normal_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

X = df[numeric_cols + categorical_cols]
y = df["churn_binary"]

X_transformed = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
print(f"\nTransformed feature matrix shape: {X_transformed.shape} "
      f"(started from {len(numeric_cols) + len(categorical_cols)} raw columns, "
      f"ended with {len(feature_names)} after one-hot encoding)")

joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

features_matrix = pd.DataFrame(X_transformed, columns=feature_names)
features_matrix["churn"] = y.values
features_matrix.to_csv("data/features_matrix.csv", index=False)

df.to_csv("data/features_readable.csv", index=False)

# 7. Feature selection

mi_X = df[numeric_cols].copy()
for col in numeric_cols:
    mi_X[col] = mi_X[col].fillna(mi_X[col].median())

le_cache = {}
for col in categorical_cols:
    le = LabelEncoder()
    filled = df[col].fillna("Unknown")
    mi_X[col] = le.fit_transform(filled)
    le_cache[col] = le

mi_scores = mutual_info_classif(mi_X, y, discrete_features=[c in categorical_cols for c in mi_X.columns], random_state=42)
mi_ranking = pd.Series(mi_scores, index=mi_X.columns).sort_values(ascending=False)
print("\nFeature importance (mutual information with churn), top 15:")
print(mi_ranking.head(15))

fig, ax = plt.subplots(figsize=(7, 6))
mi_ranking.head(15).sort_values().plot(kind="barh", ax=ax, color="#C44E52")
ax.set_title("Top 15 features by mutual information with churn")
ax.set_xlabel("Mutual information score")
fig.tight_layout()
fig.savefig("plots/feature_importance_mi.png", dpi=110)
plt.close(fig)

# 8. A few more EDA visuals

corr_df = df[numeric_cols + ["churn_binary"]].copy()
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
ax.set_title("Correlation heatmap — numeric features + churn")
fig.tight_layout()
fig.savefig("plots/correlation_heatmap.png", dpi=110)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
churn_by_contract = df.groupby("contract_type")["churn_binary"].mean().sort_values(ascending=False)
churn_by_contract.plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("Churn rate by contract type")
axes[0].set_ylabel("Churn rate")
axes[0].tick_params(axis="x", rotation=30)

churn_by_city = df.groupby("city")["churn_binary"].mean().sort_values(ascending=False)
churn_by_city.plot(kind="bar", ax=axes[1], color="#DD8452")
axes[1].set_title("Churn rate by city")
axes[1].set_ylabel("Churn rate")
axes[1].tick_params(axis="x", rotation=60)
fig.tight_layout()
fig.savefig("plots/churn_rate_by_category.png", dpi=110)
plt.close(fig)

print("\nSaved: data/features_readable.csv, data/features_matrix.csv, "
      "artifacts/preprocessor.joblib, plots/*.png")
