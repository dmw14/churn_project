# Mobile Churn Prediction — Written Results Summary
### Maharashtra Cities Dataset | End-to-End ML Project (Week 8 Deliverable)

---

## 1. Problem Statement

The goal was to predict whether a mobile customer will churn (`Yes`/`No`), using
a deliberately messy, realistic dataset of 1,515 customers across Maharashtra
cities, and to expose the final model as a FastAPI endpoint for real-time
predictions. The project follows a full pipeline: data cleaning → feature
engineering → unsupervised exploration (PCA + clustering) → supervised
modeling (Logistic Regression) → tuning → evaluation → deployment.

## 2. Dataset Overview

- **1,515 rows, 30 columns** — 17 numeric, 12 categorical, 1 target (`churn`)
- **Target imbalance:** ~82% No / ~18% Yes — this shapes every modeling and
  evaluation decision made later in the project
- Deliberately injected messiness: inconsistent city spellings, mixed data
  types (age/income stored as text), 4 different date formats in one column,
  outliers, and various null-like tokens

## 3. Data Cleaning

| Issue | Action taken |
|---|---|
| 14 exact duplicate rows | Dropped |
| City written 30+ inconsistent ways (`Mumbai`/`MUMBAI`/`mumbai`/`Bombay`...) | Mapped to 12 canonical city names |
| `age`, `monthly_income` stored as text (`"52 yrs"`, `"Rs.14200.0"`) | Stripped units, cast to numeric |
| Impossible values (age ≤0 or >100; income ≤0 or >₹500k) | 5 impossible ages and 5 income outliers set to NaN (not guessed at) |
| `signup_date` mixed across 4 formats | Parsed with format-specific regex + range-based disambiguation for ambiguous `DD/MM` vs `MM/DD` cases; only the 1 near-empty "junk" row failed to parse |
| `complaint_resolved` — 433 "missing" values | 429 of them lined up exactly with `number_of_complaints == 0`; relabeled as `"No Complaint Filed"` rather than left as missing |
| Various null-like tokens (`"-"`, `"?"`, `"N/A"`) across categorical columns | Standardized to real `NaN` |

**Note on an assumption made:** for `/`-separated signup dates where both
day and month could be ≤12, the format is genuinely ambiguous. I defaulted
to `DD/MM/YYYY` (the Indian convention) in those cases — a documented
assumption, not a guarantee.

## 4. Feature Engineering

- **Additional outlier fix caught here:** `tenure_months` had values from
  -3 to 500; 5 rows outside the valid 0–96 range were set to NaN.
- **Skewness:** checked skew on all 17 numeric columns; 12 exceeded the
  |skew| > 0.75 threshold and got a `log1p` transform before scaling
  (e.g. `data_usage_gb` skew dropped from 2.40 → 0.02, `monthly_income`
  from 2.16 → 0.07). The remaining 5 (age, sms_count, call_minutes_used,
  satisfaction_score, tenure_days_from_signup) were scaled only.
- **Imputation:** median for numeric columns, `"Unknown"` constant for
  categorical columns — done inside the pipeline so it's reproducible and
  reusable at prediction time.
- **Encoding:** one-hot encoding across all 11 categorical columns —
  28 raw feature columns expanded to 71 after encoding.
- **Rows dropped:** 1 row with no `churn` label (the same near-empty junk
  row) — can't train or evaluate on an unlabeled example. Final modeling
  dataset: **1,500 rows**.
- **Feature selection (mutual information vs. churn):** `tenure_months`
  and `satisfaction_score` dominate by a wide margin, followed by
  `monthly_income`, `app_usage_hours_per_week`, and `avg_monthly_bill`.
  This matches what the data dictionary hinted the real signal should be.
- **Category-level insight:** churn rate is 19% for `No Contract`
  customers vs. 16% for `With Contract`; by city it ranges from ~24%
  (Nagpur) down to ~11% (Amravati).

## 5. Dimensionality Reduction & Clustering

- **PCA:** 36 of the 71 encoded features are needed to capture 95% of the
  variance — meaningful redundancy, but not extreme. A 2-component version
  (built purely for visualization) captures only 14% of variance.
- **K-means:** tested k = 2 through 8, using both the elbow method and
  silhouette score to choose k. Silhouette scores were **very low across
  the board** — the best was only 0.05, at k=2, and it got worse as k
  increased. There was no clear elbow either.
- **Honest finding:** the 2 clusters that resulted split customers mainly
  by `avg_monthly_bill` (₹337 vs. ₹118 average) — essentially a
  "spends more / spends less" divide — but when plotted against actual
  churn, churned customers were scattered evenly across both clusters
  with no visible pattern. **This dataset does not have natural,
  well-separated customer segments**, and churn risk isn't tied to
  usage-behavior clustering here. This is a legitimate result worth
  reporting as-is, and it's consistent with the data dictionary's note
  that churn is only weakly correlated with the underlying signal, plus
  noise.

## 6. Model Training & Tuning

- **Model:** Logistic Regression (`class_weight="balanced"` to counter
  the 82/18 imbalance), wrapped in one end-to-end sklearn `Pipeline`
  together with the preprocessing steps — so preprocessing is refit
  inside every cross-validation fold, with no leakage between folds.
- **Validation:** 5-fold `StratifiedKFold` (keeps the 82/18 churn ratio
  consistent in every fold — important given the imbalance).
- **Tuning:** `GridSearchCV` over `C ∈ {0.01, 0.1, 1, 10, 100}` and
  `penalty ∈ {l1, l2}`, scored on F1 (not accuracy — see below for why).
- **Best parameters found:** `C=0.1, penalty=l1, solver=liblinear`
- **Best cross-validated F1:** 0.448
- **Train/test split:** 80/20, stratified — 1,200 training rows, 300 held
  out for final testing.

## 7. Evaluation Results (held-out test set, 300 customers)

**Confusion matrix:**

|  | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | 182 (TN) | 64 (FP) |
| **Actual Churn** | 12 (FN) | 42 (TP) |

| Metric | Value |
|---|---|
| Recall (churn) | **78%** — catches 42 of 54 actual churners |
| Precision (churn) | **40%** — trade-off for that recall: plenty of false alarms |
| ROC-AUC | **0.82** |
| Model accuracy | 75% |
| Baseline accuracy ("always predict No Churn") | 82% |

**Why accuracy is the wrong headline metric here:** the naive baseline of
predicting "No Churn" for every single customer already scores 82%
accuracy — higher than the actual model (75%). That's not the model
failing; it's the direct, visible cost of telling `class_weight="balanced"`
to prioritize catching churners over raw accuracy. A retention team cares
far more about the 78% recall (most churners get flagged) than about
matching a baseline that, by construction, catches zero churners. This is
exactly why precision, recall, F1, and ROC-AUC were reported instead of
leading with accuracy.

## 8. What's Actually Driving the Model

Reading the logistic regression coefficients directly:

- **Pushes toward churn:** `number_of_complaints`, `late_payment_count`,
  `customer_service_calls` (the three strongest positive coefficients),
  followed by `gender_Male` and `device_brand_Xiaomi`
- **Pushes toward staying:** `tenure_months` (by far the strongest
  negative coefficient), `satisfaction_score`, `plan_type_Prepaid`,
  `streaming_service_used_Yes`

This lines up closely with the data dictionary's stated design (churn
tied to complaints, late payments, service calls, satisfaction, and low
tenure) — a good sign the model learned real signal rather than noise.

## 9. Bonus Regression Task (for a genuine RMSE)

Since churn itself is a classification problem, RMSE doesn't apply to it
directly. As a small side-task, a plain Linear Regression was trained to
predict `avg_monthly_bill` from usage-behavior features (tenure, data
usage, call minutes, SMS count, app/social media usage, VAS count, family
lines) — deliberately excluding other billing-related columns to avoid a
trivial, leaky prediction.

- **RMSE:** 151.3 (against a target mean of ~₹237)
- **R²:** -0.023 (worse than simply predicting the average every time)

**Honest limitation:** the chosen usage features don't explain billing
amount well in this dataset. Including `monthly_recharge_amount` as a
predictor would very likely improve this substantially, but was left out
here as a documented limitation rather than quietly picked for an
easier result.

## 10. Deployment — FastAPI Endpoint

The final fitted pipeline (preprocessing + tuned Logistic Regression) was
serialized as a single `.joblib` artifact and wrapped in a standalone
FastAPI app (`POST /predict-churn`), separate from the existing Employee
Management API project. It returns a churn probability, a Yes/No
prediction, and a Low/Medium/High risk tier for retention prioritization.
All fields accept missing values, since the pipeline's own imputers
handle them exactly as they were trained to.

Tested end-to-end with three profiles:

| Profile | Result |
|---|---|
| Low tenure, 3 complaints, late payments, low satisfaction | 78% churn — High risk |
| 4-year tenure, satisfaction = 5, rest blank | 18% churn — Low risk |
| Fully empty payload (pure imputation) | 54% churn — Medium risk |

## 11. Limitations & Future Work

- Clustering did not reveal actionable customer segments in this dataset
  — a different unsupervised approach (e.g. DBSCAN, or clustering on a
  narrower, hand-picked feature set) might do better, but wasn't in scope
  here.
- Precision on churn (40%) means retention outreach based on this model
  would target real customers who weren't actually leaving — a real
  business cost to weigh against the value of the 78% recall.
- The bonus regression task underperformed; adding `monthly_recharge_amount`
  as a predictor is a clear, quick next step.
- The `DD/MM` vs `MM/DD` date ambiguity for a subset of `signup_date`
  values is a documented assumption, not a certainty.

## 12. Conclusion

The final model (Logistic Regression, tuned via 5-fold stratified CV) 
reaches an ROC-AUC of 0.82 and catches 78% of actual churners on unseen
data, at the cost of a 40% precision — a reasonable trade-off for a
retention use case where missing a churner is usually costlier than a
wasted outreach call. Tenure and satisfaction score are the strongest
protective factors against churn; complaints, late payments, and support
calls are the strongest risk factors. The full pipeline — from raw messy
CSV to a live prediction endpoint — is reproducible end-to-end and
deployed as a working FastAPI service.
