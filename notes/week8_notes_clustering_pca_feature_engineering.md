# Week 8 Notes — Clustering, PCA, Feature Engineering, Tuning & Evaluation
### Applied to: Mobile Churn Prediction (Maharashtra Cities)

---

## 0. The big picture — how this week's pieces fit together

Think of building this project like running a restaurant kitchen for one dish (churn prediction):

1. **Feature engineering** = washing, peeling, cutting the vegetables (raw data → usable ingredients)
2. **PCA / clustering** = tasting and grouping ingredients before cooking, to understand what you're working with
3. **Model + tuning** = actually cooking, adjusting heat and time until it's right
4. **Metrics** = the taste test — did it come out good, and *how* good, in ways that matter
5. **FastAPI endpoint** = plating it and serving it to a customer (another program) who just wants to ask "will this person churn?" and get an answer back

Everything below maps onto one of these five steps.

---

## 1. Feature Engineering — cleaning and preparing the ingredients

This dataset is *deliberately* messy (that's the point of the exercise). I checked it directly — some real numbers:

- **1,515 rows, 30 columns**, with **3 exact duplicate rows** (same `customer_id` repeated) — drop these first.
- `city` has the same city written 6+ different ways: `Mumbai`, `Mumbai ` (trailing space), `mumbai`, `MUMBAI`, `Bombay` — all the same city. `Pune`, `PUNE`, ` Pune`, `pune` — same story. This needs **string cleaning + alias mapping** before anything else.
- `age` and `monthly_income` are stored as **text**, not numbers (`"52 yrs"`, `"Rs.14200.0"`) — you have to strip the units and cast to numeric before you can do any math on them.
- `signup_date` has **4 different date formats mixed in the same column** (`2019-05-11`, `25-07-2019`, `06/10/2022`, `04/30/2021`) — you can't just `pd.to_datetime()` blindly, because `06/10/2022` and `04/30/2021` aren't unambiguous without knowing which format rule applies to which row.
- `churn` (your target) is **82% No / 18% Yes** (1,237 vs 275, plus 3 missing) — a real imbalance you'll need to handle in both modeling and metric choice.

**Analogy:** if PCA/clustering/modeling are cooking, skipping feature engineering is like trying to cook unwashed vegetables with the stems still on — the recipe won't fail loudly, it'll just quietly taste wrong (a model trained on `"52 yrs"` as a string, or Mumbai split into 6 fake "different" cities, will learn nonsense).

### The sub-tasks, in the order you'd actually do them:

| Step | What it means | Applied here |
|---|---|---|
| **Data cleaning** | Fix things that are wrong at the raw level | Drop duplicate `customer_id`s, strip whitespace/casing, map `Bombay`→`Mumbai`, strip `" yrs"` / `"Rs."` and cast to numeric, standardize null-like tokens (`"N/A"`, `"-"`, `"NULL"`) to real `NaN` |
| **Date parsing** | Turn text into a real, comparable date | Try each of the 4 known formats per row (`dateutil.parser` with `dayfirst` toggled, or explicit regex-based format detection), then derive a useful feature like `tenure_days` or `signup_year` |
| **Missing data / Imputation** | Fill in gaps sensibly, don't just drop rows | Numeric (e.g. `satisfaction_score`, `data_usage_gb`): median or KNN imputation. Categorical (e.g. `device_brand`, `payment_method`): mode, or an explicit `"Unknown"` category. `complaint_resolved` has 433 missing — but that's expected, because it's `N/A` when `number_of_complaints == 0`, so this isn't really "missing," it's a logical `"No complaint"` category |
| **Encoding** | Turn categories into numbers a model can use | One-hot encoding for low-cardinality columns (`plan_type`, `gender`, `network_type`); consider target/frequency encoding for higher-cardinality ones like `city` or `device_brand` if one-hot gets too wide |
| **Scaling** | Put numeric features on comparable ranges | `StandardScaler` or `MinMaxScaler` on things like `monthly_income`, `tenure_months`, `data_usage_gb` — **essential for k-means and PCA**, which are distance-based and get dominated by whichever column has the biggest raw numbers otherwise |
| **Skewness handling** | Fix lopsided distributions | `monthly_income` (lognormal), `tenure_months`, `monthly_recharge_amount`, `data_usage_gb`, `app_usage_hours_per_week`, `social_media_usage_hours` are all flagged as skewed in the data dictionary. A `log1p` transform (or `PowerTransformer`) before scaling will help both the clustering and the regression pieces |
| **Feature selection** | Drop what doesn't help | Drop `customer_id` (pure identifier, no signal). Check correlation / mutual information between features and `churn` to see which of `number_of_complaints`, `late_payment_count`, `customer_service_calls`, `satisfaction_score`, `tenure_months` actually carry the signal the dictionary says they should |
| **Visualization** | Look before you model | Histograms (before/after skew fix), boxplots for outliers (age of 0 or 200, income of −5000 or 9999999 are obviously bad data, not real customers), a correlation heatmap, and a bar chart of churn rate by category (e.g. does `contract_type` or `city` correlate with churn?) |

---

## 2. Dimensionality Reduction — PCA

**What it actually does:** you have ~20 numeric features after cleaning. PCA finds a smaller set of new "combined" features (principal components) that capture most of the *spread/variation* in the data, so you can work with fewer dimensions without losing much information.

**Analogy:** imagine photographing a 3D object with a single camera. You lose the "true" 3D shape, but if you pick the *right angle*, a 2D photo still tells you almost everything about the object's shape. PCA finds that best angle mathematically, for numbers instead of physical objects.

**Why it's useful for this project specifically:**
- After encoding, you might have 30–40 numeric columns. PCA reducing this to 2–3 components lets you actually **plot and see** customer clusters on a chart (impossible to visualize in 40 dimensions).
- It also removes redundancy — e.g. `app_usage_hours_per_week` and `data_usage_gb` are probably correlated (heavy app users use more data), so PCA can compress that overlap.
- Rule of thumb: keep enough components to explain ~90–95% of variance (check the "explained variance ratio" plot — an "elbow" chart similar to k-means below).

You'd typically apply PCA **after scaling** (component 1 in `1. Feature Engineering`), and use it mainly for (a) visualizing clusters, and optionally (b) feeding a reduced feature set into the model if you want to compare performance with vs. without PCA.

---

## 3. Clustering — k-means

**What it does:** unsupervised grouping — it looks at customers *without* knowing who churned, and groups them into `k` clusters of customers who "look similar" on the numeric features (spending, usage, tenure, complaints, etc.).

**Analogy:** if you dropped 1,500 strangers into a room and said "form small groups with people similar to you, but nobody's allowed to discuss it out loud" — k-means is the mathematical version of that: it randomly places `k` "group centers," assigns everyone to their nearest center, recalculates the center based on who joined, and repeats until the groups stop changing.

**Why it's useful here:** even though you already have a churn label, clustering can reveal *customer segments* that the label alone doesn't show — e.g. you might discover a cluster of "low-tenure, high-complaint, low-satisfaction" customers that has a much higher churn rate than average, which is valuable for the "written results summary" (a business insight, not just a prediction).

**How you'd do it:**
1. Use the scaled numeric features (ideally the PCA-reduced ones for speed/visualization).
2. Pick `k` using the **elbow method** (plot inertia/within-cluster-sum-of-squares vs. `k`, look for where the drop-off flattens) or **silhouette score** (measures how well-separated clusters are; higher is better).
3. Fit `KMeans(n_clusters=k)`, then **cross-tabulate cluster ID against actual churn** to describe each cluster in plain English (e.g. "Cluster 2 = high-value, low-churn; Cluster 4 = at-risk").

---

## 4. Cross-Validation & Hyperparameter Tuning

**Cross-validation (k-fold):** instead of one train/test split (Week 7's approach), you split the training data into `k` folds (commonly 5), train on `k-1` of them and validate on the held-out fold, and rotate through all `k` combinations, averaging the results.

**Analogy:** one train/test split is like taking one practice exam and assuming that score tells you everything. K-fold CV is like taking 5 different practice exams (each covering slightly different questions) and averaging your scores — much more reliable, because you're less likely to get a fluke result from an easy (or unusually hard) test.

**Why it matters especially here:** with an 18% churn rate, a single random split could accidentally put very few "Yes" churn cases in your test set, making your metrics unreliable. Use **Stratified K-Fold** (keeps the 82/18 ratio consistent in every fold) rather than plain K-Fold.

**Hyperparameter tuning (Grid Search vs Random Search):**
- **Grid Search:** exhaustively tries every combination of hyperparameters you specify (e.g. `C = [0.01, 0.1, 1, 10]` for logistic regression) — guaranteed to find the best combo *within your grid*, but slow if the grid is large.
- **Random Search:** samples random combinations from specified ranges — faster, and often nearly as good, especially when you have many hyperparameters and only a few actually matter.
- Practical approach for this project: `GridSearchCV(model, param_grid, cv=StratifiedKFold(5), scoring='f1')` — note **F1 as the scoring metric, not accuracy** (explained below).

---

## 5. Metrics — how to actually judge the model

**Confusion Matrix** — the foundation everything else is built from:

|  | Predicted No Churn | Predicted Churn |
|---|---|---|
| **Actual No Churn** | True Negative (TN) | False Positive (FP) |
| **Actual Churn** | False Negative (FN) | True Positive (TP) |

From this:
- **Accuracy** = (TP+TN) / total — **misleading here**: a model that just predicts "No churn" for *everyone* would already score 82% accuracy (matching your 1,237/1,515 "No" rate) while being completely useless. Don't optimize for this alone.
- **Precision** = TP / (TP+FP) — "of everyone I flagged as likely to churn, how many actually did?" High precision = few wasted retention-offer calls to people who weren't leaving anyway.
- **Recall** = TP / (TP+FN) — "of everyone who actually churned, how many did I catch?" High recall = few churners slip through undetected. In a churn business context, **recall usually matters more** — missing a churner (losing a customer silently) is often more costly than a wasted retention offer.
- **F1 score** = harmonic mean of precision and recall — a single number balancing both, useful for comparing models/tuning runs when you don't want to pick just one of precision/recall.
- **ROC-AUC** = measures how well the model *ranks* churners above non-churners across all possible thresholds (0 to 1) — good for imbalanced problems since it doesn't depend on picking one cutoff.
- **RMSE** (Root Mean Squared Error) — this is a *regression* metric (predicting a continuous number, not Yes/No), so it doesn't apply directly to churn itself. It fits if you add a genuine regression sub-task — see the note in the plan below.

**Practical rule for this project:** report the confusion matrix + precision/recall/F1/ROC-AUC as your primary evaluation, and explicitly call out in your written summary *why* accuracy alone would be misleading, given the 82/18 imbalance.

---

## 6. Clearing up "any regression and clustering algorithm"

The brief asks for a regression algorithm **and** a clustering algorithm inside one pipeline, but the target (`churn`) is Yes/No — technically a classification problem. Here's the cleanest way to satisfy both, and what I'd recommend building:

1. **Clustering → k-means** on the cleaned/scaled features, for customer segmentation (Section 3). This is straightforward and directly useful for the written summary.
2. **"Regression" → Logistic Regression** for the churn prediction itself. Logistic regression *is* a regression algorithm (it models log-odds as a linear function of your features) even though its output is used for classification — this is the natural continuation of Week 7, where you already covered it.
3. **Optional but recommended:** add one small, genuinely-continuous regression side-task (e.g. predicting `avg_monthly_bill` or `monthly_recharge_amount` from usage features) purely so you have a real RMSE to report — takes maybe 20 extra minutes and fully covers every metric listed in the brief.

---

## 7. The build plan, step by step

1. **EDA & cleaning** — load data, profile it (`df.info()`, `df.describe()`, `df.isnull().sum()`), fix duplicates, casing/aliases, mixed dtypes, null-like tokens, parse `signup_date`.
2. **Feature engineering** — impute missing values, encode categoricals, log-transform skewed columns, scale numerics, drop `customer_id`.
3. **Unsupervised exploration** — PCA (for visualization + optional dimensionality reduction) → k-means (customer segments) → describe segments against actual churn rate.
4. **Supervised model** — Logistic Regression on churn, `StratifiedKFold` cross-validation, `GridSearchCV`/`RandomizedSearchCV` for hyperparameter tuning (`C`, `penalty`, `class_weight='balanced'` given the imbalance).
5. **Evaluation** — confusion matrix, precision/recall/F1/ROC-AUC on a held-out test set; (optional) linear regression + RMSE side-task.
6. **Persist the model** — same `pickle` pattern you used in Week 7, save the fitted preprocessing pipeline + model together (e.g. as a single `sklearn.Pipeline` object) so the API loads one artifact.
7. **FastAPI endpoint** — extend your existing FastAPI project:
   - `POST /predict-churn` — accepts raw customer fields (JSON body via a Pydantic model matching the dataset columns), runs them through the *same* saved preprocessing pipeline, returns `{"churn_probability": 0.73, "churn_prediction": "Yes"}`.
   - Optionally `GET /customer-segments/{customer_id}` — returns which cluster a customer belongs to and its profile.
   - Reuse the Swagger docs / Pydantic validation patterns from your Employee Management API (Week 3), and load the pickled pipeline once at startup (not per-request) for performance.
8. **Written results summary** — what you cleaned and why, chosen `k` for clustering with justification (elbow/silhouette), PCA variance explained, final model's precision/recall/F1/ROC-AUC with an honest note on the class imbalance, and 2–3 plain-English business insights (e.g. "high-complaint, low-tenure customers in Cluster X churn at 3x the average rate").

---

**Next step:** if you want, we can start with Step 1 (EDA + cleaning script) against your actual uploaded CSV, and build outward from there.
