# Data Dictionary — Mobile Churn Prediction Dataset (Maharashtra Cities)

## Column Reference

| # | Column | Type | Description | Messiness injected |
|---|--------|------|--------------|---------------------|
| 1 | `customer_id` | string | Unique customer identifier (`CUSTxxxxx`) | 3 duplicate IDs (from duplicate rows) |
| 2 | `city` | categorical | City in Maharashtra | Inconsistent casing/spacing, aliases ("Bombay" for Mumbai), null-like tokens, **biased toward Mumbai/Pune** |
| 3 | `age` | numeric | Customer age | Missing values, outliers (negative, 0, 130–200), some stored as text ("52 yrs") — **mixed dtype** |
| 4 | `gender` | categorical | Gender | Inconsistent labels (Male/M/male), missing values |
| 5 | `monthly_income` | numeric | Monthly income (INR) | **Right-skewed (lognormal)**, missing values, outliers (negative, 0, 9999999), some as text ("Rs.14200.0") |
| 6 | `plan_type` | categorical | Prepaid / Postpaid | Inconsistent casing |
| 7 | `tenure_months` | numeric | Months since signup | Exponential distribution (skewed), outliers (negative, >96 months) |
| 8 | `monthly_recharge_amount` | numeric | Avg. recharge amount (INR) | Gamma distribution (skewed) |
| 9 | `data_usage_gb` | numeric | Monthly data usage (GB) | Exponential distribution (skewed), missing values |
| 10 | `call_minutes_used` | numeric | Monthly call minutes | Missing values |
| 11 | `sms_count` | numeric | Monthly SMS count | Poisson distributed |
| 12 | `number_of_complaints` | numeric | Complaints raised | Poisson distributed |
| 13 | `complaint_resolved` | categorical | Was complaint resolved | "N/A" when no complaints, inconsistent casing |
| 14 | `network_type` | categorical | 2G/3G/4G/5G | Inconsistent casing, null-like tokens |
| 15 | `device_brand` | categorical | Phone brand | Missing values, inconsistent casing, "Unknown" |
| 16 | `device_price_range` | categorical | Device price bracket | Null-like tokens |
| 17 | `avg_monthly_bill` | numeric | Avg. monthly bill (INR) | Missing values |
| 18 | `late_payment_count` | numeric | Count of late payments | Poisson distributed, feeds into churn signal |
| 19 | `contract_type` | categorical | Contract status | Inconsistent casing, null-like tokens |
| 20 | `customer_service_calls` | numeric | Support calls made | Poisson distributed, feeds into churn signal |
| 21 | `international_roaming_used` | categorical | Yes/No | **Imbalanced** (~8% Yes) |
| 22 | `value_added_services_count` | numeric | Count of VAS subscribed | Poisson distributed |
| 23 | `satisfaction_score` | numeric | 1 (low) – 5 (high) | Missing values, feeds into churn signal |
| 24 | `app_usage_hours_per_week` | numeric | App usage hours/week | Exponential (skewed), missing values |
| 25 | `social_media_usage_hours` | numeric | Social media hours/week | Exponential (skewed) |
| 26 | `streaming_service_used` | categorical | Yes/No | — |
| 27 | `number_of_family_lines` | numeric | Linked family connections | — |
| 28 | `payment_method` | categorical | Payment mode | Inconsistent casing, missing values |
| 29 | `signup_date` | date (string) | Signup date | **4 mixed date formats** (`YYYY-MM-DD`, `DD/MM/YYYY`, `DD-MM-YYYY`, `MM/DD/YYYY`) — must be parsed carefully |
| 30 | `churn` | categorical (target) | Yes / No | **Imbalanced (~18% Yes / 82% No)** — realistic churn bias; weakly correlated with complaints, late payments, service calls, satisfaction, and low tenure, plus noise |

