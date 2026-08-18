
#Step 1 — EDA & Data Cleaning

import re
import numpy as np
import pandas as pd

RAW_PATH = "data/raw.csv"
CLEAN_PATH = "data/cleaned.csv"

# 1. Load + initial profile

df = pd.read_csv(RAW_PATH)
print(f"Raw shape: {df.shape}")

n_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {n_dupes} exact duplicate rows -> shape now {df.shape}")

NULL_TOKENS = {"", "-", "?", "n/a", "na", "null", "none", "nan"}

def clean_null_tokens(series: pd.Series) -> pd.Series:

    s = series.astype(str).str.strip()
    s = s.where(~s.str.lower().isin(NULL_TOKENS), np.nan)
    s = s.replace({"nan": np.nan})
    return s


# 2. city

df["city"] = clean_null_tokens(df["city"])

CITY_MAP = {
    "mumbai": "Mumbai", "bombay": "Mumbai",
    "pune": "Pune",
    "thane": "Thane",
    "nagpur": "Nagpur", "nagpur city": "Nagpur",
    "nashik": "Nashik", "nasik": "Nashik",
    "solapur": "Solapur",
    "amravati": "Amravati",
    "navimumbai": "Navi Mumbai", "navi mumbai": "Navi Mumbai",
    "kolhapur": "Kolhapur",
    "aurangabad": "Chhatrapati Sambhajinagar", 
    "chhatrapati sambhajinagar": "Chhatrapati Sambhajinagar",
    "sangli": "Sangli",
    "akola": "Akola",
    "unknown": np.nan,
}
df["city"] = (
    df["city"]
    .str.lower().str.strip()
    .map(CITY_MAP)
)
print("\ncity after cleaning:")
print(df["city"].value_counts(dropna=False))


# 3. age

df["age"] = (
    df["age"].astype(str).str.extract(r"(-?\d+\.?\d*)")[0].astype(float)
)
before = df["age"].notna().sum()
df.loc[(df["age"] <= 0) | (df["age"] > 100), "age"] = np.nan
after = df["age"].notna().sum()
print(f"\nage: {before - after} impossible values (<=0 or >100) set to NaN")

# 4. monthly_income

df["monthly_income"] = (
    df["monthly_income"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
)
df["monthly_income"] = pd.to_numeric(df["monthly_income"], errors="coerce")
before = df["monthly_income"].notna().sum()
df.loc[(df["monthly_income"] <= 0) | (df["monthly_income"] > 500_000), "monthly_income"] = np.nan
after = df["monthly_income"].notna().sum()
print(f"monthly_income: {before - after} outliers (<=0 or >500k) set to NaN")

# 5. signup_date

def parse_signup_date(raw):
    if pd.isna(raw):
        return pd.NaT
    s = str(raw).strip()

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)      
    if m:
        y, mo, d = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d)

    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", s)         
    if m:
        d, mo, y = map(int, m.groups())
        return pd.Timestamp(year=y, month=mo, day=d)

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)     
    if m:
        a, b, y = map(int, m.groups())
        if b > 12 and a <= 12:
            mo, d = a, b                    
        elif a > 12 and b <= 12:
            d, mo = a, b                  
        else:
            d, mo = a, b                   
        try:
            return pd.Timestamp(year=y, month=mo, day=d)
        except ValueError:
            return pd.NaT

    return pd.NaT

df["signup_date"] = df["signup_date"].apply(parse_signup_date)
n_failed = df["signup_date"].isna().sum()
print(f"\nsignup_date: parsed OK, {n_failed} rows could not be parsed")

REFERENCE_DATE = pd.Timestamp("2026-08-11")  
df["tenure_days_from_signup"] = (REFERENCE_DATE - df["signup_date"]).dt.days

# 6.categorical cleanups

def normalize_categorical(series, mapping=None):
    s = clean_null_tokens(series)
    s = s.str.strip()
    if mapping:
        s = s.apply(lambda x: mapping.get(x.lower(), x) if pd.notna(x) else x)
    return s

df["gender"] = normalize_categorical(df["gender"], {
    "m": "Male", "male": "Male",
    "f": "Female", "female": "Female",
    "other": "Other",
})

df["plan_type"] = normalize_categorical(df["plan_type"], {
    "prepaid": "Prepaid", "postpaid": "Postpaid",
})

df["network_type"] = normalize_categorical(df["network_type"], {
    "2g": "2G", "3g": "3G", "4g": "4G", "5g": "5G", "unknown": np.nan,
})

df["contract_type"] = normalize_categorical(df["contract_type"], {
    "no contract": "No Contract", "with contract": "With Contract",
    "unknown": np.nan,
})

df["device_price_range"] = normalize_categorical(df["device_price_range"], {
    "unknown": np.nan,
})

df["payment_method"] = normalize_categorical(df["payment_method"], {
    "upi": "UPI", "debit card": "Debit Card", "credit card": "Credit Card",
    "cash": "Cash", "wallet": "Wallet", "net banking": "Net Banking",
})

df["device_brand"] = normalize_categorical(df["device_brand"], {
    "samsung": "Samsung", "apple": "Apple", "xiaomi": "Xiaomi",
    "vivo": "Vivo", "oppo": "Oppo", "realme": "Realme", "oneplus": "OnePlus",
    "unknown": np.nan,
})

df["complaint_resolved"] = normalize_categorical(df["complaint_resolved"], {
    "yes": "Yes", "no": "No",
})
mask_no_complaint = df["number_of_complaints"].fillna(0) == 0
implied = df["complaint_resolved"].isna() & mask_no_complaint
print(f"\ncomplaint_resolved: {implied.sum()} of the NaNs line up with "
      f"number_of_complaints == 0 (confirms it's a logical 'No complaint' case)")
df.loc[mask_no_complaint, "complaint_resolved"] = df.loc[mask_no_complaint, "complaint_resolved"].fillna("No Complaint Filed")

df["international_roaming_used"] = normalize_categorical(df["international_roaming_used"])
df["streaming_service_used"] = normalize_categorical(df["streaming_service_used"])
df["churn"] = normalize_categorical(df["churn"])

# 7. Save + summary

df.to_csv(CLEAN_PATH, index=False)

print("\n" + "=" * 60)
print(f"Cleaned shape: {df.shape}")
print("Missing values per column after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))
print(f"\nSaved -> {CLEAN_PATH}")
