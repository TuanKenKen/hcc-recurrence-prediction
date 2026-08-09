import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/synthetic_hcc_dummy.xlsx"
OUTPUT_DIR = "data/skewness_check"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERIC_COLS = [
    "Tuổi",
    "AFP (ng/ml)",
    "K/thước u (cm)",
    "Gly (mg%)",
    "Cre (mg%)",
    "ALT (U/l)",
    "AST (U/l)",
    "Bil (mg%)",
    "Alb (g%)",
    "PT (sec)",
    "APTT (sec)",
    "TC (G/l)",
    "Fib (G/l)",
    "APRI",
    "FIB 4",
    "ALBI"
]

# HELPER FUNCTIONS

def clean_column_name(col):
    return (
        str(col)
        .replace("\u00a0", " ")
        .strip()
    )


def make_safe_col_name(col):
    col = clean_column_name(col)
    col = col.replace("%", "percent")
    col = col.replace("/", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace(" ", "_")
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")
    return col


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.replace("\u00a0", "")
    value = value.replace(",", ".")

    if value.startswith(">"):
        value = value.replace(">", "").strip()

    if value.startswith("<"):
        value = value.replace("<", "").strip()

    return pd.to_numeric(value, errors="coerce")


def classify_skewness(skew):
    if pd.isna(skew):
        return "Cannot calculate"

    abs_skew = abs(skew)

    if abs_skew < 0.5:
        return "Approximately symmetric"
    elif abs_skew < 1:
        return "Moderately skewed"
    else:
        return "Strongly skewed"


def recommend_transformation(raw_skew, log_skew, min_valid, has_negative):
    if min_valid < 30:
        return "Check manually - too few valid values"

    if has_negative:
        return "Keep raw - contains negative values"

    if pd.isna(raw_skew) or pd.isna(log_skew):
        return "Check manually - skewness unavailable"

    if abs(raw_skew) >= 1 and abs(log_skew) < abs(raw_skew):
        return "Use log"

    if abs(raw_skew) >= 0.75 and abs(log_skew) <= 0.5:
        return "Use log"

    if abs(raw_skew) >= 0.75 and abs(log_skew) < abs(raw_skew) * 0.7:
        return "Consider log"

    return "Keep raw"


def plot_distribution(series, title, xlabel, save_path):
    plt.figure(figsize=(8, 5))
    series.dropna().hist(bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_boxplot(series, title, xlabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.boxplot(series.dropna(), vert=False)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# LOAD DATA

df = pd.read_excel(DATA_PATH)

print("\nRAW COLUMN NAMES:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {repr(col)}")

df.columns = [clean_column_name(col) for col in df.columns]

print("\nCLEANED COLUMN NAMES:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:02d}. {repr(col)}")

# CHECK MISSING COLUMNS

missing_cols = [col for col in NUMERIC_COLS if col not in df.columns]

if missing_cols:
    print("\nERROR: These numeric columns are missing:")
    print(missing_cols)
    raise ValueError("Please fix NUMERIC_COLS column names.")

print("\nAll numeric columns found.")

# CLEAN NUMERIC COLUMNS

clean_df = pd.DataFrame()

for col in NUMERIC_COLS:
    clean_df[col] = df[col].apply(clean_numeric_value)

# SKEWNESS CHECK

summary_rows = []

for col in NUMERIC_COLS:
    raw = clean_df[col]
    valid_raw = raw.dropna()

    total_rows = len(raw)
    missing_count = raw.isna().sum()
    valid_count = raw.notna().sum()

    raw_min = valid_raw.min() if valid_count > 0 else np.nan
    raw_max = valid_raw.max() if valid_count > 0 else np.nan
    raw_mean = valid_raw.mean() if valid_count > 0 else np.nan
    raw_median = valid_raw.median() if valid_count > 0 else np.nan
    raw_skew = valid_raw.skew() if valid_count > 2 else np.nan

    has_negative = (valid_raw < 0).any() if valid_count > 0 else False
    has_zero_or_positive = (valid_raw >= 0).all() if valid_count > 0 else False

    # log1p requires values > -1.
    can_log1p = valid_count > 0 and (valid_raw > -1).all()

    if can_log1p:
        log_values = np.log1p(raw)
        valid_log = log_values.dropna()
        log_mean = valid_log.mean() if len(valid_log) > 0 else np.nan
        log_median = valid_log.median() if len(valid_log) > 0 else np.nan
        log_skew = valid_log.skew() if len(valid_log) > 2 else np.nan
    else:
        log_values = pd.Series([np.nan] * len(raw))
        log_mean = np.nan
        log_median = np.nan
        log_skew = np.nan

    recommendation = recommend_transformation(
        raw_skew=raw_skew,
        log_skew=log_skew,
        min_valid=valid_count,
        has_negative=has_negative
    )

    summary_rows.append({
        "column": col,
        "total_rows": total_rows,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "min": raw_min,
        "max": raw_max,
        "mean_raw": raw_mean,
        "median_raw": raw_median,
        "skew_raw": raw_skew,
        "raw_skew_class": classify_skewness(raw_skew),
        "can_log1p": can_log1p,
        "mean_log1p": log_mean,
        "median_log1p": log_median,
        "skew_log1p": log_skew,
        "log_skew_class": classify_skewness(log_skew),
        "abs_skew_reduction": abs(raw_skew) - abs(log_skew) if not pd.isna(raw_skew) and not pd.isna(log_skew) else np.nan,
        "recommendation": recommendation
    })

    safe_col = make_safe_col_name(col)

    plot_distribution(
        raw,
        title=f"Distribution of Raw {col}",
        xlabel=col,
        save_path=os.path.join(OUTPUT_DIR, f"{safe_col}_raw_hist.png")
    )

    plot_boxplot(
        raw,
        title=f"Boxplot of Raw {col}",
        xlabel=col,
        save_path=os.path.join(OUTPUT_DIR, f"{safe_col}_raw_boxplot.png")
    )

    if can_log1p:
        plot_distribution(
            log_values,
            title=f"Distribution of log1p({col})",
            xlabel=f"{safe_col}_log",
            save_path=os.path.join(OUTPUT_DIR, f"{safe_col}_log_hist.png")
        )

        plot_boxplot(
            log_values,
            title=f"Boxplot of log1p({col})",
            xlabel=f"{safe_col}_log",
            save_path=os.path.join(OUTPUT_DIR, f"{safe_col}_log_boxplot.png")
        )

# SAVE SUMMARY

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by="skew_raw",
    key=lambda s: s.abs(),
    ascending=False
)

summary_path = os.path.join(OUTPUT_DIR, "numeric_skewness_summary.xlsx")
summary_df.to_excel(summary_path, index=False)

print("\n=========================")
print("NUMERIC SKEWNESS SUMMARY")
print("=========================")

print(summary_df[
    [
        "column",
        "valid_count",
        "missing_count",
        "min",
        "max",
        "skew_raw",
        "skew_log1p",
        "abs_skew_reduction",
        "recommendation"
    ]
])

print("\nSaved summary file:")
print(summary_path)

print("\nSaved histogram and boxplot images in:")
print(OUTPUT_DIR)

# PRINT RECOMMENDED LISTS

use_log_cols = summary_df.loc[
    summary_df["recommendation"].isin(["Use log", "Consider log"]),
    "column"
].tolist()

keep_raw_cols = summary_df.loc[
    summary_df["recommendation"].eq("Keep raw"),
    "column"
].tolist()

manual_check_cols = summary_df.loc[
    summary_df["recommendation"].str.contains("Check manually", na=False),
    "column"
].tolist()

print("\n=========================")
print("RECOMMENDED LOG_TRANSFORM_COLS")
print("=========================")
print(use_log_cols)

print("\n=========================")
print("RECOMMENDED RAW_NUMERIC_COLS_TO_KEEP")
print("=========================")
print(keep_raw_cols)

if manual_check_cols:
    print("\n=========================")
    print("MANUAL CHECK COLUMNS")
    print("=========================")
    print(manual_check_cols)