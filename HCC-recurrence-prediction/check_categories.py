import pandas as pd

df = pd.read_excel("data/synthetic_hcc_dummy.xlsx")

df.columns = (
    df.columns
    .str.replace("\u00a0", " ", regex=False)
    .str.strip()
)

TARGET = "Tái phát"

CATEGORICAL_COLS = [
    "Giới",
    "Nhiễm virus VG",
    "Mdcg",
    "Xnmm",
    "Số lượng u",
    "Hoại tử u",
    "DMH",
    "ES",
    "Bệnh gan nền",
    "Di căn"
]

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "có": 1,
        "co": 1,
        "yes": 1,
        "1": 1,
        "không": 0,
        "khong": 0,
        "no": 0,
        "0": 0
    })
)

df = df.dropna(subset=[TARGET])
df[TARGET] = df[TARGET].astype(int)

print("\n=========================")
print("CATEGORICAL COLUMN CHECK")
print("=========================")

for col in CATEGORICAL_COLS:
    print("\n" + "=" * 60)
    print(f"Column: {col}")
    print("=" * 60)

    if col not in df.columns:
        print(f"WARNING: Column '{col}' not found in dataset.")
        continue

    print("\nTotal rows:", len(df))
    print("Missing values:", df[col].isna().sum())
    print("Non-missing values:", df[col].notna().sum())
    print("Number of unique categories:", df[col].nunique(dropna=True))

    print("\nCategory counts:")
    print(df[col].value_counts(dropna=False))

    print("\nCategory percentages:")
    print((df[col].value_counts(dropna=False, normalize=True) * 100).round(2))

MIN_COUNT = 10

print("\n\n=========================")
print(f"RARE CATEGORY CHECK: count < {MIN_COUNT}")
print("=========================")

for col in CATEGORICAL_COLS:
    if col not in df.columns:
        continue

    counts = df[col].value_counts(dropna=False)
    rare = counts[counts < MIN_COUNT]

    print("\n" + "=" * 60)
    print(f"Column: {col}")
    print("=" * 60)

    if rare.empty:
        print("No rare categories found.")
    else:
        print("Rare categories:")
        print(rare)

output_path = "data/categorical_category_check.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(dropna=False, normalize=True) * 100

        report = pd.DataFrame({
            "category": counts.index.astype(str),
            "count": counts.values,
            "percentage": percentages.round(2).values
        })

        report["is_rare"] = report["count"] < MIN_COUNT

        sheet_name = col[:31]  # Excel sheet names must be <= 31 characters
        report.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nSaved categorical category report to: {output_path}")