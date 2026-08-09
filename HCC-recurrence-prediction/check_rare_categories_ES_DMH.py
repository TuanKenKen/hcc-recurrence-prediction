import pandas as pd

train_dev_path = "data/xgboost/train_dev_xgboost_v1.xlsx"
test_path = "data/xgboost/test_xgboost_v1.xlsx"

train_dev_data = pd.read_excel(train_dev_path)
test_data = pd.read_excel(test_path)

RARE_CHECKS = {
    "ES": ["ES I"],
    "DMH": ["bè"]
}

def normalize_text(series):
    return series.astype(str).str.strip()

print("\n==============================")
print("RARE CATEGORY CHECK IN EXISTING SPLITS")
print("==============================")

print("\nTrain+Dev rows:", len(train_dev_data))
print("Test rows:", len(test_data))

summary_rows = []

for col, rare_values in RARE_CHECKS.items():
    print("\n" + "=" * 60)
    print(f"Column: {col}")
    print("=" * 60)

    if col not in train_dev_data.columns:
        print(f"WARNING: Column '{col}' not found in Train+Dev file.")
        continue

    if col not in test_data.columns:
        print(f"WARNING: Column '{col}' not found in Test file.")
        continue

    train_col = normalize_text(train_dev_data[col])
    test_col = normalize_text(test_data[col])

    print("\nTrain+Dev category counts:")
    print(train_col.value_counts(dropna=False))

    print("\nTest category counts:")
    print(test_col.value_counts(dropna=False))

    for value in rare_values:
        train_count = (train_col == value).sum()
        test_count = (test_col == value).sum()
        total_count = train_count + test_count

        print("\nRare category:", value)
        print("Train+Dev count:", train_count)
        print("Test count:", test_count)
        print("Total count:", total_count)

        if train_count == 0 and test_count > 0:
            status = "PROBLEM: appears in Test but not in Train+Dev"
        elif train_count == 0 and test_count == 0:
            status = "Not found in either split"
        elif train_count < 5:
            status = "Weak: present in Train+Dev but very few samples"
        else:
            status = "Probably okay"

        print("Status:", status)

        summary_rows.append({
            "column": col,
            "rare_category": value,
            "train_dev_count": train_count,
            "test_count": test_count,
            "total_count": total_count,
            "status": status
        })

summary_df = pd.DataFrame(summary_rows)

output_path = "data/xgboost/rare_category_ES_DMH_existing_split_check.xlsx"
summary_df.to_excel(output_path, index=False)

print("\n==============================")
print("SUMMARY")
print("==============================")
print(summary_df)

print(f"\nSaved report to: {output_path}")