import os
import re
import difflib
import inspect
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib.pyplot as plt

from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc


DATA_PATH = "data/synthetic_hcc_dummy.xlsx"

os.makedirs("data", exist_ok=True)
os.makedirs("data/mlp", exist_ok=True)
os.makedirs("model", exist_ok=True)

TARGET = "Tái phát"
SURVIVAL_TIME_COL = "DFS"

DFS_UNIT = "days"

FEATURES = [
    "Giới",
    "Tuổi",
    "AFP (ng/ml)",
    "Nhiễm virus VG",
    "Mdcg",
    "Xnmm",
    "Số lượng u",
    "K/thước u (cm)",
    "Hoại tử u",
    "DMH",
    "ES",
    "Bệnh gan nền",
    "Di căn",
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

BASE_NUMERIC_COLS = [
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

LOG_TRANSFORM_COLS = [
    "AFP (ng/ml)",
    "K/thước u (cm)",
    "Gly (mg%)",
    "Cre (mg%)",
    "ALT (U/l)",
    "AST (U/l)",
    "Bil (mg%)",
    "Alb (g%)",
    "APRI",
    "FIB 4",
    "PT (sec)",
    "APTT (sec)",
    "TC (G/l)",
    "Fib (G/l)"
]

RAW_NUMERIC_COLS_TO_KEEP = [
    "Tuổi",
    "ALBI"
]

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

CENSORED_NUMERIC_COLS = [
    "AFP (ng/ml)",
    "PT (sec)",
    "APTT (sec)",
    "Fib (G/l)"
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


def check_required_columns(df, required_cols):
    existing_cols = df.columns.tolist()
    missing_cols = [col for col in required_cols if col not in existing_cols]

    if missing_cols:
        print("\nERROR: Some required columns are missing.\n")

        for col in missing_cols:
            suggestions = difflib.get_close_matches(col, existing_cols, n=5, cutoff=0.4)
            print(f"Missing column: {col}")
            print(f"Possible matches: {suggestions}")
            print()

        raise ValueError("Column name check failed. Please fix FEATURES / TARGET names.")

    print("\nColumn check passed. All required columns exist.")


def clean_numeric_value(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    value = value.replace("\u00a0", "")
    value = value.replace(",", ".")

    if value.startswith(">"):
        value = value.replace(">", "").strip()
        return pd.to_numeric(value, errors="coerce")

    if value.startswith("<"):
        value = value.replace("<", "").strip()
        return pd.to_numeric(value, errors="coerce")

    return pd.to_numeric(value, errors="coerce")


def create_censoring_indicators(X, cols):
    indicator_cols = []

    for col in cols:
        raw = X[col].astype(str).str.strip()
        safe_col = make_safe_col_name(col)

        greater_col = f"{safe_col}_is_greater_than"
        less_col = f"{safe_col}_is_less_than"

        X[greater_col] = raw.str.startswith(">").astype(int)
        X[less_col] = raw.str.startswith("<").astype(int)

        indicator_cols.extend([greater_col, less_col])

    return X, indicator_cols


def clean_categorical_value(value):
    if pd.isna(value):
        return np.nan

    value = str(value)
    value = value.replace("\u00a0", " ")
    value = value.strip()

    if value == "":
        return np.nan

    return value


def create_log_features(X, cols):
    log_cols = []

    for col in cols:
        safe_col = make_safe_col_name(col)
        log_col = f"{safe_col}_log"

        if (X[col].dropna() < 0).any():
            raise ValueError(
                f"Column '{col}' contains negative values. "
                f"Do not apply log1p to this column."
            )

        X[log_col] = np.log1p(X[col])
        log_cols.append(log_col)

    return X, log_cols


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def mlp_supports_sample_weight():
    return "sample_weight" in inspect.signature(MLPClassifier.fit).parameters


def fit_with_optional_sample_weight(pipeline, X_train, y_train):
    if mlp_supports_sample_weight():
        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        print(
            "\nWARNING: This scikit-learn version does not support "
            "sample_weight for MLPClassifier. Fitting without class weights."
        )
        pipeline.fit(X_train, y_train)

    return pipeline

# LOAD DATA

df = pd.read_excel(DATA_PATH)

df.columns = [clean_column_name(col) for col in df.columns]

required_cols = FEATURES + [TARGET, SURVIVAL_TIME_COL]
check_required_columns(df, required_cols)

# CLEAN TARGET

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

print("\nMissing target values:", df[TARGET].isna().sum())
print("\nTarget distribution before dropping missing:")
print(df[TARGET].value_counts(dropna=False))

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

print("\nTarget distribution after cleaning:")
print(df[TARGET].value_counts())
print(df[TARGET].value_counts(normalize=True))


X = df[FEATURES].copy()
y = df[TARGET].copy()

# CENSORING INDICATORS

X, INDICATOR_COLS = create_censoring_indicators(X, CENSORED_NUMERIC_COLS)

print("\nCreated censoring indicator columns:")
print(INDICATOR_COLS)

# CLEAN NUMERIC FEATURES

for col in BASE_NUMERIC_COLS:
    X[col] = X[col].apply(clean_numeric_value)

# CREATE LOG FEATURES

X, LOG_NUMERIC_COLS = create_log_features(X, LOG_TRANSFORM_COLS)

print("\nCreated log-transformed numeric columns:")
print(LOG_NUMERIC_COLS)

# CLEAN CATEGORICAL FEATURES

for col in CATEGORICAL_COLS:
    X[col] = X[col].apply(clean_categorical_value)

if "DMH" in X.columns:
    X["DMH"] = X["DMH"].replace({
        "bè": "Bè"
    })

for col in CATEGORICAL_COLS:
    X[col] = X[col].astype("object")

# 10. FINAL FEATURE LISTS

NUMERIC_COLS = (
    RAW_NUMERIC_COLS_TO_KEEP
    + LOG_NUMERIC_COLS
    + INDICATOR_COLS
)

print("\nFinal numeric columns used for prediction:")
print(NUMERIC_COLS)

print("\nFinal categorical columns used for prediction:")
print(CATEGORICAL_COLS)

check_required_columns(X, NUMERIC_COLS + CATEGORICAL_COLS)

# TRAIN / TEST SPLIT

X_train_dev, X_test, y_train_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

print("\nSplit sizes:")
print("Train+Dev:", X_train_dev.shape)
print("Test:", X_test.shape)

print("\nRecurrence ratio:")
print("Full:")
print(y.value_counts(normalize=True))

print("\nTrain+Dev:")
print(y_train_dev.value_counts(normalize=True))

print("\nTest:")
print(y_test.value_counts(normalize=True))

# SAVE SPLIT FILES

train_dev_data = X_train_dev.copy()
train_dev_data[TARGET] = y_train_dev

test_data = X_test.copy()
test_data[TARGET] = y_test

train_dev_data = train_dev_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_dev_data.to_excel("data/mlp/train_dev_mlp_v1.xlsx", index=False)
test_data.to_excel("data/mlp/test_mlp_v1.xlsx", index=False)

print("\nSaved train/dev and test split files.")

# PREPROCESSING PIPELINE

numeric_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler())
])

categorical_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", make_onehot_encoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess, NUMERIC_COLS),
        ("cat", categorical_preprocess, CATEGORICAL_COLS)
    ],
    sparse_threshold=0
)

# MLP MODEL

mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",

    alpha=0.001,
    learning_rate="adaptive",
    learning_rate_init=0.001,

    batch_size=64,
    max_iter=1000,

    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,

    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", mlp_model)
])

# 5-FOLD CROSS-VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_rows = []

print("\n5-FOLD CROSS-VALIDATION RESULTS")

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_dev, y_train_dev), start=1):
    X_fold_train = X_train_dev.iloc[train_idx]
    X_fold_val = X_train_dev.iloc[val_idx]

    y_fold_train = y_train_dev.iloc[train_idx]
    y_fold_val = y_train_dev.iloc[val_idx]

    fold_clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",

            alpha=0.001,
            learning_rate="adaptive",
            learning_rate_init=0.001,

            batch_size=64,
            max_iter=1000,

            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,

            random_state=42 + fold_idx
        ))
    ])

    fold_clf = fit_with_optional_sample_weight(
        fold_clf,
        X_fold_train,
        y_fold_train
    )

    val_prob = fold_clf.predict_proba(X_fold_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)

    fold_result = {
        "fold": fold_idx,
        "roc_auc": roc_auc_score(y_fold_val, val_prob),
        "average_precision": average_precision_score(y_fold_val, val_prob),
        "accuracy": accuracy_score(y_fold_val, val_pred),
        "precision": precision_score(y_fold_val, val_pred, zero_division=0),
        "recall": recall_score(y_fold_val, val_pred, zero_division=0),
        "f1": f1_score(y_fold_val, val_pred, zero_division=0)
    }

    fold_rows.append(fold_result)

fold_results = pd.DataFrame(fold_rows)

cv_summary = pd.DataFrame({
    "metric": [
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ],
    "mean": [
        fold_results["roc_auc"].mean(),
        fold_results["average_precision"].mean(),
        fold_results["accuracy"].mean(),
        fold_results["precision"].mean(),
        fold_results["recall"].mean(),
        fold_results["f1"].mean()
    ],
    "std": [
        fold_results["roc_auc"].std(),
        fold_results["average_precision"].std(),
        fold_results["accuracy"].std(),
        fold_results["precision"].std(),
        fold_results["recall"].std(),
        fold_results["f1"].std()
    ]
})

print(cv_summary)

cv_summary.to_excel(
    "data/mlp/cv_results_mlp_v1.xlsx",
    index=False
)

fold_results.to_excel(
    "data/mlp/cv_fold_results_mlp_v1.xlsx",
    index=False
)

# FINAL MODEL

final_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",

        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,

        batch_size=64,
        max_iter=1000,

        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,

        random_state=42
    ))
])

final_clf = fit_with_optional_sample_weight(
    final_clf,
    X_train_dev,
    y_train_dev
)

# SAVE FINAL FEATURE NAMES

feature_names = final_clf.named_steps["preprocessor"].get_feature_names_out()

feature_names_df = pd.DataFrame({
    "feature_index": range(1, len(feature_names) + 1),
    "feature_name": feature_names
})

feature_names_df.to_excel(
    "data/mlp/final_feature_names_after_preprocessing_mlp_v1.xlsx",
    index=False
)

print("\nSaved feature names.")

# FINAL TEST EVALUATION

test_prob = final_clf.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)

print("\nFINAL TEST RESULTS")
print(classification_report(y_test, test_pred, zero_division=0))

test_roc_auc = roc_auc_score(y_test, test_prob)
test_pr_auc = average_precision_score(y_test, test_prob)

print("ROC-AUC:", test_roc_auc)
print("PR-AUC:", test_pr_auc)

cm = confusion_matrix(y_test, test_pred)
print("Confusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

print("Sensitivity / Recall:", sensitivity)
print("Specificity:", specificity)

binary_brier = brier_score_loss(y_test, test_prob)
print("Binary Brier score:", binary_brier)

prob_true, prob_pred = calibration_curve(
    y_test,
    test_prob,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="MLP")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed recurrence rate")
plt.title("Calibration Curve - MLP")
plt.legend()
plt.tight_layout()
plt.savefig("data/mlp/calibration_curve_mlp_v1.png", dpi=300)
plt.close()

print("Saved calibration curve: data/mlp/calibration_curve_mlp_v1.png")

test_time = pd.to_numeric(df.loc[X_test.index, SURVIVAL_TIME_COL], errors="coerce")
test_event = y_test.copy()

valid_surv_mask = (
    test_time.notna()
    & test_event.notna()
    & pd.Series(test_prob, index=X_test.index).notna()
)

test_time_valid = test_time.loc[valid_surv_mask]
test_event_valid = test_event.loc[valid_surv_mask]
test_prob_valid = pd.Series(test_prob, index=X_test.index).loc[valid_surv_mask]

c_index = concordance_index(
    event_times=test_time_valid,
    predicted_scores=-test_prob_valid,
    event_observed=test_event_valid
)

print("C-index using DFS and MLP risk score:", c_index)

# RISK GROUPS FROM TRAIN/DEV

train_dev_prob = final_clf.predict_proba(X_train_dev)[:, 1]

low_cutoff = np.quantile(train_dev_prob, 1 / 3)
high_cutoff = np.quantile(train_dev_prob, 2 / 3)

print("Low cutoff:", low_cutoff)
print("High cutoff:", high_cutoff)


def assign_risk_group(prob):
    if prob <= low_cutoff:
        return "Low risk"
    elif prob <= high_cutoff:
        return "Intermediate risk"
    else:
        return "High risk"


test_risk_group = pd.Series(test_prob, index=X_test.index).apply(assign_risk_group)

km_data = pd.DataFrame({
    SURVIVAL_TIME_COL: test_time,
    "event": test_event,
    "risk_group": test_risk_group
}).dropna()

km_data.to_excel(
    "data/mlp/km_risk_group_data_mlp_v1.xlsx",
    index=False
)

print("Saved KM risk group data: data/mlp/km_risk_group_data_mlp_v1.xlsx")

plt.figure(figsize=(8, 6))

kmf = KaplanMeierFitter()

for group_name in ["Low risk", "Intermediate risk", "High risk"]:
    group_df = km_data[km_data["risk_group"] == group_name]

    if len(group_df) == 0:
        continue

    kmf.fit(
        durations=group_df[SURVIVAL_TIME_COL],
        event_observed=group_df["event"],
        label=group_name
    )
    kmf.plot_survival_function(ci_show=True)

plt.xlabel("Time after surgery")
plt.ylabel("Recurrence-free survival probability")
plt.title("Kaplan-Meier Curves by MLP Risk Group")
plt.tight_layout()
plt.savefig("data/mlp/km_curves_mlp_v1.png", dpi=300)
plt.close()

print("Saved KM curve: data/mlp/km_curves_mlp_v1.png")

logrank_result = multivariate_logrank_test(
    event_durations=km_data[SURVIVAL_TIME_COL],
    groups=km_data["risk_group"],
    event_observed=km_data["event"]
)

print("Log-rank p-value:", logrank_result.p_value)

train_time = pd.to_numeric(df.loc[X_train_dev.index, SURVIVAL_TIME_COL], errors="coerce")
train_event = y_train_dev.copy()

test_time = pd.to_numeric(df.loc[X_test.index, SURVIVAL_TIME_COL], errors="coerce")
test_event = y_test.copy()

train_surv_df = pd.DataFrame({
    "time": train_time,
    "event": train_event
}).dropna()

test_surv_df = pd.DataFrame({
    "time": test_time,
    "event": test_event,
    "risk_score": test_prob
}).dropna()

y_train_surv = Surv.from_arrays(
    event=train_surv_df["event"].astype(bool).values,
    time=train_surv_df["time"].astype(float).values
)

y_test_surv = Surv.from_arrays(
    event=test_surv_df["event"].astype(bool).values,
    time=test_surv_df["time"].astype(float).values
)

risk_scores_test = test_surv_df["risk_score"].values

if DFS_UNIT == "days":
    time_points = np.array([365, 730, 1095])
elif DFS_UNIT == "months":
    time_points = np.array([12, 24, 36])
else:
    raise ValueError("DFS_UNIT must be either 'days' or 'months'.")

max_test_time = test_surv_df["time"].max()
min_test_time = test_surv_df["time"].min()

time_points = time_points[
    (time_points > min_test_time)
    & (time_points < max_test_time)
]

if len(time_points) > 0:
    aucs, mean_auc = cumulative_dynamic_auc(
        y_train_surv,
        y_test_surv,
        risk_scores_test,
        time_points
    )

    td_auc_df = pd.DataFrame({
        "time": time_points,
        "time_dependent_auc": aucs
    })

    print("\nTime-dependent AUC:")
    print(td_auc_df)
    print("Mean time-dependent AUC:", mean_auc)

    td_auc_df.to_excel(
        "data/mlp/time_dependent_auc_mlp_v1.xlsx",
        index=False
    )
else:
    print("No valid time points for time-dependent AUC. Check DFS unit and follow-up range.")

# SAVE FINAL METRICS

final_metrics = pd.DataFrame({
    "metric": [
        "test_roc_auc",
        "test_pr_auc",
        "binary_brier_score",
        "c_index_DFS",
        "logrank_p_value",
        "sensitivity",
        "specificity"
    ],
    "value": [
        test_roc_auc,
        test_pr_auc,
        binary_brier,
        c_index,
        logrank_result.p_value,
        sensitivity,
        specificity
    ]
})

final_metrics.to_excel(
    "data/mlp/final_test_metrics_mlp_v1.xlsx",
    index=False
)

print("Saved final test metrics: data/mlp/final_test_metrics_mlp_v1.xlsx")

# SAVE TEST PREDICTIONS

test_predictions = X_test.copy()
test_predictions["true_label"] = y_test
test_predictions["recurrence_probability"] = test_prob.round(4)
test_predictions["prediction"] = test_pred
test_predictions["prediction_label"] = test_predictions["prediction"].map({
    1: "Có tái phát",
    0: "Không tái phát"
})
test_predictions["risk_group"] = test_risk_group

test_predictions = test_predictions.reset_index(drop=True)

test_predictions.to_excel(
    "data/mlp/test_predictions_mlp_v1.xlsx",
    index=False
)

# SAVE MODEL

joblib.dump(
    final_clf,
    "model/liver_recurrence_mlp_v1.joblib"
)

print("\nSaved model: model/liver_recurrence_mlp_v1.joblib")
print("Saved CV summary: data/mlp/cv_results_mlp_v1.xlsx")
print("Saved fold results: data/mlp/cv_fold_results_mlp_v1.xlsx")
print("Saved feature names: data/mlp/final_feature_names_after_preprocessing_mlp_v1.xlsx")
print("Saved test predictions: data/mlp/test_predictions_mlp_v1.xlsx")