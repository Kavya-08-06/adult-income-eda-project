"""
Case Study: Exploratory Data Analysis and Visualization of Adult Income
Dataset: UCI Adult / Census Income (14 input attributes + income target)
Run: python src/adult_income_project.py
The script downloads the public dataset, performs cleaning/EDA, trains an MLP,
and saves plots and metrics under outputs/. No results are hard-coded.
"""
from pathlib import Path
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

def download_file(url, destination):
    if not destination.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, destination)
    return destination

def load_dataset():
    train_path = download_file(TRAIN_URL, DATA_DIR / "adult.data")
    test_path = download_file(TEST_URL, DATA_DIR / "adult.test")
    train = pd.read_csv(train_path, names=COLUMNS, na_values="?", skipinitialspace=True)
    test = pd.read_csv(test_path, names=COLUMNS, na_values="?", skipinitialspace=True,
                       skiprows=1, skipinitialspace=True)
    df = pd.concat([train, test], ignore_index=True)
    df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"?": pd.NA, "": pd.NA, "nan": pd.NA})
    return df

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    sns.set_theme(style="whitegrid")
    df = load_dataset()
    print("\\nDATASET SHAPE:", df.shape)
    print("\\nFIRST ROWS\\n", df.head())
    print("\\nINFO\\n", df.info())
    print("\\nDESCRIPTIVE STATISTICS\\n", df.describe(include="all").T)
    print("\\nMISSING VALUES\\n", df.isna().sum().sort_values(ascending=False))
    print("Duplicate rows:", df.duplicated().sum())

    # Remove exact duplicate rows; imputation is performed inside the ML pipeline.
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(DATA_DIR / "adult_income_cleaned.csv", index=False)

    # Visual 1: target class distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="income", order=["<=50K", ">50K"])
    plt.title("Income Class Distribution")
    plt.xlabel("Annual income category"); plt.ylabel("Number of records")
    save_plot("01_income_class_distribution.png")

    # Visual 2: age distribution by income
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x="age", hue="income", bins=30, element="step",
                 stat="density", common_norm=False)
    plt.title("Age Distribution by Income Category")
    plt.xlabel("Age (years)"); plt.ylabel("Density")
    save_plot("02_age_distribution.png")

    # Visual 3: weekly hours boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="income", y="hours_per_week", order=["<=50K", ">50K"])
    plt.title("Weekly Working Hours by Income Category")
    plt.xlabel("Income category"); plt.ylabel("Hours worked per week")
    save_plot("03_hours_boxplot.png")

    # Visual 4: correlation heatmap for numerical fields
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True)
    plt.title("Correlation Heatmap of Numerical Features")
    save_plot("04_correlation_heatmap.png")

    # Visual 5: education category distribution
    education_order = df["education"].value_counts().index
    plt.figure(figsize=(11, 6))
    sns.countplot(data=df, y="education", order=education_order, color="steelblue")
    plt.title("Records by Education Level")
    plt.xlabel("Number of records"); plt.ylabel("Education level")
    save_plot("05_education_frequency.png")

    # Visual 6: age vs weekly hours, colored by target
    sample = df.sample(min(5000, len(df)), random_state=RANDOM_STATE)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=sample, x="age", y="hours_per_week", hue="income",
                    alpha=.45, s=22)
    plt.title("Age vs Weekly Working Hours (sample)")
    plt.xlabel("Age (years)"); plt.ylabel("Hours per week")
    save_plot("06_age_hours_scatter.png")

    # Distribution/statistical summaries for report evidence
    stats = df.groupby("income")[["age", "hours_per_week", "education_num",
                                  "capital_gain", "capital_loss"]].agg(["mean", "median", "std"])
    stats.to_csv(OUT / "grouped_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(OUT / "missing_values.csv")

    # Target encoding and split: hold out test data before fitting transformations.
    X = df.drop(columns=["income"])
    y = (df["income"] == ">50K").astype(int)
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical = X.select_dtypes(include=np.number).columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical)
    ])

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Basic hyperparameter comparison using validation set.
    candidates = [
        {"hidden_layer_sizes": (64, 32), "alpha": 0.0001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (100, 50), "alpha": 0.001, "learning_rate_init": 0.001},
        {"hidden_layer_sizes": (64, 32), "alpha": 0.001, "learning_rate_init": 0.0005}
    ]
    tuning_rows, best_model, best_score = [], None, -1
    for params in candidates:
        model = MLPClassifier(
            **params, activation="relu", solver="adam", batch_size=256,
            max_iter=35, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=5, random_state=RANDOM_STATE, verbose=False
        )
        model.fit(X_train_t, y_train)
        val_pred = model.predict(X_val_t)
        score = accuracy_score(y_val, val_pred)
        tuning_rows.append({**params, "validation_accuracy": score,
                            "epochs_run": model.n_iter_})
        if score > best_score:
            best_score, best_model = score, model
    pd.DataFrame(tuning_rows).to_csv(OUT / "hyperparameter_comparison.csv", index=False)

    # Evaluate once on the held-out test set.
    y_pred = best_model.predict(X_test_t)
    y_prob = best_model.predict_proba(X_test_t)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"], output_dict=True)
    pd.DataFrame(report).T.to_csv(OUT / "classification_report.csv")
    metrics = pd.DataFrame([{
        "test_accuracy": accuracy, "roc_auc": auc,
        "best_validation_accuracy": best_score,
        "train_rows": len(X_train), "validation_rows": len(X_val),
        "test_rows": len(X_test), "features_before_encoding": X.shape[1]
    }])
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    print("\\nTEST METRICS\\n", metrics.to_string(index=False))
    print("\\nCLASSIFICATION REPORT\\n", classification_report(y_test, y_pred,
          target_names=["<=50K", ">50K"]))

    # Result visual 1: confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["<=50K", ">50K"], cmap="Blues", values_format="d")
    plt.title("MLP Confusion Matrix (Test Set)")
    save_plot("07_confusion_matrix.png")

    # Result visual 2: ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.title("ROC Curve — Adult Income Classification")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_plot("08_roc_curve.png")

    # Result visual 3: training loss per epoch
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(best_model.loss_curve_) + 1), best_model.loss_curve_, marker="o")
    plt.title("MLP Training Loss vs Iteration")
    plt.xlabel("Iteration / epoch"); plt.ylabel("Training loss")
    save_plot("09_loss_curve.png")

    # Result visual 4: permutation feature importance on a manageable test sample.
    # Importance is computed on the transformed feature matrix.
    feat_names = preprocessor.get_feature_names_out()
    X_imp = X_test_t[:min(2500, len(X_test_t))]
    y_imp = y_test.iloc[:len(X_imp)]
    perm = permutation_importance(best_model, X_imp, y_imp, n_repeats=3,
                                  random_state=RANDOM_STATE, scoring="accuracy", n_jobs=-1)
    imp = pd.DataFrame({"feature": feat_names, "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std})
    imp = imp.sort_values("importance_mean", ascending=False).head(15)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=imp, x="importance_mean", y="feature", color="slateblue")
    plt.title("Top 15 Permutation Feature Importances")
    plt.xlabel("Mean decrease in accuracy after permutation"); plt.ylabel("Encoded feature")
    save_plot("10_feature_importance.png")

    joblib.dump({"preprocessor": preprocessor, "model": best_model},
                OUT / "adult_income_mlp.joblib")
    print(f"\\nFinished. Plots and metrics saved to: {OUT}")
    print("Note: results are generated from the dataset at runtime; use those actual values in your report.")

if __name__ == "__main__":
    main()
