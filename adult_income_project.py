
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, roc_curve, roc_auc_score
)
from sklearn.inspection import permutation_importance
from scipy.sparse import issparse
import joblib

# Create output folder
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

df = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)
df["income"] = df["income"].str.strip()
df = df.drop_duplicates().reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescriptive statistics:\n", df.describe(include="all").T)

# EDA visualizations
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="income")
plt.title("Income Class Distribution")
plt.savefig(OUT / "01_income_distribution.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig(OUT / "02_age_distribution.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="income", y="hours_per_week")
plt.title("Weekly Hours by Income")
plt.savefig(OUT / "03_hours_boxplot.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
sns.heatmap(df.select_dtypes(include="number").corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numerical Correlation Heatmap")
plt.savefig(OUT / "04_correlation_heatmap.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
sns.countplot(data=df, y="education",
              order=df["education"].value_counts().index)
plt.title("Education Distribution")
plt.savefig(OUT / "05_education_distribution.png", bbox_inches="tight")
plt.show()

sample = df.sample(min(3000, len(df)), random_state=42)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=sample, x="age", y="hours_per_week",
                hue="income", alpha=0.5)
plt.title("Age vs Weekly Working Hours")
plt.savefig(OUT / "06_age_hours_scatter.png", bbox_inches="tight")
plt.show()

# Prepare features and target
X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
])

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)

# Basic hyperparameter comparison
tuning_results = []
best_model = None
best_score = -1

for layers, alpha in [((32, 16), 0.0001),
                      ((64, 32), 0.001),
                      ((100, 50), 0.001)]:
    candidate = MLPClassifier(
        hidden_layer_sizes=layers, alpha=alpha,
        activation="relu", solver="adam", max_iter=35,
        early_stopping=True, random_state=42
    )
    candidate.fit(X_train_t, y_train)
    score = accuracy_score(y_val, candidate.predict(X_val_t))
    tuning_results.append({
        "hidden_layers": str(layers),
        "alpha": alpha,
        "validation_accuracy": score
    })
    if score > best_score:
        best_score = score
        best_model = candidate

pd.DataFrame(tuning_results).to_csv(
    OUT / "hyperparameter_comparison.csv", index=False
)

# Test evaluation
y_pred = best_model.predict(X_test_t)
y_prob = best_model.predict_proba(X_test_t)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Test accuracy:", accuracy)
print("ROC-AUC:", auc)
print(classification_report(y_test, y_pred))

pd.DataFrame([{
    "test_accuracy": accuracy,
    "roc_auc": auc,
    "validation_accuracy": best_score
}]).to_csv(OUT / "model_metrics.csv", index=False)

pd.DataFrame(classification_report(
    y_test, y_pred,
    target_names=["<=50K", ">50K"],
    output_dict=True
)).T.to_csv(OUT / "classification_report.csv")

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["<=50K", ">50K"], cmap="Blues"
)
plt.title("MLP Confusion Matrix")
plt.savefig(OUT / "07_confusion_matrix.png", bbox_inches="tight")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(OUT / "08_roc_curve.png", bbox_inches="tight")
plt.show()

# Training loss
plt.figure(figsize=(7, 5))
plt.plot(best_model.loss_curve_)
plt.title("Training Loss vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(OUT / "09_training_loss.png", bbox_inches="tight")
plt.show()

# Permutation feature importance
X_imp = X_test_t[:min(1000, X_test_t.shape[0])]
y_imp = y_test.iloc[:X_imp.shape[0]]
if issparse(X_imp):
    X_imp = X_imp.toarray()

importance = permutation_importance(
    best_model, X_imp, y_imp, n_repeats=3,
    random_state=42, scoring="accuracy", n_jobs=-1
)
importance_df = pd.DataFrame({
    "feature": preprocessor.get_feature_names_out(),
    "importance": importance.importances_mean
}).sort_values("importance", ascending=False)

importance_df.to_csv(OUT / "feature_importance.csv", index=False)

joblib.dump(
    {"preprocessor": preprocessor, "model": best_model},
    OUT / "adult_income_mlp.joblib"
)
df.to_csv(OUT / "adult_income_cleaned.csv", index=False)

print("Project completed. All results saved in the outputs folder.")
