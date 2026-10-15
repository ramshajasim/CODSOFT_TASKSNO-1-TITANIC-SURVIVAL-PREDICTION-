"""
Titanic Survival Prediction - Model Training Script
====================================================
Loads the Titanic dataset, cleans it, handles missing values, encodes
categorical features, trains a Random Forest classifier and saves the
full trained pipeline (preprocessor + model) as titanic_model.pkl.

Usage:
    python train_model.py [optional_path_to_csv]
"""

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "Titanic-Dataset.csv"
MODEL_OUTPUT_PATH = "titanic_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the Titanic CSV dataset into a DataFrame."""
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Dataset loaded successfully from '{path}'")
        print(f"[INFO] Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{path}'")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Clean & Preprocess Data
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns and perform basic cleaning."""
    df = df.copy()

    # Keep only the columns needed for modeling
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        print(f"[ERROR] Dataset is missing expected columns: {missing_cols}")
        sys.exit(1)

    df = df[FEATURES + [TARGET]].copy()

    # Standardize categorical text values
    df["Sex"] = df["Sex"].str.strip().str.lower()
    df["Embarked"] = df["Embarked"].str.strip().str.upper()

    # Rare/unknown embarked values -> most frequent (handled by imputer too)
    df.loc[~df["Embarked"].isin(["S", "C", "Q"]), "Embarked"] = np.nan

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        print(f"[INFO] Removed {before - after} duplicate rows")

    print(f"[INFO] Missing values before imputation:")
    print(df.isna().sum().to_string())

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Build Preprocessor + Model Pipeline
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """Build an sklearn Pipeline: preprocessing -> Random Forest."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ---------------------------------------------------------------------------
# 4. Train & Evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    """Train the pipeline and print evaluation metrics."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== MODEL PERFORMANCE (hold-out test set) =====")
    print(f"Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"F1 Score : {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Feature importances (post one-hot encoding)
    try:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        fi = (
            pd.Series(importances, index=feature_names)
            .sort_values(ascending=False)
            .head(10)
        )
        print("Top 10 Feature Importances:")
        print(fi.to_string())
    except Exception as e:
        print(f"[WARN] Could not extract feature importances: {e}")

    # Average survival probability sanity check
    print(f"\n[INFO] Mean predicted survival prob on test set: {y_proba.mean():.4f}")

    return pipeline, {"accuracy": round(acc, 4), "f1_score": round(f1, 4)}


# ---------------------------------------------------------------------------
# 5. Save Model
# ---------------------------------------------------------------------------
def save_model(pipeline, metrics: dict):
    """Save the trained pipeline and metadata with joblib."""
    bundle = {
        "pipeline": pipeline,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_OUTPUT_PATH)
    print(f"\n[INFO] Model saved successfully as '{MODEL_OUTPUT_PATH}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    print("=" * 60)
    print("  TITANIC SURVIVAL PREDICTION - TRAINING PIPELINE")
    print("=" * 60)

    raw_df = load_data(data_path)
    clean_df = clean_data(raw_df)
    model_pipeline, metrics = train_and_evaluate(clean_df)
    save_model(model_pipeline, metrics)

    print("\n[INFO] Done! Run the app with:  streamlit run app.py")
