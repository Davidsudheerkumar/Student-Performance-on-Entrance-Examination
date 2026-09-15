"""
Student Performance Prediction System
analysis.py - exploratory analysis and model evaluation
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from model import load_data, FEATURES, TARGET, build_pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "performance_model.pkl"


def run_analysis():
    df = load_data()

    print("=" * 70)
    print("STUDENT PERFORMANCE DATA ANALYSIS")
    print("=" * 70)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nPerformance distribution:")
    print(df[TARGET].value_counts())
    print("\nCategorical feature summaries:")
    for col in FEATURES:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))

    # Model evaluation using the same pipeline defined in model.py.
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET],
        test_size=0.20, random_state=42, stratify=df[TARGET]
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    print("\nModel accuracy:", round(accuracy_score(y_test, pred), 4))
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, pred))

    # Performance distribution chart.
    plt.figure(figsize=(8, 5))
    df[TARGET].value_counts().plot(kind="bar")
    plt.title("Student Performance Distribution")
    plt.xlabel("Performance")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "performance_distribution.png", dpi=150)
    plt.close()

    return df


if __name__ == "__main__":
    run_analysis()
