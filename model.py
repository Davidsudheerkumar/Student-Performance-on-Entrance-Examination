"""
Student Performance Prediction System
model.py - training and prediction pipeline
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "CEE_DATA.csv"
MODEL_PATH = BASE_DIR / "models" / "performance_model.pkl"

TARGET = "Performance"

FEATURES = [
    "Gender", "Caste", "coaching", "time",
    "Class_ten_education", "twelve_education", "medium",
    "Class_ X_Percentage", "Class_XII_Percentage",
    "Father_occupation", "Mother_occupation"
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"')
    return df


def build_pipeline():
    categorical_features = FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_model():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {sorted(y.unique())}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return pipeline, accuracy


def predict_student(student_data, pipeline=None):
    if pipeline is None:
        if not MODEL_PATH.exists():
            pipeline, _ = train_model()
        else:
            pipeline = joblib.load(MODEL_PATH)

    row = pd.DataFrame([student_data], columns=FEATURES)
    return pipeline.predict(row)[0]


if __name__ == "__main__":
    train_model()
