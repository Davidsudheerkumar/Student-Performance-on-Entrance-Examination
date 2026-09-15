import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


NUMERICAL_FEATURES = [
    "Class_X_Percentage",
    "Class_XII_Percentage",
    "Average_Academic_Percentage",
    "Academic_Gap",
    "Parent_Employment_Count",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Caste",
    "Coaching",
    "Class_ten_education",
    "twelve_education",
    "medium",
    "Father_occupation",
    "Mother_occupation",
]


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df["Class_X_Percentage"] = pd.to_numeric(
        df["Class_X_Percentage"], errors="coerce"
    )
    df["Class_XII_Percentage"] = pd.to_numeric(
        df["Class_XII_Percentage"], errors="coerce"
    )

    for column in ["Class_X_Percentage", "Class_XII_Percentage"]:
        df[column] = df[column].fillna(df[column].median())

    for column in CATEGORICAL_FEATURES:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].mode()[0])

    df["Average_Academic_Percentage"] = (
        df["Class_X_Percentage"] + df["Class_XII_Percentage"]
    ) / 2

    df["Academic_Gap"] = (
        df["Class_XII_Percentage"] - df["Class_X_Percentage"]
    )

    # Both occupation fields are present after missing-value handling.
    df["Parent_Employment_Count"] = (
        df["Father_occupation"].notna().astype(int)
        + df["Mother_occupation"].notna().astype(int)
    )

    return df


def prepare_ml_data(df):
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df["Performance"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


def create_preprocessor():
    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def train_models(X_train, y_train, preprocessor):
    model_definitions = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
    }

    models = {}

    for name, classifier in model_definitions.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        models[name] = pipeline

    return models


def evaluate_models(models, X_test, y_test):
    results = {}

    print("=" * 75)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 75)
    print(
        f"{'Model':22}"
        f"{'Accuracy':12}"
        f"{'Precision':12}"
        f"{'Recall':12}"
        f"{'F1 Score':12}"
    )

    for name, model in models.items():
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        print(
            f"{name:22}"
            f"{accuracy:12.4f}"
            f"{precision:12.4f}"
            f"{recall:12.4f}"
            f"{f1:12.4f}"
        )

    print("=" * 75)

    return results


def save_best_model(models, evaluation_results):
    best_model_name = max(
        evaluation_results,
        key=lambda name: (
            evaluation_results[name]["accuracy"],
            evaluation_results[name]["f1_score"]
        )
    )

    best_model = models[best_model_name]

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        "models/performance_model.pkl"
    )

    return best_model_name, best_model


if __name__ == "__main__":
    print("Loading dataset...")

    df = load_and_prepare_data("data/CEE_DATA.csv")

    print(f"Dataset loaded: {len(df)} students")

    X_train, X_test, y_train, y_test = prepare_ml_data(df)

    print("Train/Test split completed.")

    preprocessor = create_preprocessor()

    print("Training classification models...")

    models = train_models(
        X_train,
        y_train,
        preprocessor
    )

    evaluation_results = evaluate_models(
        models,
        X_test,
        y_test
    )

    best_model_name, best_model = save_best_model(
        models,
        evaluation_results
    )

    print(f"Best Model: {best_model_name}")
    print(
        "Model saved to "
        "models/performance_model.pkl"
    )
    print("Model training complete.")
