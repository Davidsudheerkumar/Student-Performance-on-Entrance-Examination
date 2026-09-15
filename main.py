import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Student Performance Prediction System API",
    description=(
        "API for analyzing and predicting student entrance "
        "examination performance"
    ),
    version="1.0.0"
)


CATEGORICAL_COLUMNS = [
    "Gender",
    "Caste",
    "Coaching",
    "Class_ten_education",
    "twelve_education",
    "medium",
    "Father_occupation",
    "Mother_occupation"
]


def load_data():
    df = pd.read_csv("data/CEE_DATA.csv")
    df.columns = df.columns.str.strip()

    df["Class_X_Percentage"] = pd.to_numeric(
        df["Class_X_Percentage"], errors="coerce"
    )
    df["Class_XII_Percentage"] = pd.to_numeric(
        df["Class_XII_Percentage"], errors="coerce"
    )

    for column in ["Class_X_Percentage", "Class_XII_Percentage"]:
        df[column] = df[column].fillna(df[column].median())

    for column in CATEGORICAL_COLUMNS:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].mode()[0])

    df["Average_Academic_Percentage"] = (
        df["Class_X_Percentage"] + df["Class_XII_Percentage"]
    ) / 2

    df["Academic_Gap"] = (
        df["Class_XII_Percentage"] - df["Class_X_Percentage"]
    )

    df["Parent_Employment_Count"] = (
        df["Father_occupation"].notna().astype(int)
        + df["Mother_occupation"].notna().astype(int)
    )

    return df


df = load_data()

MODEL_PATH = "models/performance_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


class StudentInput(BaseModel):
    Gender: str
    Caste: str
    Coaching: str
    Class_ten_education: str
    twelve_education: str
    medium: str
    Class_X_Percentage: float = Field(
        ..., ge=0, le=100,
        description="Class X percentage must be between 0 and 100"
    )
    Class_XII_Percentage: float = Field(
        ..., ge=0, le=100,
        description="Class XII percentage must be between 0 and 100"
    )
    Father_occupation: str
    Mother_occupation: str


@app.get("/")
def root():
    return {
        "message": "Student Performance Prediction System API",
        "docs": "Visit /docs for full API documentation",
        "version": "1.0.0"
    }


@app.get("/summary")
def summary():
    distribution = df["Performance"].value_counts().to_dict()

    return {
        "total_students": int(len(df)),
        "average_class_x": round(
            float(df["Class_X_Percentage"].mean()), 2
        ),
        "average_class_xii": round(
            float(df["Class_XII_Percentage"].mean()), 2
        ),
        "average_academic_percentage": round(
            float(df["Average_Academic_Percentage"].mean()), 2
        ),
        "performance_distribution": {
            str(key): int(value)
            for key, value in distribution.items()
        }
    }


@app.get("/students")
def students():
    columns = [
        "Gender",
        "Caste",
        "Coaching",
        "Class_ten_education",
        "twelve_education",
        "medium",
        "Class_X_Percentage",
        "Class_XII_Percentage",
        "Performance"
    ]

    records = df[columns].copy()
    return records.to_dict(orient="records")


@app.get("/top-students")
def top_students():
    records = (
        df.sort_values(
            "Average_Academic_Percentage",
            ascending=False
        )
        .head(5)
        .reset_index()
    )

    result = []

    for index, row in records.iterrows():
        result.append({
            "student_index": int(row["index"]),
            "Class_X_Percentage": float(row["Class_X_Percentage"]),
            "Class_XII_Percentage": float(row["Class_XII_Percentage"]),
            "Average_Academic_Percentage": float(
                row["Average_Academic_Percentage"]
            ),
            "Performance": str(row["Performance"])
        })

    return result


@app.post("/predict")
def predict(student: StudentInput):
    if model is None:
        return {
            "error": (
                "Trained model not found. "
                "Run model.py first."
            )
        }

    input_data = pd.DataFrame([student.model_dump()])

    input_data["Average_Academic_Percentage"] = (
        input_data["Class_X_Percentage"]
        + input_data["Class_XII_Percentage"]
    ) / 2

    input_data["Academic_Gap"] = (
        input_data["Class_XII_Percentage"]
        - input_data["Class_X_Percentage"]
    )

    input_data["Parent_Employment_Count"] = 2

    prediction = model.predict(input_data)[0]

    return {
        "predicted_performance": str(prediction),
        "message": "Prediction generated successfully"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
