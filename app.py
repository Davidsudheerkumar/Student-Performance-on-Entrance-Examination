"""
Student Performance Prediction System
app.py - Streamlit user interface
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "CEE_DATA.csv"
MODEL_PATH = BASE_DIR / "models" / "performance_model.pkl"

FEATURES = [
    "Gender", "Caste", "coaching", "time",
    "Class_ten_education", "twelve_education", "medium",
    "Class_ X_Percentage", "Class_XII_Percentage",
    "Father_occupation", "Mother_occupation"
]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"')
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction System")
st.write("Predict a student's entrance-examination performance using the supplied academic, demographic, coaching, and family-background data.")

df = load_data()
model = load_model()

st.sidebar.header("Student Information")

def selectbox_for(column, label=None):
    values = sorted(df[column].dropna().astype(str).unique())
    return st.sidebar.selectbox(label or column, values)

student = {}
for col in FEATURES:
    student[col] = selectbox_for(col)

if st.sidebar.button("Predict Performance", type="primary"):
    input_df = pd.DataFrame([student])
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction")
    st.success(f"Predicted Performance: **{prediction}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        prob_df = pd.DataFrame({"Performance": classes, "Probability": probabilities})
        prob_df = prob_df.sort_values("Probability", ascending=False)
        st.subheader("Prediction Confidence")
        st.dataframe(prob_df, use_container_width=True)

st.divider()
st.subheader("Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Candidates", len(df))
c2.metric("Features", len(FEATURES))
c3.metric("Performance Classes", df["Performance"].nunique())

with st.expander("View Performance Distribution"):
    st.bar_chart(df["Performance"].value_counts())

with st.expander("View Sample Data"):
    st.dataframe(df.head(20), use_container_width=True)
