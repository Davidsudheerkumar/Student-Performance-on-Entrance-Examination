import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Student Performance Prediction System",
    layout="wide",
    page_icon="🎓"
)


@st.cache_data
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

    categorical_columns = [
        "Gender",
        "Caste",
        "Coaching",
        "Class_ten_education",
        "twelve_education",
        "medium",
        "Father_occupation",
        "Mother_occupation"
    ]

    for column in categorical_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].mode()[0])

    df["Average_Academic_Percentage"] = (
        df["Class_X_Percentage"] +
        df["Class_XII_Percentage"]
    ) / 2

    df["Academic_Gap"] = (
        df["Class_XII_Percentage"] -
        df["Class_X_Percentage"]
    )

    df["Parent_Employment_Count"] = 2

    return df


df = load_data()

MODEL_PATH = "models/performance_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


st.title("🎓 Student Performance Prediction System")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    len(df)
)

col2.metric(
    "Average Class X %",
    f"{df['Class_X_Percentage'].mean():.2f}%"
)

col3.metric(
    "Average Class XII %",
    f"{df['Class_XII_Percentage'].mean():.2f}%"
)

col4.metric(
    "Average Academic %",
    f"{df['Average_Academic_Percentage'].mean():.2f}%"
)


st.subheader("📊 Performance Analysis")

left, right = st.columns(2)

with left:
    fig1 = px.scatter(
        df,
        x="Class_X_Percentage",
        y="Class_XII_Percentage",
        color="Performance",
        hover_data=[
            "Gender",
            "Coaching",
            "Class_X_Percentage",
            "Class_XII_Percentage",
            "Performance"
        ],
        title="Class X vs Class XII Performance"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with right:
    coaching_average = (
        df.groupby("Coaching")[
            "Average_Academic_Percentage"
        ]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        coaching_average,
        x="Coaching",
        y="Average_Academic_Percentage",
        color="Coaching",
        title="Average Academic Percentage by Coaching"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


st.subheader("📋 Student Analysis")

performance_options = ["All"] + sorted(
    df["Performance"].astype(str).unique().tolist()
)

selected_performance = st.selectbox(
    "Filter by Performance",
    performance_options
)

if selected_performance == "All":
    filtered_df = df
else:
    filtered_df = df[
        df["Performance"].astype(str) == selected_performance
    ]

display_columns = [
    "Gender",
    "Caste",
    "Coaching",
    "Class_X_Percentage",
    "Class_XII_Percentage",
    "Average_Academic_Percentage",
    "Performance"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True
)


st.subheader("🔮 Predict Student Performance")

if model is None:
    st.warning(
        "Model not found. Run `python model.py` first."
    )
else:
    with st.form("prediction_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox(
                "Gender",
                sorted(df["Gender"].astype(str).unique())
            )
            caste = st.selectbox(
                "Caste",
                sorted(df["Caste"].astype(str).unique())
            )
            coaching = st.selectbox(
                "Coaching",
                sorted(df["Coaching"].astype(str).unique())
            )
            class_ten_education = st.selectbox(
                "Class 10 Education",
                sorted(
                    df["Class_ten_education"]
                    .astype(str)
                    .unique()
                )
            )

        with c2:
            twelve_education = st.selectbox(
                "Class 12 Education",
                sorted(
                    df["twelve_education"]
                    .astype(str)
                    .unique()
                )
            )
            medium = st.selectbox(
                "Medium",
                sorted(df["medium"].astype(str).unique())
            )
            class_x = st.number_input(
                "Class X Percentage",
                min_value=0.0,
                max_value=100.0,
                value=70.0
            )
            class_xii = st.number_input(
                "Class XII Percentage",
                min_value=0.0,
                max_value=100.0,
                value=70.0
            )

        with c3:
            father_occupation = st.selectbox(
                "Father Occupation",
                sorted(
                    df["Father_occupation"]
                    .astype(str)
                    .unique()
                )
            )
            mother_occupation = st.selectbox(
                "Mother Occupation",
                sorted(
                    df["Mother_occupation"]
                    .astype(str)
                    .unique()
                )
            )

        submitted = st.form_submit_button(
            "Predict Performance"
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Caste": caste,
            "Coaching": coaching,
            "Class_ten_education": class_ten_education,
            "twelve_education": twelve_education,
            "medium": medium,
            "Class_X_Percentage": class_x,
            "Class_XII_Percentage": class_xii,
            "Father_occupation": father_occupation,
            "Mother_occupation": mother_occupation
        }])

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

        st.success(
            f"🎯 Predicted Performance: {prediction}"
        )
