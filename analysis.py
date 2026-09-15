import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def load_and_prepare_data(filepath):
    """Load the CEE dataset and prepare it for analysis."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    df["Class_X_Percentage"] = pd.to_numeric(
        df["Class_X_Percentage"], errors="coerce"
    )
    df["Class_XII_Percentage"] = pd.to_numeric(
        df["Class_XII_Percentage"], errors="coerce"
    )

    numerical_columns = [
        "Class_X_Percentage",
        "Class_XII_Percentage"
    ]

    for column in numerical_columns:
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

    df["Parent_Employment_Count"] = (
        df["Father_occupation"].notna().astype(int) +
        df["Mother_occupation"].notna().astype(int)
    )

    return df


def calculate_statistics(df):
    total_students = len(df)
    average_class_x = np.mean(df["Class_X_Percentage"])
    average_class_xii = np.mean(df["Class_XII_Percentage"])
    average_academic_percentage = np.mean(
        df["Average_Academic_Percentage"]
    )

    highest_class_x = np.max(df["Class_X_Percentage"])
    highest_class_xii = np.max(df["Class_XII_Percentage"])
    lowest_class_x = np.min(df["Class_X_Percentage"])
    lowest_class_xii = np.min(df["Class_XII_Percentage"])

    performance_distribution = df["Performance"].value_counts().to_dict()

    correlation_matrix = np.corrcoef(
        df["Class_X_Percentage"],
        df["Class_XII_Percentage"],
        df["Average_Academic_Percentage"]
    )

    return {
        "total_students": total_students,
        "average_class_x": average_class_x,
        "average_class_xii": average_class_xii,
        "average_academic_percentage": average_academic_percentage,
        "highest_class_x": highest_class_x,
        "highest_class_xii": highest_class_xii,
        "lowest_class_x": lowest_class_x,
        "lowest_class_xii": lowest_class_xii,
        "performance_distribution": performance_distribution,
        "correlation_matrix": correlation_matrix
    }


def generate_static_charts(df):
    os.makedirs("output", exist_ok=True)

    performance_counts = df["Performance"].value_counts()

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        performance_counts.index.astype(str),
        performance_counts.values
    )
    plt.title("Student Performance Distribution")
    plt.xlabel("Performance")
    plt.ylabel("Number of Students")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig("output/performance_distribution.png")
    plt.close()

    average_values = [
        df["Class_X_Percentage"].mean(),
        df["Class_XII_Percentage"].mean()
    ]

    labels = ["Class X", "Class XII"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, average_values)
    plt.title("Average Class X vs Class XII Percentage")
    plt.xlabel("Examination Level")
    plt.ylabel("Average Percentage")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig("output/academic_percentage_comparison.png")
    plt.close()


def generate_interactive_charts(df):
    fig1 = px.scatter(
        df,
        x="Class_X_Percentage",
        y="Class_XII_Percentage",
        color="Performance",
        hover_data=[
            "Gender",
            "Caste",
            "Coaching",
            "Class_X_Percentage",
            "Class_XII_Percentage",
            "Performance"
        ],
        title="Class X vs Class XII Percentage"
    )
    fig1.show()

    coaching_average = (
        df.groupby("Coaching")["Average_Academic_Percentage"]
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
    fig2.show()


def print_summary(stats):
    print("=" * 55)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 55)

    print(f"Total Students              : {stats['total_students']}")
    print(f"Average Class X             : {stats['average_class_x']:.2f}%")
    print(f"Average Class XII           : {stats['average_class_xii']:.2f}%")
    print(
        f"Average Academic Percentage : "
        f"{stats['average_academic_percentage']:.2f}%"
    )
    print(f"Highest Class X             : {stats['highest_class_x']:.2f}%")
    print(f"Highest Class XII           : {stats['highest_class_xii']:.2f}%")

    print("\nPerformance Distribution:")
    for category, count in stats["performance_distribution"].items():
        print(f"{str(category):25} : {count}")

    print("=" * 55)


if __name__ == "__main__":
    df = load_and_prepare_data("data/CEE_DATA.csv")
    stats = calculate_statistics(df)
    generate_static_charts(df)
    generate_interactive_charts(df)
    print_summary(stats)

    print("\nAnalysis complete. Charts saved to output/ folder.")
