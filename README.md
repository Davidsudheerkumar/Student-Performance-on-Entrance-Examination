# Student Performance Prediction System

This project implements the use case described in the supplied PRD: a machine-learning classification system that predicts a student's `Performance` in a medical/common entrance examination using academic, demographic, coaching, and family-background information.

## Project structure

```text
Student_Performance_Prediction/
├── app.py
├── model.py
├── analysis.py
├── requirements.txt
├── data/
│   └── CEE_DATA.csv
└── models/
    └── performance_model.pkl
```

## Run

```bash
pip install -r requirements.txt
python model.py
python analysis.py
streamlit run app.py
```

The supplied dataset contains 666 candidates. The actual CSV contains the additional `time` feature, so it is included in preprocessing and prediction along with the fields identified by the PRD.

## Notes

- Target variable: `Performance`
- Task: classification
- Preprocessing: categorical one-hot encoding
- Model: Random Forest classifier
- Train/test split: 80/20 with stratification
- Random seed: 42
