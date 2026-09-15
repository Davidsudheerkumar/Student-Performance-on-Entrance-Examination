# Student Performance Prediction System (SPPS)

Machine Learning classification project for predicting student performance in an entrance examination using academic, demographic, coaching, and family-background information.

## Dataset
- File: `data/CEE_DATA.csv`
- Records: 666 candidates

## Execution Order

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run analysis:
   `python analysis.py`

3. Train ML models:
   `python model.py`

4. Start FastAPI:
   `python main.py`

5. Open API documentation:
   `http://localhost:8000/docs`

6. Start Streamlit:
   `streamlit run app.py`

## Models
- Logistic Regression
- Decision Tree
- Random Forest

The best model is selected using accuracy, with F1 score used as the tie-breaker, and saved as `models/performance_model.pkl`.
