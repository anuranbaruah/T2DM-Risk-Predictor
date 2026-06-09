# Dataset: diabetes_binary_health_indicators_BRFSS2015.csv
# kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# Web UI: streamlit run app.py

from t2dm_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(progress=print)
