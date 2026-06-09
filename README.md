# T2DM Risk Predictor

Early prediction of Type 2 diabetes using an ensemble of Random Forest, XGBoost, and LightGBM, with SHAP, LIME, and counterfactual explanations. Includes a Streamlit web app for interactive risk estimation.

## Dataset

[CDC BRFSS 2015 Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) — 253,680 rows, 21 features covering demographics, lifestyle, and medical history.

Place `diabetes_binary_health_indicators_BRFSS2015.csv` in the project root before running.

## Setup

```bash
pip install -r requirements.txt
```

GPU acceleration is optional. XGBoost and LightGBM will probe for CUDA and fall back to CPU automatically.

## Usage

**Web UI**
```bash
streamlit run app.py
```

**CLI**
```bash
python ML_project.py
```

The pipeline runs Optuna hyperparameter tuning (30 trials per model by default), trains all models, evaluates on a held-out test set, and generates figures and CSVs under the output folder.

Use **Quick mode** in the sidebar (or set `n_trials=5`) for a faster run during development.

## Models

| Model | Notes |
|---|---|
| Random Forest | scikit-learn, CPU |
| XGBoost | CUDA / CPU, monotone constraints |
| LightGBM | CUDA / CPU, monotone constraints |
| Soft Voting | Average predicted probabilities |
| Stacking | LR meta-learner over 5-fold OOF predictions |

Monotone constraints encode clinical expectations — e.g. higher BMI and age can only increase predicted risk.

## Explainability

- **SHAP** (TreeExplainer on LightGBM) — global feature importance and stability across folds
- **LIME** — local explanation for individual predictions
- **DiCE** — counterfactual examples showing what would lower a high-risk prediction

## Outputs

Each run writes to the chosen output folder:

```
model_comparison_results.csv
shap_feature_importance.csv
fig1_shap_summary.png
fig2_shap_bar.png
fig3_lime_explanation.png
fig4_confusion_matrix.png
fig5_roc_curves.png
fig6_model_comparison.png
fig7_correlation_heatmap.png
```

## External Validation

Optionally tests on the [Pima Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (`diabetes.csv`) to check out-of-distribution generalisation.

## Disclaimer

For research and educational purposes only. Not a medical device.
