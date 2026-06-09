# T2DM risk prediction — training, evaluation, and explainability pipeline.

from __future__ import annotations

import os
import warnings
from typing import Any, Callable, Literal, Optional, Union

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

import dice_ml

from gpu_config import (
    GPUCapabilities,
    detect_gpu_capabilities,
    lgbm_classifier_kwargs,
    xgb_classifier_kwargs,
)

TARGET = "Diabetes_binary"


def _log(progress: Optional[Callable[[str], None]], msg: str) -> None:
    if progress:
        progress(msg)
    else:
        print(msg)


def evaluate(name: str, model: Any, X_ev: pd.DataFrame, y_ev: pd.Series, proba: bool = True) -> dict:
    y_pred = model.predict(X_ev)
    y_prob = model.predict_proba(X_ev)[:, 1] if proba else None
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_ev, y_pred), 3),
        "Precision": round(precision_score(y_ev, y_pred), 3),
        "Recall": round(recall_score(y_ev, y_pred), 3),
        "F1": round(f1_score(y_ev, y_pred), 3),
        "AUC-ROC": round(roc_auc_score(y_ev, y_prob), 3) if proba else "—",
        "MCC": round(matthews_corrcoef(y_ev, y_pred), 3),
    }


def run_pipeline(
    main_csv: str = "diabetes_binary_health_indicators_BRFSS2015.csv",
    pidd_csv: Optional[str] = "diabetes.csv",
    n_trials: int = 30,
    n_shap_sample: int = 500,
    lime_sample_count: int = 5,
    dice_high_risk_count: int = 3,
    output_dir: str = ".",
    run_shap_stability: bool = True,
    use_gpu: Union[bool, Literal["auto"]] = "auto",
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """
    Full training, evaluation, and explanation pipeline.
    Returns dict with models, scaler, results, file paths, and training stats for UI prediction.

    use_gpu: "auto" or True probes CUDA for XGBoost / LightGBM; False forces CPU tree backends.
    RandomForest always trains on CPU (scikit-learn).
    """
    os.makedirs(output_dir, exist_ok=True)

    caps: GPUCapabilities = detect_gpu_capabilities(False if use_gpu is False else "auto")
    xgb_kw = xgb_classifier_kwargs(caps)
    lgbm_kw = lgbm_classifier_kwargs(caps)
    _log(progress, f"Compute: {caps.message}")

    _log(progress, "Loading data…")
    df = pd.read_csv(main_csv)
    features_all = [c for c in df.columns if c != TARGET]
    X = df[features_all].copy()
    y = df[TARGET].copy()

    clip_bounds: dict[str, tuple[float, float]] = {}
    for col in ["BMI", "MentHlth", "PhysHlth"]:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        clip_bounds[col] = (lo, hi)
        X[col] = X[col].clip(lo, hi)

    corr = X.corrwith(y).abs()
    selected_features = corr[corr >= 0.05].index.tolist()
    X = X[selected_features]

    # +1: higher value → higher risk, -1: higher value → lower risk
    _MONO_POS = {"GenHlth", "BMI", "Age", "HighBP", "HighChol", "MentHlth", "PhysHlth",
                 "DiffWalk", "Stroke", "HeartDiseaseorAttack", "NoDocbcCost",
                 "Smoker", "HvyAlcoholConsump"}
    _MONO_NEG = {"AnyHealthcare", "PhysActivity", "Fruits", "Veggies", "Income", "Education"}
    _mono_vec = tuple(
        1 if f in _MONO_POS else -1 if f in _MONO_NEG else 0
        for f in selected_features
    )
    lgbm_mono = list(_mono_vec)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    y_train_sm = pd.Series(y_train_sm)

    scaler = MinMaxScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_sm), columns=selected_features)
    X_val_sc = pd.DataFrame(scaler.transform(X_val), columns=selected_features)
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

    train_medians = X_train.median()

    _log(progress, f"Tuning models ({n_trials} Optuna trials each)…")

    def rf_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        return cross_val_score(clf, X_train_sc, y_train_sm, cv=3, scoring="roc_auc", n_jobs=-1).mean()

    rf_study = optuna.create_study(direction="maximize")
    def _rf_trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        _log(progress, f"TRIAL:RF:{trial.number + 1}/{n_trials}")
    rf_study.optimize(rf_objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_rf_trial_cb])
    best_rf_params = rf_study.best_params
    best_rf_params["random_state"] = 42
    best_rf_params["n_jobs"] = -1

    def xgb_objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "monotone_constraints": _mono_vec,
        }
        params.update(xgb_kw)
        clf = XGBClassifier(**params)
        return cross_val_score(
            clf, X_train_sc, y_train_sm, cv=3, scoring="roc_auc", n_jobs=caps.tree_cv_n_jobs
        ).mean()

    xgb_study = optuna.create_study(direction="maximize")
    def _xgb_trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        _log(progress, f"TRIAL:XGB:{trial.number + 1}/{n_trials}")
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_xgb_trial_cb])
    best_xgb_params = dict(xgb_study.best_params)
    best_xgb_params.update({"eval_metric": "logloss", "random_state": 42})
    best_xgb_params.update(xgb_kw)
    best_xgb_params["monotone_constraints"] = _mono_vec

    def lgbm_objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "random_state": 42,
            "verbose": -1,
            "monotone_constraints": lgbm_mono,
        }
        params.update(lgbm_kw)
        clf = LGBMClassifier(**params)
        return cross_val_score(
            clf, X_train_sc, y_train_sm, cv=3, scoring="roc_auc", n_jobs=caps.tree_cv_n_jobs
        ).mean()

    lgbm_study = optuna.create_study(direction="maximize")
    def _lgbm_trial_cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        _log(progress, f"TRIAL:LGBM:{trial.number + 1}/{n_trials}")
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_lgbm_trial_cb])
    best_lgbm_params = dict(lgbm_study.best_params)
    best_lgbm_params.update({"random_state": 42, "verbose": -1})
    best_lgbm_params.update(lgbm_kw)
    best_lgbm_params["monotone_constraints"] = lgbm_mono

    _log(progress, "Training base learners & ensembles…")
    rf_model = RandomForestClassifier(**best_rf_params)
    xgb_model = XGBClassifier(**best_xgb_params)
    lgbm_model = LGBMClassifier(**best_lgbm_params)

    rf_model.fit(X_train_sc, y_train_sm)
    xgb_model.fit(X_train_sc, y_train_sm, eval_set=[(X_val_sc, y_val)], verbose=False)
    lgbm_model.fit(X_train_sc, y_train_sm, eval_set=[(X_val_sc, y_val)], callbacks=[])
    _log(progress, "Base learners trained — building ensembles…")

    base_results = []
    for name, model in [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
        ("LightGBM", lgbm_model),
    ]:
        base_results.append(evaluate(name, model, X_test_sc, y_test))

    voting = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model), ("lgbm", lgbm_model)],
        voting="soft",
        n_jobs=caps.ensemble_n_jobs,
    )
    voting.fit(X_train_sc, y_train_sm)

    # class_weight='balanced': SMOTE only rebalances base learner inputs, not the stacking OOF labels
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    stacking = StackingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model), ("lgbm", lgbm_model)],
        final_estimator=meta_lr,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=False,
        n_jobs=caps.ensemble_n_jobs,
    )
    stacking.fit(X_train_sc, y_train_sm)

    ensemble_results = []
    for name, model in [("Soft Voting", voting), ("Stacking", stacking)]:
        ensemble_results.append(evaluate(name, model, X_test_sc, y_test))

    all_results = base_results + ensemble_results
    all_df = pd.DataFrame(all_results)

    auc_col = all_df["AUC-ROC"].apply(lambda v: float(v) if v != "—" else -1.0)
    best_model_name = str(all_df.loc[auc_col.idxmax(), "Model"])
    all_df["Model"] = all_df["Model"].apply(
        lambda n: f"{n} (Best)" if n == best_model_name else n
    )

    out_csv = os.path.join(output_dir, "model_comparison_results.csv")
    all_df.to_csv(out_csv, index=False)

    pidd_df = None
    if pidd_csv:
        try:
            pidd = pd.read_csv(pidd_csv)
            pidd_features = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]
            for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
                pidd[col] = pidd[col].replace(0, np.nan)
                pidd[col] = pidd[col].fillna(pidd[col].median())

            X_pidd = pidd[pidd_features]
            y_pidd = pidd["Outcome"]
            X_pidd_aligned = pd.DataFrame(0, index=X_pidd.index, columns=selected_features)
            for col in pidd_features:
                if col in selected_features:
                    X_pidd_aligned[col] = X_pidd[col].values
            X_pidd_sc = pd.DataFrame(scaler.transform(X_pidd_aligned), columns=selected_features)

            pidd_results = []
            for name, model in [
                ("Soft Voting (PIDD)", voting),
                ("Stacking (PIDD, no retrain)", stacking),
            ]:
                pidd_results.append(evaluate(name, model, X_pidd_sc, y_pidd))
            pidd_df = pd.DataFrame(pidd_results)
            pidd_df.to_csv(os.path.join(output_dir, "pidd_external_validation.csv"), index=False)
        except FileNotFoundError:
            pidd_df = None

    _log(progress, "SHAP & LIME…")
    n_shap = min(n_shap_sample, len(X_test_sc))
    X_shap = X_test_sc.sample(n=n_shap, random_state=42)
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_shap)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_shap = pd.Series(np.abs(sv).mean(axis=0), index=selected_features).sort_values(ascending=False)
    mean_shap.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"))

    plt.figure()
    shap.summary_plot(sv, X_shap, feature_names=selected_features, show=False, plot_size=(10, 7))
    plt.title("SHAP Summary Plot (LightGBM)")
    plt.tight_layout()
    fig1 = os.path.join(output_dir, "fig1_shap_summary.png")
    plt.savefig(fig1, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    mean_shap.head(15).plot(kind="barh", color="#2c3e50")
    plt.xlabel("Mean |SHAP Value|")
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig2 = os.path.join(output_dir, "fig2_shap_bar.png")
    plt.savefig(fig2, dpi=150, bbox_inches="tight")
    plt.close()

    mean_spearman_rho = None
    if run_shap_stability:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_importances = []
        for tr, te in skf.split(X_train_sc, y_train_sm):
            fold_model = LGBMClassifier(**best_lgbm_params)
            fold_model.fit(X_train_sc.iloc[tr], y_train_sm.iloc[tr])
            exp_f = shap.TreeExplainer(fold_model)
            sv_fold = exp_f.shap_values(X_train_sc.iloc[te[:200]])
            sv_f = sv_fold[1] if isinstance(sv_fold, list) else sv_fold
            fold_importances.append(np.abs(sv_f).mean(axis=0))
        corr_matrix = []
        for i in range(5):
            for j in range(i + 1, 5):
                rho, _ = spearmanr(fold_importances[i], fold_importances[j])
                corr_matrix.append(rho)
        mean_spearman_rho = float(np.mean(corr_matrix))

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_sc.values,
        feature_names=selected_features,
        class_names=["Non-Diabetic", "Diabetic"],
        mode="classification",
        random_state=42,
    )
    sample_indices = X_test_sc.sample(min(lime_sample_count, len(X_test_sc)), random_state=42).index
    fidelity_scores = []
    for idx in sample_indices:
        instance = X_test_sc.loc[idx].values
        exp = lime_explainer.explain_instance(
            instance, stacking.predict_proba, num_features=len(selected_features), num_samples=1000
        )
        fidelity_scores.append(exp.score)

    exp_plot = lime_explainer.explain_instance(
        X_test_sc.iloc[0].values, stacking.predict_proba, num_features=10
    )
    fig = exp_plot.as_pyplot_figure()
    plt.title("LIME Local Explanation — First Test Instance")
    plt.tight_layout()
    fig3 = os.path.join(output_dir, "fig3_lime_explanation.png")
    plt.savefig(fig3, dpi=150, bbox_inches="tight")
    plt.close()

    mean_lime_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else None

    _log(progress, "DiCE counterfactuals…")
    train_data_dice = X_train_sc.copy()
    train_data_dice[TARGET] = y_train_sm.values
    d = dice_ml.Data(
        dataframe=train_data_dice,
        continuous_features=selected_features,
        outcome_name=TARGET,
    )
    m = dice_ml.Model(model=stacking, backend="sklearn")
    exp_dice = dice_ml.Dice(d, m, method="random")
    high_risk = X_test_sc[stacking.predict(X_test_sc) == 1].head(dice_high_risk_count)
    dice_validity_total = 0
    dice_validity_max = len(high_risk) * 3
    for _, row in high_risk.iterrows():
        query = row.to_frame().T
        try:
            cf = exp_dice.generate_counterfactuals(
                query, total_CFs=3, desired_class="opposite", verbose=False
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                dice_validity_total += int((cf_df[TARGET] == 0).sum())
        except Exception:
            pass

    _log(progress, "Saving figures…")
    y_pred_stack = stacking.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred_stack)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Non-Diabetic", "Diabetic"]).plot(
        ax=ax, colorbar=False, cmap="Greys"
    )
    ax.set_title("Confusion Matrix — Stacking (Test)")
    plt.tight_layout()
    fig4 = os.path.join(output_dir, "fig4_confusion_matrix.png")
    plt.savefig(fig4, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
        ("LightGBM", lgbm_model),
        ("Soft Voting", voting),
        ("Stacking", stacking),
    ]:
        proba = model.predict_proba(X_test_sc)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        # Highlight the best model in the ROC plot
        display_name = f"{name} (Best)" if name == best_model_name else name
        ls = "-" if name == best_model_name else "--"
        lw = 2.5 if name == best_model_name else 1.5
        ax.plot(fpr, tpr, linestyle=ls, linewidth=lw, label=f"{display_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig5 = os.path.join(output_dir, "fig5_roc_curves.png")
    plt.savefig(fig5, dpi=150, bbox_inches="tight")
    plt.close()

    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    all_df_num = all_df.set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.15
    all_vals = []
    for i, model_name in enumerate(all_df_num.index):
        vals = [float(all_df_num.loc[model_name, m]) for m in metrics]
        all_vals.extend(vals)
        ax.bar(x + i * width, vals, width, label=model_name)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metrics)
    y_floor = max(0.0, min(all_vals) - 0.05)
    ax.set_ylim(y_floor, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    fig6 = os.path.join(output_dir, "fig6_model_comparison.png")
    plt.savefig(fig6, dpi=150, bbox_inches="tight")
    plt.close()

    top15 = mean_shap.head(15).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X_test_sc[top15].corr(), annot=True, fmt=".2f", cmap="Greys", linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation (Top 15 SHAP Features)")
    plt.tight_layout()
    fig7 = os.path.join(output_dir, "fig7_correlation_heatmap.png")
    plt.savefig(fig7, dpi=150, bbox_inches="tight")
    plt.close()

    feature_ranges = pd.DataFrame(
        {
            "min": X_train[selected_features].min(),
            "max": X_train[selected_features].max(),
            "median": train_medians.reindex(selected_features),
        }
    )

    output_files = [
        out_csv,
        fig1,
        fig2,
        fig3,
        fig4,
        fig5,
        fig6,
        fig7,
        os.path.join(output_dir, "shap_feature_importance.csv"),
    ]
    if pidd_df is not None:
        output_files.append(os.path.join(output_dir, "pidd_external_validation.csv"))

    return {
        "all_results_df": all_df,
        "pidd_df": pidd_df,
        "selected_features": selected_features,
        "clip_bounds": clip_bounds,
        "scaler": scaler,
        "train_medians": train_medians,
        "feature_ranges": feature_ranges,
        "models": {
            "rf": rf_model,
            "xgb": xgb_model,
            "lgbm": lgbm_model,
            "voting": voting,
            "stacking": stacking,
        },
        "best_params": {
            "rf": best_rf_params,
            "xgb": best_xgb_params,
            "lgbm": best_lgbm_params,
        },
        "X_test_sc": X_test_sc,
        "y_test": y_test,
        "mean_shap": mean_shap,
        "explainer_lgbm": explainer,
        "lime_explainer": lime_explainer,
        "mean_spearman_shap": mean_spearman_rho,
        "mean_lime_fidelity": mean_lime_fidelity,
        "dice_validity": (dice_validity_total, dice_validity_max),
        "output_files": output_files,
        "figures": {
            "shap_summary": fig1,
            "shap_bar": fig2,
            "lime": fig3,
            "confusion": fig4,
            "roc": fig5,
            "comparison": fig6,
            "heatmap": fig7,
        },
        "gpu_capabilities": caps,
        "gpu_info": {
            "message": caps.message,
            "xgboost_cuda": caps.xgboost_cuda,
            "lightgbm_cuda": caps.lightgbm_cuda,
        },
    }


def prepare_scaled_features(
    raw: dict[str, float],
    selected_features: list[str],
    clip_bounds: dict[str, tuple[float, float]],
    train_medians: pd.Series,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """One scaled row aligned with training (IQR clip on BMI/MentHlth/PhysHlth, medians for missing)."""
    row: dict[str, float] = {}
    for c in selected_features:
        v = raw.get(c, np.nan)
        if pd.isna(v):
            v = train_medians.get(c, 0.0)
        row[c] = float(v)
    for col in ["BMI", "MentHlth", "PhysHlth"]:
        if col in row and col in clip_bounds:
            lo, hi = clip_bounds[col]
            row[col] = float(np.clip(row[col], lo, hi))
    X = pd.DataFrame([row])[selected_features]
    return pd.DataFrame(scaler.transform(X), columns=selected_features)


def predict_from_raw(
    raw: dict[str, float],
    selected_features: list[str],
    clip_bounds: dict[str, tuple[float, float]],
    train_medians: pd.Series,
    scaler: MinMaxScaler,
    model: Any,
) -> tuple[float, int]:
    """Return (P(diabetes), predicted_class) for one raw feature dict."""
    Xs = prepare_scaled_features(raw, selected_features, clip_bounds, train_medians, scaler)
    p = float(model.predict_proba(Xs)[0, 1])
    cls = int(model.predict(Xs)[0])
    return p, cls
