from __future__ import annotations

import os

import matplotlib.pyplot as plt
import streamlit as st

from t2dm_pipeline import predict_from_raw, prepare_scaled_features, run_pipeline

st.set_page_config(
    page_title="T2DM Risk Lab",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif; }
  .hero {
    padding: 1.25rem 1.5rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #0f766e 0%, #134e4a 55%, #0f172a 100%);
    color: #f8fafc;
    margin-bottom: 1rem;
    box-shadow: 0 8px 30px rgba(15, 118, 110, 0.25);
  }
  .hero h1 { margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; }
  .hero p { margin: 0.5rem 0 0 0; opacity: 0.92; font-size: 1rem; line-height: 1.45; }
  .badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(255,255,255,0.15);
    margin-top: 0.65rem;
  }
  .card {
    padding: 1rem 1.1rem;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    background: #fff;
  }
  .muted { color: #64748b; font-size: 0.9rem; }
  div[data-testid="stSidebarNav"] { display: none; }
</style>
    """,
    unsafe_allow_html=True,
)


def _figure_path(figs: dict, key: str) -> str | None:
    p = figs.get(key)
    return p if p and os.path.isfile(p) else None


def _gpu_badges(gpu_info: dict | None) -> str:
    if not gpu_info:
        return ""
    x = "CUDA" if gpu_info.get("xgboost_cuda") else "CPU"
    l = "CUDA" if gpu_info.get("lightgbm_cuda") else "CPU"
    return f"XGBoost: {x} · LightGBM: {l} · RF: CPU"


if "result" not in st.session_state:
    st.session_state.result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None


with st.sidebar:
    st.markdown("### Training")
    default_csv = os.path.join(os.path.dirname(__file__) or ".", "diabetes_binary_health_indicators_BRFSS2015.csv")
    main_csv = st.text_input(
        "BRFSS CSV",
        value=default_csv if os.path.isfile(default_csv) else "diabetes_binary_health_indicators_BRFSS2015.csv",
        help="CDC BRFSS diabetes health indicators file.",
    )
    pidd_default = os.path.join(os.path.dirname(__file__) or ".", "diabetes.csv")
    pidd_csv = st.text_input(
        "PIDD CSV (optional)",
        value=pidd_default if os.path.isfile(pidd_default) else "diabetes.csv",
        help="External validation dataset; skipped if missing.",
    )
    use_pidd = st.toggle("External validation (PIDD)", value=True)

    st.divider()
    force_cpu = st.toggle(
        "Force CPU for tree models",
        value=False,
        help="If off, XGBoost & LightGBM probe CUDA (NVIDIA + drivers). Random Forest always uses CPU.",
    )
    use_gpu_param: bool | str = False if force_cpu else "auto"

    quick = st.toggle("Quick mode (5 Optuna trials / model)", value=False)
    n_trials = 5 if quick else st.slider("Optuna trials per model", 5, 50, 30, 5)
    out_dir = st.text_input("Output folder", value=".")

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

    st.divider()
    st.markdown("##### About")
    st.caption(
        "Ensembles: soft voting + stacking. Explainability: SHAP (LightGBM), LIME (stacking), DiCE. "
        "Not a medical device — research / education only."
    )

if run_btn:
    if not os.path.isfile(main_csv):
        st.session_state.last_error = f"File not found: {main_csv}"
        st.session_state.result = None
    else:
        st.session_state.last_error = None

        _pbar = st.progress(0, text="Starting pipeline…")

        _CHECKPOINTS = [
            ("Compute:",               2),
            ("Loading data",           5),
            ("Tuning models",          7),
            ("Base learners trained", 62),
            ("SHAP & LIME",           74),
            ("DiCE",                  88),
            ("Saving figures",        93),
        ]
        _TRIAL_RANGES = {
            "RF":   ( 7, 29),
            "XGB":  (29, 51),
            "LGBM": (51, 62),
        }
        _cur_pct = [0]

        def progress(msg: str) -> None:
            if msg.startswith("TRIAL:"):
                _, model, frac = msg.split(":")
                cur, tot = frac.split("/")
                lo, hi = _TRIAL_RANGES.get(model, (_cur_pct[0], _cur_pct[0]))
                pct = int(lo + int(cur) / int(tot) * (hi - lo))
                _cur_pct[0] = pct
                _pbar.progress(pct, text=f"Tuning {model} — trial {cur}/{tot}  ({pct}%)")
            else:
                for kw, pct in _CHECKPOINTS:
                    if kw in msg:
                        _cur_pct[0] = pct
                        _pbar.progress(pct, text=f"{msg.strip()}  ({pct}%)")
                        return
                _pbar.progress(_cur_pct[0], text=msg.strip())

        try:
            st.session_state.result = run_pipeline(
                main_csv=main_csv,
                pidd_csv=pidd_csv if use_pidd else None,
                n_trials=n_trials,
                output_dir=out_dir,
                use_gpu=use_gpu_param,
                progress=progress,
            )
            _pbar.progress(100, text="Done! (100%)")
        except Exception as e:
            st.session_state.last_error = str(e)
            st.session_state.result = None

st.markdown(
    """
<div class="hero">
  <h1>T2DM risk — ensemble & explainability</h1>
  <p>Random Forest, XGBoost, and LightGBM tuned with Optuna, then combined with soft voting and stacking.
  After training: SHAP, LIME, and counterfactual-style analysis.</p>
</div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

res = st.session_state.result

if res is None:
    st.markdown(
        """
<div class="card">
  <p class="muted" style="margin:0;">
    <strong>Get started:</strong> confirm the BRFSS CSV path in the sidebar, choose <em>Quick mode</em> for a faster run,
    then click <strong>Run pipeline</strong>. When training finishes, metrics, figures, and a risk estimator unlock below.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )
else:
    all_df = res["all_results_df"]
    figs = res["figures"]
    selected = res["selected_features"]
    clip_bounds = res["clip_bounds"]
    train_medians = res["train_medians"]
    scaler = res["scaler"]
    feature_ranges = res["feature_ranges"]
    gpu_info = res.get("gpu_info") or {}

    _best_name_row = all_df.loc[all_df["Model"].str.contains("Best", na=False)]
    _best_display = _best_name_row.iloc[0]["Model"] if len(_best_name_row) else "Stacking"
    _best_key_map = {
        "Random Forest": "rf", "XGBoost": "xgb", "LightGBM": "lgbm",
        "Soft Voting": "voting", "Stacking": "stacking",
    }
    _best_base = next((k for k in _best_key_map if _best_display.startswith(k)), "stacking")
    best_model = res["models"][_best_key_map.get(_best_base, "stacking")]
    stacking = res["models"]["stacking"]

    st.markdown(
        f'<span class="badge">{_gpu_badges(gpu_info)}</span>',
        unsafe_allow_html=True,
    )
    st.caption(gpu_info.get("message", ""))

    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Figures", "Risk estimator", "Explainability"])

    with tab1:
        st.subheader("Performance")

        _auc_numeric = all_df["AUC-ROC"].apply(lambda v: float(v) if v != "—" else -1.0)
        _sorted_df = all_df.assign(_auc=_auc_numeric).sort_values("_auc", ascending=False).drop(columns="_auc").reset_index(drop=True)

        best_row = _sorted_df.loc[_sorted_df["Model"].str.contains("Best", na=False)]
        if len(best_row):
            r = best_row.iloc[0]
            _label = r["Model"].replace(" (Best)", "")
            st.caption(f"Best model by AUC-ROC: **{_label}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{r['Accuracy']:.3f}")
            m2.metric("F1", f"{r['F1']:.3f}")
            m3.metric("AUC-ROC", f"{r['AUC-ROC']}")
            m4.metric("MCC", f"{r['MCC']:.3f}")

        st.dataframe(_sorted_df, use_container_width=True, hide_index=True)

        pidd_df = res.get("pidd_df")
        if pidd_df is not None:
            st.subheader("External validation (PIDD)")
            st.dataframe(pidd_df, use_container_width=True, hide_index=True)

        st.subheader("XAI diagnostics")
        x1, x2, x3 = st.columns(3)
        mss = res.get("mean_spearman_shap")
        with x1:
            st.metric("SHAP stability (mean Spearman ρ)", f"{mss:.4f}" if mss is not None else "—")
        with x2:
            mf = res.get("mean_lime_fidelity")
            st.metric("LIME fidelity (mean R²)", f"{mf:.4f}" if mf is not None else "—")
        with x3:
            dv, dmax = res.get("dice_validity", (0, 0))
            st.metric("DiCE valid CFs", f"{dv} / {dmax}")

    with tab2:
        st.caption("PNGs are saved under your output folder; previews reload from disk.")
        paths = [
            ("SHAP summary (beeswarm)", "shap_summary"),
            ("SHAP — global importance", "shap_bar"),
            ("LIME — example instance", "lime"),
            ("Confusion matrix (stacking)", "confusion"),
            ("ROC curves", "roc"),
            ("Metric comparison", "comparison"),
            ("Feature correlation (top SHAP)", "heatmap"),
        ]
        for title, key in paths:
            with st.expander(title, expanded=(key in ("roc", "shap_summary"))):
                p = _figure_path(figs, key)
                if p:
                    st.image(p, use_container_width=True)
                else:
                    st.warning("Image not found — check the output path.")

    with tab3:
        # Bootstrap Icons (MIT) — loaded once per page render
        st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">', unsafe_allow_html=True)
        st.markdown(
            "Answer the questions below to get a personalised diabetes risk estimate. "
            "All data stays on your device — nothing is sent anywhere."
        )
        st.caption("Tip: answer as accurately as possible for the best estimate.")

        _BIN = {
            "HighBP":               ("Do you have high blood pressure?",
                                     "Have you ever been told by a doctor that you have high blood pressure?"),
            "HighChol":             ("Do you have high cholesterol?",
                                     "Have you ever been told by a doctor that you have high cholesterol?"),
            "CholCheck":            ("Cholesterol checked in the last 5 years?",
                                     "Have you had a cholesterol test within the past 5 years?"),
            "Smoker":               ("Have you smoked 100+ cigarettes in your life?",
                                     "That's roughly 5 packs total over your lifetime."),
            "Stroke":               ("Have you ever had a stroke?",
                                     "Were you ever told by a doctor that you had a stroke?"),
            "HeartDiseaseorAttack": ("Have you had heart disease or a heart attack?",
                                     "Includes coronary heart disease (CHD) and myocardial infarction (MI)."),
            "PhysActivity":         ("Physically active in the past 30 days?",
                                     "Any exercise or physical activity outside of your regular job."),
            "Fruits":               ("Do you eat fruit at least once a day?",
                                     "Consume fruit 1 or more times per day."),
            "Veggies":              ("Do you eat vegetables at least once a day?",
                                     "Consume vegetables 1 or more times per day."),
            "HvyAlcoholConsump":    ("Do you drink heavily?",
                                     "Men: more than 14 drinks per week · Women: more than 7 drinks per week."),
            "AnyHealthcare":        ("Do you have health insurance or any coverage?",
                                     "Includes private insurance, Medicare, Medicaid, or any government plan."),
            "NoDocbcCost":          ("Skipped a doctor visit due to cost in the last 12 months?",
                                     "Was there a time you needed care but couldn't afford it?"),
            "DiffWalk":             ("Do you have serious difficulty walking or climbing stairs?",
                                     "Includes any mobility limitation that makes walking hard."),
        }
        _SEL = {
            "Sex":       ("Biological sex", "",
                          {0: "Female", 1: "Male"}),
            "GenHlth":   ("How would you rate your overall health?",
                          "Think about your physical and mental health together.",
                          {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}),
            "Age":       ("Age group", "Select the range that includes your age.",
                          {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
                           6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
                           11: "70–74", 12: "75–79", 13: "80 or older"}),
            "Education": ("Highest education level completed", "",
                          {1: "Never attended / Kindergarten",
                           2: "Elementary school (Grades 1–8)",
                           3: "Some high school (Grades 9–11)",
                           4: "High school graduate / GED",
                           5: "Some college (1–3 years)",
                           6: "College graduate (4+ years)"}),
            "Income":    ("Annual household income", "Approximate combined income for your household.",
                          {1: "Less than $10,000", 2: "$10,000 – $14,999",
                           3: "$15,000 – $19,999",  4: "$20,000 – $24,999",
                           5: "$25,000 – $34,999",  6: "$35,000 – $49,999",
                           7: "$50,000 – $74,999",  8: "$75,000 or more"}),
        }
        _NUM = {
            "BMI":      ("BMI (Body Mass Index)",
                         "Normal: 18.5–24.9 · Overweight: 25–29.9 · Obese: 30+  "
                         "You can calculate yours at cdc.gov/bmi."),
            "MentHlth": ("Days of poor mental health (last 30 days)",
                         "Days when stress, depression, or emotional problems affected your well-being. Enter 0 if none."),
            "PhysHlth": ("Days of poor physical health (last 30 days)",
                         "Days of illness, injury, or pain in the last 30 days. Enter 0 if none."),
        }

        def _render_section(title: str, icon: str, feats: list) -> dict:
            vals: dict[str, float] = {}
            st.markdown(
                f'<h4 style="display:flex;align-items:center;gap:0.45rem;margin:0.25rem 0 0.75rem;">'
                f'<i class="bi {icon}" style="color:#0f766e;font-size:1.05rem;"></i>{title}</h4>',
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            col_i = 0
            for feat in feats:
                if feat in _BIN:
                    lbl, tip = _BIN[feat]
                    med_val = float(feature_ranges.loc[feat, "median"]) if feat in feature_ranges.index else 0.0
                    default_i = 1 if med_val >= 0.5 else 0
                    with cols[col_i % 2]:
                        ans = st.radio(lbl, ["No", "Yes"], index=default_i,
                                       horizontal=True, help=tip, key=f"pred_{feat}")
                        vals[feat] = 1.0 if ans == "Yes" else 0.0
                    col_i += 1
                elif feat in _SEL:
                    lbl, tip, opts = _SEL[feat]
                    med_val = float(feature_ranges.loc[feat, "median"]) if feat in feature_ranges.index else list(opts)[len(opts) // 2]
                    closest = min(opts, key=lambda k: abs(k - med_val))
                    opt_labels = list(opts.values())
                    with cols[col_i % 2]:
                        if feat in ("Income", "Education"):
                            default_idx = list(opts.keys()).index(closest)
                            sel = st.selectbox(lbl, options=opt_labels, index=default_idx,
                                               help=tip if tip else None, key=f"pred_{feat}")
                        else:
                            sel = st.select_slider(lbl, options=opt_labels,
                                                   value=opts[closest],
                                                   help=tip if tip else None, key=f"pred_{feat}")
                        vals[feat] = float(next(k for k, v in opts.items() if v == sel))
                    col_i += 1
                elif feat in _NUM:
                    lbl, tip = _NUM[feat]
                    mn  = float(feature_ranges.loc[feat, "min"])   if feat in feature_ranges.index else 0.0
                    mx  = float(feature_ranges.loc[feat, "max"])   if feat in feature_ranges.index else 100.0
                    med = float(feature_ranges.loc[feat, "median"]) if feat in feature_ranges.index else (mn + mx) / 2
                    med = max(mn, min(mx, med))
                    step = 0.1 if (mx - mn) <= 10 else 1.0
                    with cols[col_i % 2]:
                        vals[feat] = st.number_input(lbl, min_value=mn, max_value=mx,
                                                      value=med, step=step,
                                                      help=tip, key=f"pred_{feat}")
                    col_i += 1
            return vals

        raw: dict[str, float] = {}
        _grp = {
            "demo":  [f for f in selected if f in ("Sex", "Age", "Education", "Income")],
            "med":   [f for f in selected if f in ("HighBP", "HighChol", "CholCheck", "Stroke",
                                                    "HeartDiseaseorAttack", "DiffWalk")],
            "life":  [f for f in selected if f in ("Smoker", "PhysActivity", "Fruits",
                                                    "Veggies", "HvyAlcoholConsump")],
            "hlth":  [f for f in selected if f in ("BMI", "GenHlth", "MentHlth", "PhysHlth")],
            "care":  [f for f in ("AnyHealthcare", "NoDocbcCost") if f in _BIN],
        }
        mapped = {f for grp in _grp.values() for f in grp}
        _grp["other"] = [f for f in selected if f not in mapped]

        if _grp["demo"]:
            raw.update(_render_section(" About You ", " bi-person ", _grp["demo"]))
            st.divider()
        if _grp["med"]:
            raw.update(_render_section(" Medical History ", " bi-heart-pulse ", _grp["med"]))
            st.divider()
        if _grp["life"]:
            raw.update(_render_section(" Lifestyle ", " bi-bicycle ", _grp["life"]))
            st.divider()
        if _grp["hlth"]:
            raw.update(_render_section(" How Do You Feel? ", " bi-thermometer-half ", _grp["hlth"]))
            st.divider()
        if _grp["care"]:
            raw.update(_render_section(" Healthcare Access ", " bi-shield-check ", _grp["care"]))
            st.divider()
        if _grp["other"]:
            with st.expander("Other factors", expanded=False):
                cols_o = st.columns(3)
                for i, feat in enumerate(_grp["other"]):
                    mn = float(feature_ranges.loc[feat, "min"])
                    mx = float(feature_ranges.loc[feat, "max"])
                    if mn > mx:
                        mn, mx = mx, mn
                    med = float(feature_ranges.loc[feat, "median"])
                    med = max(mn, min(mx, med))
                    with cols_o[i % 3]:
                        raw[feat] = st.number_input(feat, min_value=mn, max_value=mx,
                                                     value=med, key=f"pred_{feat}")

        _model_options = ["Stacking", "Soft Voting", "LightGBM", "XGBoost", "Random Forest"]
        _best_base_clean = _best_base
        _default_idx = next((i for i, o in enumerate(_model_options) if o == _best_base_clean), 0)
        with st.expander("Advanced — choose prediction model", expanded=False):
            model_choice = st.selectbox(
                "Model",
                _model_options,
                index=_default_idx,
                help="Defaults to the model with the highest AUC-ROC on the held-out test set.",
            )
        name_to_model = {
            "Stacking":      res["models"]["stacking"],
            "Soft Voting":   res["models"]["voting"],
            "LightGBM":      res["models"]["lgbm"],
            "XGBoost":       res["models"]["xgb"],
            "Random Forest": res["models"]["rf"],
        }
        model = name_to_model.get(model_choice, best_model)

        st.markdown("")
        go = st.button("Check My Risk", type="primary", use_container_width=True)

        if go:
            p, cls = predict_from_raw(raw, selected, clip_bounds, train_medians, scaler, model)

            if p < 0.20:
                risk_label, bg, border, advice = (
                    "Low Risk", "#f0fdf4", "#16a34a",
                    "Your responses suggest a lower likelihood of diabetes. Keep up the healthy habits!",
                )
            elif p < 0.40:
                risk_label, bg, border, advice = (
                    "Moderate Risk", "#fffbeb", "#d97706",
                    "Some risk factors detected. Consider discussing them with your doctor at your next check-up.",
                )
            elif p < 0.60:
                risk_label, bg, border, advice = (
                    "High Risk", "#fff1f2", "#dc2626",
                    "Several risk factors present. We recommend scheduling an appointment with a healthcare professional.",
                )
            else:
                risk_label, bg, border, advice = (
                    "Very High Risk", "#fef2f2", "#7f1d1d",
                    "High probability of risk factors associated with diabetes. Please consult a doctor as soon as possible.",
                )

            st.markdown(
                f"""
<div style="padding:1.5rem 2rem;border-radius:14px;border:2px solid {border};
            background:{bg};text-align:center;margin:1.2rem 0;">
  <div style="font-size:1.5rem;font-weight:700;color:{border};letter-spacing:-0.01em;">{risk_label}</div>
  <div style="font-size:3rem;font-weight:800;color:{border};margin:0.25rem 0;line-height:1;">{p:.1%}</div>
  <div style="font-size:0.95rem;color:#374151;margin-top:0.6rem;max-width:480px;margin-left:auto;margin-right:auto;">{advice}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(p, 0.0), 1.0))
            st.caption(
                "This tool is for educational / research purposes only and does not constitute medical advice. "
                "Consult a qualified healthcare professional for diagnosis and treatment."
            )

            with st.expander("What factors influenced this estimate?", expanded=False):
                st.caption(
                    "Bars pushing toward 'Diabetic' increase the risk score; "
                    "bars pushing the other way reduce it."
                )
                lime_ex = res["lime_explainer"]
                Xs = prepare_scaled_features(raw, selected, clip_bounds, train_medians, scaler)
                exp = lime_ex.explain_instance(
                    Xs.iloc[0].values,
                    stacking.predict_proba,
                    num_features=min(10, len(selected)),
                    num_samples=2000,
                )
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
                plt.close(fig)

    with tab4:
        st.markdown(
            """
**Pipeline**

- **Tuning**: Optuna maximizes ROC-AUC (3-fold CV) per base learner.
- **Trees on GPU** (when CUDA works): XGBoost & LightGBM use `device=cuda`; cross-validation runs sequentially on the GPU to limit VRAM spikes.
- **CPU**: Random Forest; SHAP / LIME / DiCE run on CPU.

**Outputs**

- CSV metrics and SHAP importances; figures for reports.
            """
        )
        st.subheader("Top drivers (mean |SHAP|, LightGBM)")
        st.dataframe(
            res["mean_shap"].head(25).to_frame("mean_abs_shap"),
            use_container_width=True,
            height=400,
        )

