# =============================================================================
# CropCast — Week 4: Ridge + Lasso + Logistic Regression + Feature Engineering
#
# Three deliverables:
# 1. Feature Engineering — engineered features from soil, weather, NDVI, sowing
# 2. Ridge + Lasso on Expected Yield (regression) — must beat W3 RMSE 1,312.10
# 3. Logistic Regression on Disease Outbreak (classification) — first classifier
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH   = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/clean_dataset.csv"
FIGURES_PATH      = "/Users/amrkalammansoori/Desktop/crop-cast/reports/figures"
MLFLOW_DIR        = "/Users/amrkalammansoori/Desktop/crop-cast/mlruns"
REGRESSION_TARGET  = "expected_yield_kg_ha"
CLASSIFICATION_TARGET = "disease_outbreak"
RANDOM_STATE       = 42

# W3 baseline to beat
W3_OLS_RMSE = 1312.10
W3_OLS_R2   = 0.2779

# ── HELPER: Regression metrics ────────────────────────────────────────────────
def reg_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  📊 {label} Metrics:")
    print(f"     MAE  = {mae:.2f} kg/ha")
    print(f"     RMSE = {rmse:.2f} kg/ha")
    print(f"     R²   = {r2:.4f}")
    return mae, rmse, r2

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    print("\n" + "="*60)
    print("STEP 1 — LOADING DATASET")
    print("="*60)

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(f"Run src/data/ingest.py first.\n{CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df

# ── STEP 2: FEATURE ENGINEERING ───────────────────────────────────────────────
def engineer_features(df):
    """
    Create domain-informed engineered features from soil, weather, NDVI,
    and sowing dates. This is the key Week 4 requirement.
    """
    print("\n" + "="*60)
    print("STEP 2 — FEATURE ENGINEERING")
    print("="*60)

    fe = df.copy()

    # ── Soil Features ──────────────────────────────────────────────────────
    # NPK composite score — weighted sum of all three macronutrients
    # Nitrogen is most critical (0.5), then Phosphorus (0.3), Potassium (0.2)
    fe["npk_total"] = (
        0.5 * (fe["soil_nitrogen"] / fe["soil_nitrogen"].max()) +
        0.3 * (fe["soil_phosphorus"] / fe["soil_phosphorus"].max()) +
        0.2 * (fe["soil_potassium"] / fe["soil_potassium"].max())
    )
    print("✅ npk_total — weighted NPK composite score (N×0.5, P×0.3, K×0.2)")

    # Soil pH optimality — Gaussian peak at 6.5, drops off either side
    fe["ph_optimality"] = np.exp(-0.5 * ((fe["soil_ph"] - 6.5) / 0.8) ** 2)
    print("✅ ph_optimality — Gaussian pH score, peaks at 6.5 (ideal for most crops)")

    # ── Weather Features ───────────────────────────────────────────────────
    # Heat-humidity interaction — primary disease driver
    # Both must be elevated simultaneously for fungal disease conditions
    fe["heat_humidity_index"] = (
        np.clip((fe["temperature"] - 20) / 15, 0, 1) *
        np.clip((fe["humidity"] - 55) / 40, 0, 1)
    )
    print("✅ heat_humidity_index — temperature × humidity interaction (disease driver)")

    # Rainfall optimality — optimal 80-150mm/month
    fe["rainfall_optimality"] = np.exp(
        -0.00005 * (fe["rainfall"] - 115) ** 2
    )
    print("✅ rainfall_optimality — Gaussian rainfall score, peaks at 115mm/month")

    # Rainfall category — drought / optimal / excess
    fe["rainfall_category"] = pd.cut(
        fe["rainfall"],
        bins=[0, 50, 80, 150, 380],
        labels=[0, 1, 2, 3]   # 0=drought, 1=low, 2=optimal, 3=excess
    ).astype(float).fillna(1)
    print("✅ rainfall_category — 0=drought, 1=low, 2=optimal, 3=excess")

    # ── NDVI / Satellite Features ──────────────────────────────────────────
    # NDVI stress flag — crops below 0.35 NDVI are in physiological stress
    fe["ndvi_stress_flag"] = (fe["ndvi"] < 0.35).astype(int)
    pct_stressed = fe["ndvi_stress_flag"].mean() * 100
    print(f"✅ ndvi_stress_flag — 1 if NDVI<0.35 (physiological stress) "
          f"[{pct_stressed:.1f}% of fields stressed]")

    # NDVI × NPK interaction — healthy vegetation + good nutrients = best yield
    fe["ndvi_npk_interaction"] = fe["ndvi"] * fe["npk_total"]
    print("✅ ndvi_npk_interaction — NDVI × NPK composite (vegetation health × nutrition)")

    # ── Sowing Date Features ───────────────────────────────────────────────
    # Sowing delay severity bucket — ordinal encoding
    fe["sowing_delay_bucket"] = pd.cut(
        fe["sowing_delay_days"],
        bins=[-1, 7, 14, 21, 45],
        labels=[0, 1, 2, 3]   # 0=optimal, 1=mild, 2=moderate, 3=severe
    ).astype(float).fillna(0)
    print("✅ sowing_delay_bucket — 0=optimal(<7d), 1=mild, 2=moderate, 3=severe(>21d)")

    # ── Categorical Encoding ───────────────────────────────────────────────
    # One-hot encode crop_type, season, agro_zone, region
    cat_cols = ["crop_type", "season", "agro_zone", "region"]
    fe = pd.get_dummies(fe, columns=cat_cols, drop_first=True, dtype=float)
    print(f"✅ One-hot encoded: {cat_cols}")
    print(f"   Dataset shape after encoding: {fe.shape}")

    # Count new features added
    orig_cols = df.shape[1]
    new_cols  = fe.shape[1]
    print(f"\n  Feature engineering summary:")
    print(f"  Original features  : {orig_cols}")
    print(f"  After engineering  : {new_cols}")
    print(f"  New features added : {new_cols - orig_cols}")

    return fe

# ── STEP 3: PREPARE TRAIN/TEST ─────────────────────────────────────────────
def prepare_data(fe):
    print("\n" + "="*60)
    print("STEP 3 — TRAIN / TEST SPLIT & SCALING")
    print("="*60)

    # Drop target columns and crop_growth_stage (ordinal, already encoded)
    drop_cols = [REGRESSION_TARGET, CLASSIFICATION_TARGET, "crop_growth_stage"]
    feature_cols = [c for c in fe.columns if c not in drop_cols]

    X = fe[feature_cols].values
    y_yield    = fe[REGRESSION_TARGET].values
    y_disease  = fe[CLASSIFICATION_TARGET].values

    # Same 80/20 split as W3 for fair comparison
    X_train, X_test, y_yield_train, y_yield_test, y_dis_train, y_dis_test = \
        train_test_split(X, y_yield, y_disease,
                         test_size=0.20, random_state=RANDOM_STATE)

    print(f"✅ Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")
    print(f"✅ Features: {X_train.shape[1]}")

    # Scale — fit on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print("✅ StandardScaler fitted on train only (no leakage)")

    # Check for NaN/Inf after encoding
    if not np.isfinite(X_train_s).all():
        X_train_s = np.nan_to_num(X_train_s)
        X_test_s  = np.nan_to_num(X_test_s)
        print("⚠️  Non-finite values replaced with 0")

    naive_rmse = np.sqrt(mean_squared_error(
        y_yield_test, np.full_like(y_yield_test, y_yield_train.mean())))
    print(f"\n  W3 OLS baseline to beat : RMSE = {W3_OLS_RMSE:.2f} kg/ha  R² = {W3_OLS_R2}")
    print(f"  Naïve baseline RMSE     : {naive_rmse:.2f} kg/ha")

    return (X_train_s, X_test_s, y_yield_train, y_yield_test,
            y_dis_train, y_dis_test, scaler, feature_cols)

# ── STEP 4: RIDGE REGRESSION ──────────────────────────────────────────────────
def run_ridge(X_train, X_test, y_train, y_test, feature_cols):
    """
    Ridge Regression — L2 regularization.
    Adds penalty: λ × Σ(wᵢ²) to the loss function.
    Shrinks ALL coefficients but none reach exactly zero.
    Best for: many features all somewhat relevant.
    """
    print("\n" + "="*60)
    print("STEP 4 — RIDGE REGRESSION (L2 Regularization)")
    print("="*60)
    print("  Loss = MSE + λ × Σ(wᵢ²)")
    print("  Shrinks all coefficients — none reach exactly zero\n")

    # Test multiple alpha values
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    print("  Alpha search (5-fold CV on train set):")
    best_alpha, best_cv_rmse = None, np.inf

    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        marker = " ← best" if cv_rmse < best_cv_rmse else ""
        print(f"    α={alpha:<8}  CV RMSE={cv_rmse:.2f}{marker}")
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_alpha   = alpha

    print(f"\n  Best α = {best_alpha}  (CV RMSE = {best_cv_rmse:.2f})")

    # Train final Ridge model
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Yield_Prediction")

    with mlflow.start_run(run_name="W4_Ridge_Regression"):
        ridge = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
        ridge.fit(X_train, y_train)

        y_pred_train = ridge.predict(X_train)
        y_pred_test  = ridge.predict(X_test)

        train_mae, train_rmse, train_r2 = reg_metrics(y_train, y_pred_train, "TRAIN")
        test_mae,  test_rmse,  test_r2  = reg_metrics(y_test,  y_pred_test,  "TEST ")

        improvement_vs_w3 = W3_OLS_RMSE - test_rmse
        beat_w3 = improvement_vs_w3 > 0

        print(f"\n  vs W3 OLS baseline:")
        print(f"  {'✅ BEAT' if beat_w3 else '❌ DID NOT BEAT'} W3 OLS!")
        print(f"  W3 RMSE: {W3_OLS_RMSE:.2f}  →  Ridge RMSE: {test_rmse:.2f}  "
              f"({improvement_vs_w3:+.2f} kg/ha)")

        # Top coefficients
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": ridge.coef_
        }).sort_values("Coefficient", key=abs, ascending=False).head(10)
        print("\n  Top 10 Ridge coefficients (none exactly zero):")
        print(coef_df.to_string(index=False))

        mlflow.log_param("model",        "Ridge")
        mlflow.log_param("week",         "W4")
        mlflow.log_param("best_alpha",   best_alpha)
        mlflow.log_param("n_features",   X_train.shape[1])
        mlflow.log_metric("train_rmse",  train_rmse)
        mlflow.log_metric("train_r2",    train_r2)
        mlflow.log_metric("test_rmse",   test_rmse)
        mlflow.log_metric("test_r2",     test_r2)
        mlflow.log_metric("w3_ols_rmse", W3_OLS_RMSE)
        mlflow.log_metric("improvement_vs_w3", improvement_vs_w3)
        mlflow.sklearn.log_model(ridge, "ridge_model")
        print(f"\n  ✅ MLflow run logged → W4_Ridge_Regression")

    return ridge, y_pred_test, test_rmse, test_r2, best_alpha

# ── STEP 5: LASSO REGRESSION ──────────────────────────────────────────────────
def run_lasso(X_train, X_test, y_train, y_test, feature_cols):
    """
    Lasso Regression — L1 regularization.
    Adds penalty: λ × Σ|wᵢ| to the loss function.
    Drives some coefficients to EXACTLY zero = automatic feature selection.
    Best for: sparse problems where many features are irrelevant.
    """
    print("\n" + "="*60)
    print("STEP 5 — LASSO REGRESSION (L1 Regularization)")
    print("="*60)
    print("  Loss = MSE + λ × Σ|wᵢ|")
    print("  Drives some coefficients to exactly ZERO — feature selection!\n")

    # Test multiple alpha values
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    print("  Alpha search (5-fold CV on train set):")
    best_alpha, best_cv_rmse = None, np.inf

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=5000, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        marker = " ← best" if cv_rmse < best_cv_rmse else ""
        print(f"    α={alpha:<8}  CV RMSE={cv_rmse:.2f}{marker}")
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_alpha   = alpha

    print(f"\n  Best α = {best_alpha}  (CV RMSE = {best_cv_rmse:.2f})")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Yield_Prediction")

    with mlflow.start_run(run_name="W4_Lasso_Regression"):
        lasso = Lasso(alpha=best_alpha, max_iter=5000, random_state=RANDOM_STATE)
        lasso.fit(X_train, y_train)

        y_pred_train = lasso.predict(X_train)
        y_pred_test  = lasso.predict(X_test)

        train_mae, train_rmse, train_r2 = reg_metrics(y_train, y_pred_train, "TRAIN")
        test_mae,  test_rmse,  test_r2  = reg_metrics(y_test,  y_pred_test,  "TEST ")

        # Feature selection analysis — the key Lasso insight
        zero_coefs   = np.sum(lasso.coef_ == 0)
        active_coefs = np.sum(lasso.coef_ != 0)
        print(f"\n  Feature selection by Lasso:")
        print(f"  Features eliminated (coef=0) : {zero_coefs}")
        print(f"  Features retained (coef≠0)   : {active_coefs}")
        print(f"  Sparsity: {zero_coefs/len(feature_cols)*100:.1f}% of features zeroed out")

        improvement_vs_w3 = W3_OLS_RMSE - test_rmse
        beat_w3 = improvement_vs_w3 > 0
        print(f"\n  {'✅ BEAT' if beat_w3 else '❌ DID NOT BEAT'} W3 OLS!")
        print(f"  W3 RMSE: {W3_OLS_RMSE:.2f}  →  Lasso RMSE: {test_rmse:.2f}  "
              f"({improvement_vs_w3:+.2f} kg/ha)")

        # Non-zero coefficients
        nonzero_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": lasso.coef_
        })
        nonzero_df = nonzero_df[nonzero_df["Coefficient"] != 0]\
                        .sort_values("Coefficient", key=abs, ascending=False)
        print(f"\n  Top retained features (non-zero coefficients):")
        print(nonzero_df.head(12).to_string(index=False))

        mlflow.log_param("model",           "Lasso")
        mlflow.log_param("week",            "W4")
        mlflow.log_param("best_alpha",      best_alpha)
        mlflow.log_param("n_features_in",   X_train.shape[1])
        mlflow.log_param("n_features_kept", int(active_coefs))
        mlflow.log_param("n_features_zero", int(zero_coefs))
        mlflow.log_metric("train_rmse",     train_rmse)
        mlflow.log_metric("train_r2",       train_r2)
        mlflow.log_metric("test_rmse",      test_rmse)
        mlflow.log_metric("test_r2",        test_r2)
        mlflow.log_metric("w3_ols_rmse",    W3_OLS_RMSE)
        mlflow.log_metric("improvement_vs_w3", improvement_vs_w3)
        mlflow.sklearn.log_model(lasso, "lasso_model")
        print(f"\n  ✅ MLflow run logged → W4_Lasso_Regression")

    return lasso, y_pred_test, test_rmse, test_r2, best_alpha, zero_coefs, active_coefs

# ── STEP 6: LOGISTIC REGRESSION ───────────────────────────────────────────────
def run_logistic(X_train, X_test, y_train, y_test):
    """
    Logistic Regression for disease-outbreak classification.
    Outputs probability via sigmoid: P(outbreak=1 | features)
    Evaluated with Precision, Recall, F1, AUC-ROC — not accuracy.

    class_weight='balanced' handles the 75/25 class imbalance automatically.
    """
    print("\n" + "="*60)
    print("STEP 6 — LOGISTIC REGRESSION (Disease Outbreak Classifier)")
    print("="*60)
    print("  Output: P(disease_outbreak=1 | soil, weather, NDVI)")
    print("  Threshold: 0.5  |  class_weight='balanced' for imbalance\n")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Disease_Prediction")

    with mlflow.start_run(run_name="W4_LogisticRegression_Classifier"):
        lr = LogisticRegression(
            C=1.0,                    # inverse of regularization strength
            class_weight="balanced",  # handles 75/25 imbalance
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs"
        )
        lr.fit(X_train, y_train)

        y_pred       = lr.predict(X_test)
        y_prob       = lr.predict_proba(X_test)[:, 1]

        # Classification metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc_roc   = roc_auc_score(y_test, y_prob)
        accuracy  = (y_pred == y_test).mean()

        print(f"  📊 Classification Metrics:")
        print(f"     Accuracy  = {accuracy:.4f}  (⚠️ misleading with imbalance)")
        print(f"     Precision = {precision:.4f}  (of flagged outbreaks, % correct)")
        print(f"     Recall    = {recall:.4f}  (of actual outbreaks, % caught)")
        print(f"     F1 Score  = {f1:.4f}  (harmonic mean of P & R)")
        print(f"     AUC-ROC   = {auc_roc:.4f}  (primary metric — higher = better)")

        majority_baseline = 1 - y_test.mean()
        print(f"\n  Majority-class baseline accuracy: {majority_baseline:.4f}")
        print(f"  Our model accuracy              : {accuracy:.4f}")
        beat_baseline = accuracy > majority_baseline
        print(f"  {'✅ BEAT' if beat_baseline else '❌ DID NOT BEAT'} majority-class baseline")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  Confusion Matrix:")
        print(f"  True Negatives  (correctly flagged no outbreak): {tn:,}")
        print(f"  False Positives (wrongly flagged as outbreak)  : {fp:,}")
        print(f"  False Negatives (missed real outbreaks)        : {fn:,}")
        print(f"  True Positives  (correctly caught outbreaks)   : {tp:,}")
        print(f"\n  ⚠️  False Negatives ({fn:,}) are the costly errors —")
        print(f"     a missed outbreak can wipe an entire harvest.")

        print(f"\n{classification_report(y_test, y_pred, target_names=['No Outbreak','Outbreak'])}")

        mlflow.log_param("model",          "LogisticRegression")
        mlflow.log_param("week",           "W4")
        mlflow.log_param("C",              1.0)
        mlflow.log_param("class_weight",   "balanced")
        mlflow.log_param("n_features",     X_train.shape[1])
        mlflow.log_metric("accuracy",      accuracy)
        mlflow.log_metric("precision",     precision)
        mlflow.log_metric("recall",        recall)
        mlflow.log_metric("f1_score",      f1)
        mlflow.log_metric("auc_roc",       auc_roc)
        mlflow.log_metric("true_positives",  int(tp))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.sklearn.log_model(lr, "logistic_model")
        print(f"  ✅ MLflow run logged → W4_LogisticRegression_Classifier")

    return lr, y_pred, y_prob, precision, recall, f1, auc_roc

# ── STEP 7: VISUALISATIONS ────────────────────────────────────────────────────
def generate_plots(df_orig, ridge, lasso, lr, X_test, y_yield_test,
                   y_dis_test, y_prob, feature_cols,
                   ridge_pred, lasso_pred,
                   ridge_rmse, lasso_rmse, ridge_alpha, lasso_alpha,
                   zero_coefs, active_coefs):
    print("\n" + "="*60)
    print("STEP 7 — GENERATING VISUALISATIONS")
    print("="*60)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # ── Plot 12: Ridge vs Lasso vs W3 OLS RMSE comparison ─────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    models = ["W3 OLS\n(Baseline)", f"W4 Ridge\n(α={ridge_alpha})", f"W4 Lasso\n(α={lasso_alpha})"]
    rmses  = [W3_OLS_RMSE, ridge_rmse, lasso_rmse]
    colors = ["#E07B39", "#2D6A4F", "#5FA8D3"]
    bars   = ax.bar(models, rmses, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.axhline(W3_OLS_RMSE, color="#E07B39", linewidth=1.5,
               linestyle="--", alpha=0.5, label="W3 OLS baseline")
    ax.set_ylabel("RMSE (kg/ha) — lower is better", fontsize=12)
    ax.set_title("W4 Model Comparison — Ridge vs Lasso vs W3 Baseline\nCropCast",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(rmses) * 1.2)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/12_w4_regression_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 13: Lasso feature selection — coefficients bar chart ──────────
    lasso_coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": lasso.coef_
    }).sort_values("Coefficient", key=abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(11, 7))
    colors_bar = ["#2D6A4F" if v > 0 else "#E07B39"
                  for v in lasso_coef_df["Coefficient"]]
    ax.barh(lasso_coef_df["Feature"], lasso_coef_df["Coefficient"],
            color=colors_bar, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lasso Coefficient (0 = eliminated by L1)", fontsize=11)
    ax.set_title(f"Lasso Feature Coefficients (α={lasso_alpha})\n"
                 f"{zero_coefs} features eliminated · {active_coefs} features retained",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_PATH}/13_lasso_coefficients.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 14: Ridge vs Lasso — Actual vs Predicted ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Ridge vs Lasso — Actual vs Predicted Yield\nCropCast W4",
                 fontsize=13, fontweight="bold")
    for ax, pred, name, col in zip(
        axes,
        [ridge_pred, lasso_pred],
        [f"Ridge (RMSE={ridge_rmse:.0f})", f"Lasso (RMSE={lasso_rmse:.0f})"],
        ["#2D6A4F", "#5FA8D3"]
    ):
        ax.scatter(y_yield_test, pred, alpha=0.15, s=5, color=col, rasterized=True)
        lims = [min(y_yield_test.min(), pred.min()),
                max(y_yield_test.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect")
        ax.set_xlabel("Actual Yield (kg/ha)"); ax.set_ylabel("Predicted Yield (kg/ha)")
        ax.set_title(name); ax.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}/14_w4_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 15: ROC Curve ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    RocCurveDisplay.from_predictions(
        y_dis_test, y_prob, ax=ax,
        color="#2D6A4F", name="Logistic Regression (W4)"
    )
    ax.plot([0,1],[0,1], "k--", linewidth=1, label="Random classifier")
    ax.set_title("ROC Curve — Disease Outbreak Classifier\nCropCast W4",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/15_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 16: Confusion matrix heatmap ──────────────────────────────────
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_dis_test, lr.predict(X_test),
        display_labels=["No Outbreak", "Outbreak"],
        cmap="Greens", ax=ax
    )
    ax.set_title("Confusion Matrix — Disease Outbreak Classifier\nCropCast W4",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_PATH}/16_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✅ Saved: {path}")

# ── STEP 8: FINAL SUMMARY ─────────────────────────────────────────────────────
def print_summary(ridge_rmse, ridge_r2, ridge_alpha,
                  lasso_rmse, lasso_r2, lasso_alpha, zero_coefs, active_coefs,
                  precision, recall, f1, auc_roc):
    print("\n" + "█"*60)
    print("  WEEK 4 RESULTS SUMMARY")
    print("█"*60)

    print(f"\n  REGRESSION — Expected Yield")
    print(f"  {'Model':<30} {'RMSE':>8}  {'R²':>8}  {'vs W3':>10}")
    print(f"  {'-'*58}")
    print(f"  {'W3 OLS Baseline':<30} {W3_OLS_RMSE:>8.2f}  {W3_OLS_R2:>8.4f}  {'(reference)':>10}")
    print(f"  {'W4 Ridge (α='+str(ridge_alpha)+')':<30} {ridge_rmse:>8.2f}  {ridge_r2:>8.4f}  "
          f"  {W3_OLS_RMSE-ridge_rmse:>+8.2f}")
    print(f"  {'W4 Lasso (α='+str(lasso_alpha)+')':<30} {lasso_rmse:>8.2f}  {lasso_r2:>8.4f}  "
          f"  {W3_OLS_RMSE-lasso_rmse:>+8.2f}")

    print(f"\n  Lasso feature selection:")
    print(f"  {zero_coefs} features eliminated (coef = exactly 0)")
    print(f"  {active_coefs} features retained (coef ≠ 0)")

    print(f"\n  CLASSIFICATION — Disease Outbreak")
    print(f"  {'Metric':<20} {'Value':>8}")
    print(f"  {'-'*30}")
    print(f"  {'Precision':<20} {precision:>8.4f}")
    print(f"  {'Recall':<20} {recall:>8.4f}")
    print(f"  {'F1 Score':<20} {f1:>8.4f}")
    print(f"  {'AUC-ROC':<20} {auc_roc:>8.4f}")

    print(f"\n  5 plots saved to: {FIGURES_PATH}")
    print(f"     12_w4_regression_comparison.png")
    print(f"     13_lasso_coefficients.png")
    print(f"     14_w4_actual_vs_predicted.png")
    print(f"     15_roc_curve.png")
    print(f"     16_confusion_matrix.png")

    print(f"\n  MLflow experiments updated:")
    print(f"     CropCast_Yield_Prediction   → W4_Ridge + W4_Lasso")
    print(f"     CropCast_Disease_Prediction → W4_LogisticRegression")

    print(f"\n  Next → Week 5: Cross-validation, one-standard-error rule,")
    print(f"                  ship the honest baseline milestone.")
    print("█"*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 4")
    print("  Ridge + Lasso + Logistic Regression + Feature Engineering")
    print("█"*60)

    # Pipeline
    df                          = load_data()
    fe                          = engineer_features(df)
    (X_train, X_test,
     y_yield_train, y_yield_test,
     y_dis_train, y_dis_test,
     scaler, feature_cols)      = prepare_data(fe)

    ridge, ridge_pred, \
        ridge_rmse, ridge_r2, \
        ridge_alpha             = run_ridge(X_train, X_test,
                                            y_yield_train, y_yield_test,
                                            feature_cols)

    lasso, lasso_pred, \
        lasso_rmse, lasso_r2, \
        lasso_alpha, \
        zero_coefs, active_coefs = run_lasso(X_train, X_test,
                                              y_yield_train, y_yield_test,
                                              feature_cols)

    lr, y_pred_dis, y_prob, \
        precision, recall, \
        f1, auc_roc             = run_logistic(X_train, X_test,
                                                y_dis_train, y_dis_test)

    generate_plots(df, ridge, lasso, lr, X_test,
                   y_yield_test, y_dis_test, y_prob, feature_cols,
                   ridge_pred, lasso_pred,
                   ridge_rmse, lasso_rmse, ridge_alpha, lasso_alpha,
                   zero_coefs, active_coefs)

    print_summary(ridge_rmse, ridge_r2, ridge_alpha,
                  lasso_rmse, lasso_r2, lasso_alpha, zero_coefs, active_coefs,
                  precision, recall, f1, auc_roc)