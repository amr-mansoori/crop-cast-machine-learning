# =============================================================================
# CropCast — Week 8: XGBoost Champion + Early Stopping + SHAP
# File: src/models/train_w8.py
#
# Requirements:
# 1. Build XGBoost for both targets (yield + disease)
# 2. Tune with early stopping
# 3. Explain predictions with SHAP values
# 4. This is the champion model — best performance across all weeks
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             precision_score, recall_score,
                             f1_score, roc_auc_score, RocCurveDisplay)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH       = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/clean_dataset.csv"
FIGURES_PATH          = "/Users/amrkalammansoori/Desktop/crop-cast/reports/figures"
MLFLOW_DIR            = "/Users/amrkalammansoori/Desktop/crop-cast/mlruns"
REGRESSION_TARGET     = "expected_yield_kg_ha"
CLASSIFICATION_TARGET = "disease_outbreak"
RANDOM_STATE          = 42

# Baselines to beat
W5_RMSE = 955.14
W5_AUC  = 0.9154

# ── FEATURE ENGINEERING (identical to W4-W7) ──────────────────────────────────
def load_and_engineer():
    print("\n" + "="*60)
    print("STEP 1 — LOAD & ENGINEER FEATURES")
    print("="*60)

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    fe = df.copy()
    fe["npk_total"]            = (0.5*(fe["soil_nitrogen"]/fe["soil_nitrogen"].max()) +
                                   0.3*(fe["soil_phosphorus"]/fe["soil_phosphorus"].max()) +
                                   0.2*(fe["soil_potassium"]/fe["soil_potassium"].max()))
    fe["ph_optimality"]        = np.exp(-0.5*((fe["soil_ph"]-6.5)/0.8)**2)
    fe["heat_humidity_index"]  = (np.clip((fe["temperature"]-20)/15,0,1) *
                                   np.clip((fe["humidity"]-55)/40,0,1))
    fe["rainfall_optimality"]  = np.exp(-0.00005*(fe["rainfall"]-115)**2)
    fe["rainfall_category"]    = pd.cut(fe["rainfall"],bins=[0,50,80,150,380],
                                          labels=[0,1,2,3]).astype(float).fillna(1)
    fe["ndvi_stress_flag"]     = (fe["ndvi"] < 0.35).astype(int)
    fe["ndvi_npk_interaction"] = fe["ndvi"] * fe["npk_total"]
    fe["sowing_delay_bucket"]  = pd.cut(fe["sowing_delay_days"],bins=[-1,7,14,21,45],
                                          labels=[0,1,2,3]).astype(float).fillna(0)

    cat_cols     = ["crop_type","season","agro_zone","region"]
    fe           = pd.get_dummies(fe, columns=cat_cols, drop_first=True, dtype=float)
    drop_cols    = [REGRESSION_TARGET, CLASSIFICATION_TARGET, "crop_growth_stage"]
    feature_cols = [c for c in fe.columns if c not in drop_cols]

    X         = fe[feature_cols].values
    y_yield   = fe[REGRESSION_TARGET].values
    y_disease = fe[CLASSIFICATION_TARGET].values

    # 70/10/20 split — need a validation set for early stopping
    X_temp, X_test, y_y_temp, y_y_test, y_d_temp, y_d_test = train_test_split(
        X, y_yield, y_disease,
        test_size=0.20, random_state=RANDOM_STATE)

    X_train, X_val, y_y_train, y_y_val, y_d_train, y_d_val = train_test_split(
        X_temp, y_y_temp, y_d_temp,
        test_size=0.125, random_state=RANDOM_STATE)  # 0.125 × 0.80 = 0.10

    print(f"✅ Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  "
          f"Test: {X_test.shape[0]:,}  Features: {X_train.shape[1]}")
    print(f"   Split: ~70% train / ~10% val (early stopping) / ~20% test")

    return (X_train, X_val, X_test,
            y_y_train, y_y_val, y_y_test,
            y_d_train, y_d_val, y_d_test,
            feature_cols)

# ── STEP 2: XGBOOST REGRESSOR ─────────────────────────────────────────────────
def run_xgb_regressor(X_train, X_val, X_test,
                      y_train, y_val, y_test, feature_cols):
    """
    XGBoost Regressor for yield prediction.
    Early stopping: stops adding trees when validation RMSE
    hasn't improved for 50 rounds — prevents overfitting automatically.
    """
    print("\n" + "="*60)
    print("STEP 2 — XGBOOST REGRESSOR (Expected Yield)")
    print("="*60)
    print("  Early stopping: stops when val RMSE stagnates for 50 rounds")
    print("  This automatically selects the optimal number of trees\n")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Yield_Prediction")

    with mlflow.start_run(run_name="W8_XGBoost_Regressor_Champion"):

        xgb_reg = xgb.XGBRegressor(
            n_estimators=1000,         # max trees — early stopping will cut this
            learning_rate=0.05,        # shrinkage — each tree contributes 5%
            max_depth=6,               # tree depth — controls complexity
            subsample=0.8,             # row sampling — reduces overfitting
            colsample_bytree=0.8,      # feature sampling — like random forest trick
            reg_alpha=0.1,             # L1 regularization
            reg_lambda=1.0,            # L2 regularization
            random_state=RANDOM_STATE,
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbosity=0
        )

        xgb_reg.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        best_n = xgb_reg.best_iteration + 1
        print(f"  Early stopping selected: {best_n} trees (out of 1000 max)")
        print(f"  Best validation RMSE: {xgb_reg.best_score:.2f} kg/ha")

        y_pred    = xgb_reg.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2   = r2_score(y_test, y_pred)
        improvement = W5_RMSE - test_rmse

        print(f"\n  📊 Test Set Metrics:")
        print(f"     RMSE = {test_rmse:.2f} kg/ha")
        print(f"     R²   = {test_r2:.4f}")
        print(f"\n  {'✅ BEAT' if improvement > 0 else '❌ DID NOT BEAT'} W5 Ridge baseline!")
        print(f"  W5 RMSE: {W5_RMSE:.2f}  →  XGB RMSE: {test_rmse:.2f}  ({improvement:+.2f})")

        # Feature importance
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': xgb_reg.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n  Top 10 XGBoost feature importances (yield):")
        print(imp_df.head(10).to_string(index=False))

        mlflow.log_param("model",            "XGBoost_Regressor")
        mlflow.log_param("week",             "W8")
        mlflow.log_param("best_n_trees",     best_n)
        mlflow.log_param("learning_rate",    0.05)
        mlflow.log_param("max_depth",        6)
        mlflow.log_param("subsample",        0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_metric("val_rmse",        xgb_reg.best_score)
        mlflow.log_metric("test_rmse",       test_rmse)
        mlflow.log_metric("test_r2",         test_r2)
        mlflow.log_metric("w5_rmse",         W5_RMSE)
        mlflow.log_metric("improvement_vs_w5", improvement)
        mlflow.sklearn.log_model(xgb_reg, "xgb_regressor_model")
        print(f"\n  ✅ MLflow run logged → W8_XGBoost_Regressor_Champion")

    return xgb_reg, y_pred, test_rmse, test_r2, best_n, imp_df

# ── STEP 3: XGBOOST CLASSIFIER ────────────────────────────────────────────────
def run_xgb_classifier(X_train, X_val, X_test,
                       y_train, y_val, y_test, feature_cols):
    """
    XGBoost Classifier for disease outbreak.
    scale_pos_weight handles class imbalance — ratio of negative/positive.
    """
    print("\n" + "="*60)
    print("STEP 3 — XGBOOST CLASSIFIER (Disease Outbreak)")
    print("="*60)

    # Handle class imbalance
    neg   = (y_train == 0).sum()
    pos   = (y_train == 1).sum()
    scale = neg / pos
    print(f"  Class ratio: {neg:,} negative / {pos:,} positive")
    print(f"  scale_pos_weight = {scale:.2f} (balances classes automatically)\n")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Disease_Prediction")

    with mlflow.start_run(run_name="W8_XGBoost_Classifier_Champion"):

        xgb_cls = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,    # handles class imbalance
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric='auc',
            early_stopping_rounds=50,
            verbosity=0,
            use_label_encoder=False
        )

        xgb_cls.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        best_n = xgb_cls.best_iteration + 1
        print(f"  Early stopping selected: {best_n} trees")

        y_pred    = xgb_cls.predict(X_test)
        y_prob    = xgb_cls.predict_proba(X_test)[:,1]
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_prob)

        print(f"\n  📊 Test Set Metrics:")
        print(f"     Precision = {precision:.4f}")
        print(f"     Recall    = {recall:.4f}")
        print(f"     F1 Score  = {f1:.4f}")
        print(f"     AUC-ROC   = {auc:.4f}")

        beat = auc > W5_AUC
        print(f"\n  {'✅ BEAT' if beat else '❌ DID NOT BEAT'} W5 baseline!")
        print(f"  W5 AUC: {W5_AUC:.4f}  →  XGB AUC: {auc:.4f}  ({auc-W5_AUC:+.4f})")

        mlflow.log_param("model",              "XGBoost_Classifier")
        mlflow.log_param("week",               "W8")
        mlflow.log_param("best_n_trees",       best_n)
        mlflow.log_param("scale_pos_weight",   round(scale,2))
        mlflow.log_param("learning_rate",      0.05)
        mlflow.log_param("max_depth",          6)
        mlflow.log_metric("precision",         precision)
        mlflow.log_metric("recall",            recall)
        mlflow.log_metric("f1_score",          f1)
        mlflow.log_metric("auc_roc",           auc)
        mlflow.log_metric("w5_auc",            W5_AUC)
        mlflow.log_metric("auc_improvement",   auc - W5_AUC)
        mlflow.sklearn.log_model(xgb_cls, "xgb_classifier_model")
        print(f"\n  ✅ MLflow run logged → W8_XGBoost_Classifier_Champion")

    return xgb_cls, y_prob, auc, precision, recall, f1, best_n

# ── STEP 4: SHAP EXPLANATIONS ─────────────────────────────────────────────────
def run_shap(xgb_reg, xgb_cls, X_test, feature_cols):
    """
    SHAP — SHapley Additive exPlanations.
    For each prediction, SHAP tells you exactly how much each feature
    pushed the prediction up or down from the average.
    This is the gold standard for model explainability.
    """
    print("\n" + "="*60)
    print("STEP 4 — SHAP EXPLANATIONS")
    print("="*60)
    print("  SHAP tells us: for each prediction, how much did")
    print("  each feature contribute? (+ = pushed higher, - = pushed lower)\n")

    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Use a sample for speed
    sample_size = min(2000, X_test.shape[0])
    idx         = np.random.choice(X_test.shape[0], sample_size, replace=False)
    X_sample    = X_test[idx]

    # ── SHAP for yield regressor ────────────────────────────────────────────
    print("  Computing SHAP values for yield model...")
    explainer_reg  = shap.TreeExplainer(xgb_reg)
    shap_values_reg = explainer_reg.shap_values(X_sample)

    # SHAP summary plot — yield
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_reg, X_sample,
                      feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title('SHAP Summary — Yield Prediction (XGBoost W8)\n'
              'Red = feature pushes yield UP  ·  Blue = feature pushes yield DOWN',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f"{FIGURES_PATH}/28_shap_yield_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # SHAP bar plot — mean absolute SHAP values
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_reg, X_sample,
                      feature_names=feature_cols,
                      plot_type='bar', show=False,
                      max_display=15)
    plt.title('SHAP Feature Importance — Yield (Mean |SHAP|)\nCropCast W8',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f"{FIGURES_PATH}/29_shap_yield_bar.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # ── SHAP for disease classifier ─────────────────────────────────────────
    print("  Computing SHAP values for disease model...")
    explainer_cls   = shap.TreeExplainer(xgb_cls)
    shap_values_cls = explainer_cls.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_cls, X_sample,
                      feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title('SHAP Summary — Disease Outbreak Prediction (XGBoost W8)\n'
              'Red = pushes toward outbreak  ·  Blue = pushes away from outbreak',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = f"{FIGURES_PATH}/30_shap_disease_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Print top SHAP features
    mean_shap_yield   = np.abs(shap_values_reg).mean(axis=0)
    shap_yield_df     = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP': mean_shap_yield
    }).sort_values('Mean_SHAP', ascending=False)

    print(f"\n  Top 10 features by mean |SHAP| for yield:")
    print(shap_yield_df.head(10).to_string(index=False))

    return shap_yield_df

# ── STEP 5: ADDITIONAL PLOTS ──────────────────────────────────────────────────
def generate_additional_plots(xgb_reg, xgb_cls, X_test,
                               y_yield_test, y_dis_test, y_prob,
                               y_reg_pred, reg_rmse, auc,
                               reg_n_trees, cls_n_trees):
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Plot 31: Actual vs Predicted yield
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_yield_test, y_reg_pred, alpha=0.15, s=5,
               color='#2D6A4F', rasterized=True)
    lims = [min(y_yield_test.min(), y_reg_pred.min()),
            max(y_yield_test.max(), y_reg_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Yield (kg/ha)')
    ax.set_ylabel('Predicted Yield (kg/ha)')
    ax.set_title(f'XGBoost Champion — Actual vs Predicted Yield\n'
                 f'RMSE={reg_rmse:.0f} kg/ha  ({reg_n_trees} trees)  CropCast W8',
                 fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}/31_xgb_yield_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Plot 32: ROC curve — disease
    fig, ax = plt.subplots(figsize=(8, 7))
    RocCurveDisplay.from_predictions(
        y_dis_test, y_prob, ax=ax,
        color='#2D6A4F', name=f'XGBoost Champion W8 (AUC={auc:.4f})')
    ax.axhline(W5_AUC, color='#E07B39', linewidth=1.5,
               linestyle='--', label=f'W5 baseline AUC={W5_AUC}')
    ax.plot([0,1],[0,1],'k--', linewidth=1)
    ax.set_title('ROC Curve — XGBoost Champion vs All Baselines\nCropCast W8',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/32_xgb_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

# ── STEP 6: CHAMPION SUMMARY ──────────────────────────────────────────────────
def print_summary(reg_rmse, reg_r2, reg_n_trees,
                  auc, precision, recall, f1, cls_n_trees,
                  shap_yield_df):
    print("\n" + "█"*60)
    print("  WEEK 8 — XGBOOST CHAMPION RESULTS")
    print("█"*60)

    print(f"\n  REGRESSION — Expected Yield")
    print(f"  {'Model':<38} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*56}")
    print(f"  {'W3 OLS (no feature engineering)':<38} {'1312.10':>8}  {'0.2779':>8}")
    print(f"  {'W5 Ridge (feature eng + 1-SE CV)':<38} {W5_RMSE:>8.2f}  {'0.6174':>8}")
    print(f"  {'W8 XGBoost Champion':<38} {reg_rmse:>8.2f}  {reg_r2:>8.4f}")
    print(f"\n  Total improvement W3→W8: {1312.10-reg_rmse:+.2f} kg/ha")
    print(f"  Trees used (early stopping): {reg_n_trees}")

    print(f"\n  CLASSIFICATION — Disease Outbreak")
    print(f"  {'Metric':<20} {'W5 Logistic':>12} {'W8 XGBoost':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Precision':<20} {'0.6278':>12} {precision:>12.4f}")
    print(f"  {'Recall':<20} {'0.8212':>12} {recall:>12.4f}")
    print(f"  {'F1 Score':<20} {'0.7116':>12} {f1:>12.4f}")
    print(f"  {'AUC-ROC':<20} {W5_AUC:>12.4f} {auc:>12.4f}")
    print(f"\n  Trees used (early stopping): {cls_n_trees}")

    print(f"\n  SHAP — Top 5 yield predictors:")
    for _, row in shap_yield_df.head(5).iterrows():
        print(f"    {row['Feature']:<30} Mean|SHAP|={row['Mean_SHAP']:.2f}")

    print(f"\n  5 plots saved: 28, 29, 30, 31, 32")
    print(f"\n  🏆 XGBoost is the CropCast champion model.")
    print(f"     It will be packaged in W11 and deployed in W12.")
    print(f"\n  Next → Week 9: K-Means/DBSCAN clustering into")
    print(f"                  agro-climatic zones + feed zone label")
    print(f"                  back as a feature into yield and disease models.")
    print("█"*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 8: XGBOOST CHAMPION")
    print("  Early Stopping · SHAP Explanations · Champion Model")
    print("█"*60)

    (X_train, X_val, X_test,
     y_y_train, y_y_val, y_y_test,
     y_d_train, y_d_val, y_d_test,
     feature_cols)              = load_and_engineer()

    (xgb_reg, y_reg_pred,
     reg_rmse, reg_r2,
     reg_n_trees, reg_imp_df)   = run_xgb_regressor(
                                    X_train, X_val, X_test,
                                    y_y_train, y_y_val, y_y_test,
                                    feature_cols)

    (xgb_cls, y_prob,
     auc, precision,
     recall, f1, cls_n_trees)   = run_xgb_classifier(
                                    X_train, X_val, X_test,
                                    y_d_train, y_d_val, y_d_test,
                                    feature_cols)

    shap_yield_df               = run_shap(xgb_reg, xgb_cls,
                                            X_test, feature_cols)

    generate_additional_plots(xgb_reg, xgb_cls, X_test,
                               y_y_test, y_d_test, y_prob,
                               y_reg_pred, reg_rmse, auc,
                               reg_n_trees, cls_n_trees)

    print_summary(reg_rmse, reg_r2, reg_n_trees,
                  auc, precision, recall, f1, cls_n_trees,
                  shap_yield_df)