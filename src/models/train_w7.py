# =============================================================================
# CropCast — Week 7: Random Forest
# File: src/models/train_w7.py
#
# Requirements:
# 1. Train Random Forest on both targets (yield + disease)
# 2. Report OOB error
# 3. Rank features by permutation importance
# 4. Must beat W6: AUC > W6 result, RMSE < 955.14
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             precision_score, recall_score,
                             f1_score, roc_auc_score, RocCurveDisplay)
from sklearn.inspection import permutation_importance

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

# ── FEATURE ENGINEERING (identical to W4/W5/W6) ───────────────────────────────
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

    (X_train, X_test,
     y_yield_train, y_yield_test,
     y_dis_train, y_dis_test) = train_test_split(
        X, y_yield, y_disease,
        test_size=0.20, random_state=RANDOM_STATE)

    # Trees don't need scaling but keep pipeline consistent
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    if not np.isfinite(X_train_s).all():
        X_train_s = np.nan_to_num(X_train_s)
        X_test_s  = np.nan_to_num(X_test_s)

    print(f"✅ Train: {X_train_s.shape[0]:,}  Test: {X_test_s.shape[0]:,}  "
          f"Features: {X_train_s.shape[1]}")
    return (X_train_s, X_test_s,
            y_yield_train, y_yield_test,
            y_dis_train, y_dis_test,
            feature_cols)

# ── STEP 2: RANDOM FOREST — YIELD REGRESSION ──────────────────────────────────
def run_rf_regressor(X_train, X_test, y_train, y_test, feature_cols):
    """
    Random Forest Regressor for expected yield.
    Key settings:
    - oob_score=True: use out-of-bag samples for free validation
    - n_estimators=300: 300 trees — enough for stable OOB estimate
    - max_features='sqrt': each split considers sqrt(39) ≈ 6 features
      This decorrelates trees — the key advantage over plain bagging
    """
    print("\n" + "="*60)
    print("STEP 2 — RANDOM FOREST REGRESSOR (Expected Yield)")
    print("="*60)
    print("  n_estimators=300 · max_features=sqrt · oob_score=True")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Yield_Prediction")

    with mlflow.start_run(run_name="W7_RandomForest_Regressor"):
        rf_reg = RandomForestRegressor(
            n_estimators=300,
            max_features='sqrt',
            max_depth=None,
            min_samples_leaf=5,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        rf_reg.fit(X_train, y_train)

        # OOB score — free validation without touching test set
        oob_r2   = rf_reg.oob_score_
        oob_rmse = np.sqrt(mean_squared_error(
            y_train,
            rf_reg.oob_prediction_
        ))
        print(f"\n  OOB R²   = {oob_r2:.4f}  (free validation on ~37% of train data)")
        print(f"  OOB RMSE = {oob_rmse:.2f} kg/ha")

        # Test set evaluation
        y_pred      = rf_reg.predict(X_test)
        test_rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2     = r2_score(y_test, y_pred)
        improvement = W5_RMSE - test_rmse

        print(f"\n  📊 Test Set Metrics:")
        print(f"     RMSE = {test_rmse:.2f} kg/ha")
        print(f"     R²   = {test_r2:.4f}")
        print(f"\n  {'✅ BEAT' if improvement > 0 else '❌ DID NOT BEAT'} W5 Ridge baseline!")
        print(f"  W5 RMSE: {W5_RMSE:.2f}  →  RF RMSE: {test_rmse:.2f}  ({improvement:+.2f})")

        # Gini feature importance
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Gini_Importance': rf_reg.feature_importances_
        }).sort_values('Gini_Importance', ascending=False)
        print(f"\n  Top 10 features by Gini importance (yield):")
        print(imp_df.head(10).to_string(index=False))

        # Permutation importance — more reliable than Gini
        print(f"\n  Computing permutation importance (this takes a moment)...")
        perm = permutation_importance(
            rf_reg, X_test, y_test,
            n_repeats=10, random_state=RANDOM_STATE,
            scoring='neg_root_mean_squared_error'
        )
        perm_df = pd.DataFrame({
            'Feature': feature_cols,
            'Perm_Importance': perm.importances_mean,
            'Perm_Std': perm.importances_std
        }).sort_values('Perm_Importance', ascending=False)
        print(f"\n  Top 10 features by Permutation importance (yield):")
        print(perm_df.head(10).to_string(index=False))

        mlflow.log_param("model",          "RandomForestRegressor")
        mlflow.log_param("week",           "W7")
        mlflow.log_param("n_estimators",   300)
        mlflow.log_param("max_features",   "sqrt")
        mlflow.log_param("min_samples_leaf", 5)
        mlflow.log_metric("oob_r2",        oob_r2)
        mlflow.log_metric("oob_rmse",      oob_rmse)
        mlflow.log_metric("test_rmse",     test_rmse)
        mlflow.log_metric("test_r2",       test_r2)
        mlflow.log_metric("w5_rmse",       W5_RMSE)
        mlflow.log_metric("improvement_vs_w5", improvement)
        mlflow.sklearn.log_model(rf_reg, "rf_regressor_model")
        print(f"\n  ✅ MLflow run logged → W7_RandomForest_Regressor")

    return rf_reg, y_pred, test_rmse, test_r2, oob_rmse, oob_r2, imp_df, perm_df

# ── STEP 3: RANDOM FOREST — DISEASE CLASSIFIER ────────────────────────────────
def run_rf_classifier(X_train, X_test, y_train, y_test, feature_cols):
    """
    Random Forest Classifier for disease outbreak.
    class_weight='balanced' handles 75/25 imbalance.
    """
    print("\n" + "="*60)
    print("STEP 3 — RANDOM FOREST CLASSIFIER (Disease Outbreak)")
    print("="*60)
    print("  n_estimators=300 · max_features=sqrt · class_weight=balanced")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Disease_Prediction")

    with mlflow.start_run(run_name="W7_RandomForest_Classifier"):
        rf_cls = RandomForestClassifier(
            n_estimators=300,
            max_features='sqrt',
            max_depth=None,
            min_samples_leaf=5,
            class_weight='balanced',
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        rf_cls.fit(X_train, y_train)

        # OOB
        oob_acc = rf_cls.oob_score_
        print(f"\n  OOB Accuracy = {oob_acc:.4f}  (free validation on ~37% of train)")

        # Test set
        y_pred    = rf_cls.predict(X_test)
        y_prob    = rf_cls.predict_proba(X_test)[:,1]
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
        print(f"\n  {'✅ BEAT' if beat else '❌ DID NOT BEAT'} W5 Logistic baseline!")
        print(f"  W5 AUC: {W5_AUC:.4f}  →  RF AUC: {auc:.4f}  ({auc-W5_AUC:+.4f})")

        # Gini importance
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Gini_Importance': rf_cls.feature_importances_
        }).sort_values('Gini_Importance', ascending=False)
        print(f"\n  Top 10 features by Gini importance (disease):")
        print(imp_df.head(10).to_string(index=False))

        # Permutation importance
        print(f"\n  Computing permutation importance...")
        perm = permutation_importance(
            rf_cls, X_test, y_test,
            n_repeats=10, random_state=RANDOM_STATE,
            scoring='roc_auc'
        )
        perm_df = pd.DataFrame({
            'Feature': feature_cols,
            'Perm_Importance': perm.importances_mean,
            'Perm_Std': perm.importances_std
        }).sort_values('Perm_Importance', ascending=False)
        print(f"\n  Top 10 features by Permutation importance (disease):")
        print(perm_df.head(10).to_string(index=False))

        mlflow.log_param("model",           "RandomForestClassifier")
        mlflow.log_param("week",            "W7")
        mlflow.log_param("n_estimators",    300)
        mlflow.log_param("max_features",    "sqrt")
        mlflow.log_param("class_weight",    "balanced")
        mlflow.log_metric("oob_accuracy",   oob_acc)
        mlflow.log_metric("precision",      precision)
        mlflow.log_metric("recall",         recall)
        mlflow.log_metric("f1_score",       f1)
        mlflow.log_metric("auc_roc",        auc)
        mlflow.log_metric("w5_auc",         W5_AUC)
        mlflow.log_metric("auc_improvement", auc - W5_AUC)
        mlflow.sklearn.log_model(rf_cls, "rf_classifier_model")
        print(f"\n  ✅ MLflow run logged → W7_RandomForest_Classifier")

    return rf_cls, y_prob, auc, precision, recall, f1, oob_acc, imp_df, perm_df

# ── STEP 4: VISUALISATIONS ────────────────────────────────────────────────────
def generate_plots(reg_perm_df, cls_perm_df, cls_gini_df,
                   y_dis_test, y_prob, auc,
                   rf_reg, y_yield_test, y_reg_pred, reg_rmse,
                   oob_rmse, oob_acc):
    print("\n" + "="*60)
    print("STEP 4 — GENERATING VISUALISATIONS")
    print("="*60)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Plot 24: Permutation importance — yield (top 20)
    top20 = reg_perm_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20['Feature'], top20['Perm_Importance'],
            xerr=top20['Perm_Std'],
            color='#2D6A4F', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Permutation Importance (RMSE increase when feature shuffled)')
    ax.set_title('Random Forest — Permutation Feature Importance (Yield)\nCropCast W7',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/24_rf_perm_importance_yield.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Plot 25: Permutation importance — disease (top 20)
    top20d = cls_perm_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20d['Feature'], top20d['Perm_Importance'],
            xerr=top20d['Perm_Std'],
            color='#E07B39', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Permutation Importance (AUC drop when feature shuffled)')
    ax.set_title('Random Forest — Permutation Feature Importance (Disease)\nCropCast W7',
                 fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/25_rf_perm_importance_disease.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Plot 26: Actual vs Predicted — yield
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_yield_test, y_reg_pred, alpha=0.15, s=5,
               color='#2D6A4F', rasterized=True)
    lims = [min(y_yield_test.min(), y_reg_pred.min()),
            max(y_yield_test.max(), y_reg_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Yield (kg/ha)')
    ax.set_ylabel('Predicted Yield (kg/ha)')
    ax.set_title(f'Random Forest — Actual vs Predicted Yield\n'
                 f'RMSE={reg_rmse:.0f} kg/ha  OOB RMSE={oob_rmse:.0f}  CropCast W7',
                 fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}/26_rf_yield_actual_vs_predicted.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Plot 27: ROC curve — disease
    fig, ax = plt.subplots(figsize=(8, 7))
    RocCurveDisplay.from_predictions(
        y_dis_test, y_prob, ax=ax,
        color='#E07B39', name=f'Random Forest W7 (AUC={auc:.4f})')
    ax.axhline(W5_AUC, color='#2D6A4F', linewidth=1.5,
               linestyle='--', label=f'W5 baseline AUC={W5_AUC}')
    ax.plot([0,1],[0,1],'k--', linewidth=1)
    ax.set_title('ROC Curve — Random Forest vs W5 Baseline\nCropCast W7',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/27_rf_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

# ── STEP 5: SUMMARY ───────────────────────────────────────────────────────────
def print_summary(reg_rmse, reg_r2, oob_rmse, oob_r2,
                  auc, precision, recall, f1, oob_acc,
                  reg_perm_df, cls_perm_df):
    print("\n" + "█"*60)
    print("  WEEK 7 RESULTS SUMMARY")
    print("█"*60)

    print(f"\n  REGRESSION — Expected Yield")
    print(f"  {'Model':<35} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*53}")
    print(f"  {'W5 Ridge (1-SE baseline)':<35} {W5_RMSE:>8.2f}  {'0.6174':>8}")
    print(f"  {'W7 Random Forest (OOB)':<35} {oob_rmse:>8.2f}  {oob_r2:>8.4f}")
    print(f"  {'W7 Random Forest (Test)':<35} {reg_rmse:>8.2f}  {reg_r2:>8.4f}")
    print(f"\n  Improvement vs W5: {W5_RMSE - reg_rmse:+.2f} kg/ha")

    print(f"\n  CLASSIFICATION — Disease Outbreak")
    print(f"  {'Metric':<20} {'W5 Logistic':>12} {'W7 RF':>12}")
    print(f"  {'-'*46}")
    print(f"  {'OOB Accuracy':<20} {'—':>12} {oob_acc:>12.4f}")
    print(f"  {'Precision':<20} {'0.6278':>12} {precision:>12.4f}")
    print(f"  {'Recall':<20} {'0.8212':>12} {recall:>12.4f}")
    print(f"  {'F1 Score':<20} {'0.7116':>12} {f1:>12.4f}")
    print(f"  {'AUC-ROC':<20} {W5_AUC:>12.4f} {auc:>12.4f}")

    print(f"\n  Top 5 yield predictors (permutation):")
    for _, row in reg_perm_df.head(5).iterrows():
        print(f"    {row['Feature']:<30} {row['Perm_Importance']:.4f}")

    print(f"\n  Top 5 disease predictors (permutation):")
    for _, row in cls_perm_df.head(5).iterrows():
        print(f"    {row['Feature']:<30} {row['Perm_Importance']:.4f}")

    print(f"\n  4 plots saved: 24, 25, 26, 27")
    print(f"\n  Next → Week 8: XGBoost champion + early stopping + SHAP")
    print("█"*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 7: RANDOM FOREST")
    print("  Cut variance · OOB error · Permutation importance")
    print("█"*60)

    (X_train, X_test,
     y_yield_train, y_yield_test,
     y_dis_train, y_dis_test,
     feature_cols)              = load_and_engineer()

    (rf_reg, y_reg_pred,
     reg_rmse, reg_r2,
     oob_rmse, oob_r2,
     reg_gini_df,
     reg_perm_df)               = run_rf_regressor(
                                    X_train, X_test,
                                    y_yield_train, y_yield_test,
                                    feature_cols)

    (rf_cls, y_prob,
     auc, precision, recall, f1,
     oob_acc,
     cls_gini_df,
     cls_perm_df)               = run_rf_classifier(
                                    X_train, X_test,
                                    y_dis_train, y_dis_test,
                                    feature_cols)

    generate_plots(reg_perm_df, cls_perm_df, cls_gini_df,
                   y_dis_test, y_prob, auc,
                   rf_reg, y_yield_test, y_reg_pred, reg_rmse,
                   oob_rmse, oob_acc)

    print_summary(reg_rmse, reg_r2, oob_rmse, oob_r2,
                  auc, precision, recall, f1, oob_acc,
                  reg_perm_df, cls_perm_df)