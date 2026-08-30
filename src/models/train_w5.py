# =============================================================================
# CropCast — Week 5: Cross-Validation + One-Standard-Error Rule + Milestone
# File: src/models/train_w5.py
#
# Three deliverables:
# 1. Cross-validation on Ridge, Lasso, Logistic Regression
# 2. One-standard-error rule — pick simplest model within 1 SE of best
# 3. Ship the honest baseline milestone — seal and document final numbers
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.model_selection import (train_test_split, KFold,
                                     cross_val_score, StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score,
                             precision_score, recall_score,
                             f1_score, roc_auc_score)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/clean_dataset.csv"
FIGURES_PATH    = "/Users/amrkalammansoori/Desktop/crop-cast/reports/figures"
MLFLOW_DIR      = "/Users/amrkalammansoori/Desktop/crop-cast/mlruns"

REGRESSION_TARGET     = "expected_yield_kg_ha"
CLASSIFICATION_TARGET = "disease_outbreak"
RANDOM_STATE          = 42
N_FOLDS               = 5

# W3 & W4 confirmed baselines
W3_OLS_RMSE   = 1312.10
W4_RIDGE_RMSE = 955.15
W4_LASSO_RMSE = 955.14
W4_AUC_ROC    = 0.9155

# ── STEP 1: LOAD & ENGINEER FEATURES (same as W4) ─────────────────────────────
def load_and_engineer():
    print("\n" + "="*60)
    print("STEP 1 — LOAD DATA & ENGINEER FEATURES (same as W4)")
    print("="*60)

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    fe = df.copy()

    # Engineered features (identical to W4 — reproducibility)
    fe["npk_total"]           = (0.5*(fe["soil_nitrogen"]/fe["soil_nitrogen"].max()) +
                                  0.3*(fe["soil_phosphorus"]/fe["soil_phosphorus"].max()) +
                                  0.2*(fe["soil_potassium"]/fe["soil_potassium"].max()))
    fe["ph_optimality"]       = np.exp(-0.5*((fe["soil_ph"]-6.5)/0.8)**2)
    fe["heat_humidity_index"] = (np.clip((fe["temperature"]-20)/15,0,1) *
                                  np.clip((fe["humidity"]-55)/40,0,1))
    fe["rainfall_optimality"] = np.exp(-0.00005*(fe["rainfall"]-115)**2)
    fe["rainfall_category"]   = pd.cut(fe["rainfall"],bins=[0,50,80,150,380],
                                         labels=[0,1,2,3]).astype(float).fillna(1)
    fe["ndvi_stress_flag"]    = (fe["ndvi"] < 0.35).astype(int)
    fe["ndvi_npk_interaction"]= fe["ndvi"] * fe["npk_total"]
    fe["sowing_delay_bucket"] = pd.cut(fe["sowing_delay_days"],bins=[-1,7,14,21,45],
                                         labels=[0,1,2,3]).astype(float).fillna(0)

    cat_cols = ["crop_type","season","agro_zone","region"]
    fe = pd.get_dummies(fe, columns=cat_cols, drop_first=True, dtype=float)

    drop_cols    = [REGRESSION_TARGET, CLASSIFICATION_TARGET, "crop_growth_stage"]
    feature_cols = [c for c in fe.columns if c not in drop_cols]

    X          = fe[feature_cols].values
    y_yield    = fe[REGRESSION_TARGET].values
    y_disease  = fe[CLASSIFICATION_TARGET].values

    # Same 80/20 split — test set remains sealed for final evaluation only
    X_train, X_test, y_yield_train, y_yield_test, y_dis_train, y_dis_test = \
        train_test_split(X, y_yield, y_disease,
                         test_size=0.20, random_state=RANDOM_STATE)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if not np.isfinite(X_train_s).all():
        X_train_s = np.nan_to_num(X_train_s)
        X_test_s  = np.nan_to_num(X_test_s)

    print(f"✅ Train: {X_train_s.shape[0]:,}  Test: {X_test_s.shape[0]:,}  "
          f"Features: {X_train_s.shape[1]}")
    print(f"✅ Feature engineering reproduced — identical to W4")

    return (X_train_s, X_test_s,
            y_yield_train, y_yield_test,
            y_dis_train, y_dis_test,
            feature_cols)

# ── STEP 2: CV ON RIDGE — ALPHA SWEEP ────────────────────────────────────────
def cv_ridge(X_train, y_train):
    """
    5-fold cross-validation across Ridge alpha values.
    Reports: mean CV RMSE ± 1 standard error for each alpha.
    Applies the one-standard-error rule to select the simplest good model.
    """
    print("\n" + "="*60)
    print("STEP 2 — RIDGE: 5-FOLD CROSS-VALIDATION + 1-SE RULE")
    print("="*60)
    print("  Evaluating RMSE stability across folds — not just mean performance\n")

    alphas   = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    kf       = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results  = []

    print(f"  {'Alpha':<10} {'Mean RMSE':>12} {'Std RMSE':>12} {'SE':>10}")
    print(f"  {'-'*46}")

    for alpha in alphas:
        model     = Ridge(alpha=alpha)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=kf, scoring='neg_mean_squared_error')
        cv_rmses  = np.sqrt(-cv_scores)
        mean_rmse = cv_rmses.mean()
        std_rmse  = cv_rmses.std()
        se        = std_rmse / np.sqrt(N_FOLDS)
        results.append({'alpha':alpha,'mean':mean_rmse,'std':std_rmse,'se':se})
        print(f"  {alpha:<10} {mean_rmse:>12.2f} {std_rmse:>12.2f} {se:>10.2f}")

    df_res   = pd.DataFrame(results)
    best_idx = df_res['mean'].idxmin()
    best_mean= df_res.loc[best_idx,'mean']
    best_se  = df_res.loc[best_idx,'se']
    threshold= best_mean + best_se    # 1-SE rule threshold

    # One-SE rule: pick the LARGEST alpha (simplest/most regularised)
    # whose mean CV RMSE is within threshold
    eligible = df_res[df_res['mean'] <= threshold]
    ose_alpha= eligible['alpha'].max()  # largest alpha = most regularised = simplest

    print(f"\n  Best mean CV RMSE : {best_mean:.2f} (α={df_res.loc[best_idx,'alpha']})")
    print(f"  1-SE threshold    : {threshold:.2f}")
    print(f"  1-SE Rule selects : α = {ose_alpha}  (simplest model within threshold)")
    print(f"\n  Why the 1-SE Rule?")
    print(f"  If two models differ by less than 1 standard error, we cannot")
    print(f"  statistically distinguish them. We pick the simpler one (higher α)")
    print(f"  to avoid overfitting to the cross-validation process itself.")

    return df_res, ose_alpha, best_mean, best_se, threshold

# ── STEP 3: CV ON LASSO — ALPHA SWEEP ────────────────────────────────────────
def cv_lasso(X_train, y_train):
    print("\n" + "="*60)
    print("STEP 3 — LASSO: 5-FOLD CROSS-VALIDATION + 1-SE RULE")
    print("="*60)
    print("  L1 penalty — also tracks how many features survive each alpha\n")

    alphas  = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    print(f"  {'Alpha':<10} {'Mean RMSE':>12} {'SE':>10} {'Features kept':>14}")
    print(f"  {'-'*50}")

    for alpha in alphas:
        model     = Lasso(alpha=alpha, max_iter=5000)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=kf, scoring='neg_mean_squared_error')
        cv_rmses  = np.sqrt(-cv_scores)
        mean_rmse = cv_rmses.mean()
        std_rmse  = cv_rmses.std()
        se        = std_rmse / np.sqrt(N_FOLDS)

        # Fit once to count non-zero features
        model.fit(X_train, y_train)
        n_kept = int(np.sum(model.coef_ != 0))
        results.append({'alpha':alpha,'mean':mean_rmse,'std':std_rmse,
                         'se':se,'n_kept':n_kept})
        print(f"  {alpha:<10} {mean_rmse:>12.2f} {se:>10.2f} {n_kept:>14}")

    df_res   = pd.DataFrame(results)
    best_idx = df_res['mean'].idxmin()
    best_mean= df_res.loc[best_idx,'mean']
    best_se  = df_res.loc[best_idx,'se']
    threshold= best_mean + best_se

    eligible = df_res[df_res['mean'] <= threshold]
    ose_alpha= eligible['alpha'].max()
    ose_kept = int(df_res[df_res['alpha']==ose_alpha]['n_kept'].values[0])

    print(f"\n  Best mean CV RMSE : {best_mean:.2f} (α={df_res.loc[best_idx,'alpha']})")
    print(f"  1-SE threshold    : {threshold:.2f}")
    print(f"  1-SE Rule selects : α={ose_alpha}  ({ose_kept} features retained)")

    return df_res, ose_alpha, best_mean, best_se, threshold

# ── STEP 4: CV ON LOGISTIC REGRESSION ────────────────────────────────────────
def cv_logistic(X_train, y_train):
    """
    5-fold stratified CV on Logistic Regression.
    Stratified = each fold preserves the 75/25 class ratio.
    Reports AUC-ROC (primary metric) across folds.
    """
    print("\n" + "="*60)
    print("STEP 4 — LOGISTIC REGRESSION: STRATIFIED 5-FOLD CV")
    print("="*60)
    print("  Stratified folds preserve 75/25 class ratio in each fold")
    print("  Primary metric: AUC-ROC (not accuracy)\n")

    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    results  = []

    print(f"  {'C (1/λ)':<12} {'Mean AUC':>10} {'Std AUC':>10} {'SE':>8}")
    print(f"  {'-'*42}")

    for C in C_values:
        model     = LogisticRegression(C=C, class_weight='balanced',
                                       max_iter=1000, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=skf, scoring='roc_auc')
        mean_auc  = cv_scores.mean()
        std_auc   = cv_scores.std()
        se        = std_auc / np.sqrt(N_FOLDS)
        results.append({'C':C,'mean':mean_auc,'std':std_auc,'se':se})
        print(f"  {C:<12} {mean_auc:>10.4f} {std_auc:>10.4f} {se:>8.4f}")

    df_res   = pd.DataFrame(results)
    best_idx = df_res['mean'].idxmax()   # higher AUC = better
    best_mean= df_res.loc[best_idx,'mean']
    best_se  = df_res.loc[best_idx,'se']
    threshold= best_mean - best_se       # within 1-SE below best

    # 1-SE rule for AUC: pick smallest C (most regularised) within threshold
    eligible  = df_res[df_res['mean'] >= threshold]
    ose_C     = eligible['C'].min()

    print(f"\n  Best mean CV AUC  : {best_mean:.4f} (C={df_res.loc[best_idx,'C']})")
    print(f"  1-SE threshold    : {threshold:.4f}")
    print(f"  1-SE Rule selects : C = {ose_C}  (most regularised within threshold)")

    return df_res, ose_C, best_mean, best_se

# ── STEP 5: FINAL MODELS ON SEALED TEST SET ───────────────────────────────────
def evaluate_final_models(X_train, X_test,
                           y_yield_train, y_yield_test,
                           y_dis_train, y_dis_test,
                           ridge_alpha, lasso_alpha, lr_C):
    """
    Train final models with 1-SE selected hyperparameters.
    Evaluate ONCE on the sealed test set.
    This is the honest milestone — these numbers go in the presentation.
    """
    print("\n" + "="*60)
    print("STEP 5 — FINAL EVALUATION ON SEALED TEST SET")
    print("="*60)
    print("  ⚠️  Test set used ONCE — these are the milestone numbers\n")

    # ── Ridge final ────────────────────────────────────────────────────────
    ridge = Ridge(alpha=ridge_alpha)
    ridge.fit(X_train, y_yield_train)
    ridge_pred  = ridge.predict(X_test)
    ridge_rmse  = np.sqrt(mean_squared_error(y_yield_test, ridge_pred))
    ridge_r2    = r2_score(y_yield_test, ridge_pred)

    # ── Lasso final ────────────────────────────────────────────────────────
    lasso = Lasso(alpha=lasso_alpha, max_iter=10000)
    lasso.fit(X_train, y_yield_train)
    lasso_pred  = lasso.predict(X_test)
    lasso_rmse  = np.sqrt(mean_squared_error(y_yield_test, lasso_pred))
    lasso_r2    = r2_score(y_yield_test, lasso_pred)
    lasso_zeros = int(np.sum(lasso.coef_ == 0))
    lasso_kept  = int(np.sum(lasso.coef_ != 0))

    # ── Logistic Regression final ──────────────────────────────────────────
    lr = LogisticRegression(C=lr_C, class_weight='balanced',
                             max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_dis_train)
    y_prob       = lr.predict_proba(X_test)[:,1]
    y_pred_class = lr.predict(X_test)
    lr_precision = precision_score(y_dis_test, y_pred_class, zero_division=0)
    lr_recall    = recall_score(y_dis_test, y_pred_class, zero_division=0)
    lr_f1        = f1_score(y_dis_test, y_pred_class, zero_division=0)
    lr_auc       = roc_auc_score(y_dis_test, y_prob)

    print(f"  REGRESSION MILESTONE:")
    print(f"  {'Model':<35} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*53}")
    print(f"  {'W3 OLS (no feature eng)':<35} {W3_OLS_RMSE:>8.2f}  {'0.2779':>8}")
    print(f"  {'W5 Ridge (α='+str(ridge_alpha)+', 1-SE selected)':<35} {ridge_rmse:>8.2f}  {ridge_r2:>8.4f}")
    print(f"  {'W5 Lasso (α='+str(lasso_alpha)+', 1-SE selected)':<35} {lasso_rmse:>8.2f}  {lasso_r2:>8.4f}")
    print(f"\n  Lasso eliminated {lasso_zeros} features, retained {lasso_kept}")

    print(f"\n  CLASSIFICATION MILESTONE:")
    print(f"  {'Metric':<20} {'Value':>8}")
    print(f"  {'-'*30}")
    print(f"  {'Precision':<20} {lr_precision:>8.4f}")
    print(f"  {'Recall':<20} {lr_recall:>8.4f}")
    print(f"  {'F1 Score':<20} {lr_f1:>8.4f}")
    print(f"  {'AUC-ROC':<20} {lr_auc:>8.4f}")

    return (ridge, ridge_rmse, ridge_r2,
            lasso, lasso_rmse, lasso_r2, lasso_zeros, lasso_kept,
            lr, lr_precision, lr_recall, lr_f1, lr_auc, y_prob)

# ── STEP 6: LOG MILESTONE TO MLFLOW ──────────────────────────────────────────
def log_milestone(ridge_alpha, ridge_rmse, ridge_r2,
                  lasso_alpha, lasso_rmse, lasso_r2, lasso_zeros, lasso_kept,
                  lr_C, lr_precision, lr_recall, lr_f1, lr_auc):
    print("\n" + "="*60)
    print("STEP 6 — LOGGING MILESTONE TO MLFLOW")
    print("="*60)

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

    # Ridge milestone
    mlflow.set_experiment("CropCast_Yield_Prediction")
    with mlflow.start_run(run_name="W5_Ridge_1SE_Milestone"):
        mlflow.log_param("model",          "Ridge_1SE")
        mlflow.log_param("week",           "W5")
        mlflow.log_param("alpha_1se",      ridge_alpha)
        mlflow.log_param("cv_folds",       N_FOLDS)
        mlflow.log_param("rule",           "one_standard_error")
        mlflow.log_metric("test_rmse",     ridge_rmse)
        mlflow.log_metric("test_r2",       ridge_r2)
        mlflow.log_metric("w3_ols_rmse",   W3_OLS_RMSE)
        mlflow.log_metric("improvement_vs_w3", W3_OLS_RMSE - ridge_rmse)
        print(f"  ✅ Logged → W5_Ridge_1SE_Milestone")

    # Lasso milestone
    with mlflow.start_run(run_name="W5_Lasso_1SE_Milestone"):
        mlflow.log_param("model",           "Lasso_1SE")
        mlflow.log_param("week",            "W5")
        mlflow.log_param("alpha_1se",       lasso_alpha)
        mlflow.log_param("cv_folds",        N_FOLDS)
        mlflow.log_param("rule",            "one_standard_error")
        mlflow.log_param("features_zeroed", lasso_zeros)
        mlflow.log_param("features_kept",   lasso_kept)
        mlflow.log_metric("test_rmse",      lasso_rmse)
        mlflow.log_metric("test_r2",        lasso_r2)
        mlflow.log_metric("w3_ols_rmse",    W3_OLS_RMSE)
        mlflow.log_metric("improvement_vs_w3", W3_OLS_RMSE - lasso_rmse)
        print(f"  ✅ Logged → W5_Lasso_1SE_Milestone")

    # Logistic milestone
    mlflow.set_experiment("CropCast_Disease_Prediction")
    with mlflow.start_run(run_name="W5_Logistic_1SE_Milestone"):
        mlflow.log_param("model",         "LogisticRegression_1SE")
        mlflow.log_param("week",          "W5")
        mlflow.log_param("C_1se",         lr_C)
        mlflow.log_param("cv_folds",      N_FOLDS)
        mlflow.log_param("rule",          "one_standard_error")
        mlflow.log_param("class_weight",  "balanced")
        mlflow.log_metric("precision",    lr_precision)
        mlflow.log_metric("recall",       lr_recall)
        mlflow.log_metric("f1_score",     lr_f1)
        mlflow.log_metric("auc_roc",      lr_auc)
        mlflow.log_metric("w4_auc_roc",   W4_AUC_ROC)
        print(f"  ✅ Logged → W5_Logistic_1SE_Milestone")

# ── STEP 7: VISUALISATIONS ────────────────────────────────────────────────────
def generate_plots(ridge_cv_df, lasso_cv_df, lr_cv_df,
                   ridge_threshold, lasso_threshold,
                   ridge_ose_alpha, lasso_ose_alpha, lr_ose_C,
                   ridge_rmse, lasso_rmse, lr_auc):
    print("\n" + "="*60)
    print("STEP 7 — GENERATING VISUALISATIONS")
    print("="*60)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # ── Plot 17: Ridge CV — RMSE vs Alpha with 1-SE band ──────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(range(len(ridge_cv_df)),
                ridge_cv_df['mean'],
                yerr=ridge_cv_df['se'],
                fmt='o-', color='#2D6A4F', linewidth=2,
                capsize=5, label='Mean CV RMSE ± 1 SE')
    ax.axhline(ridge_threshold, color='#E07B39', linewidth=1.5,
               linestyle='--', label=f'1-SE threshold = {ridge_threshold:.0f}')
    best_idx = ridge_cv_df['mean'].idxmin()
    ax.scatter(best_idx, ridge_cv_df.loc[best_idx,'mean'],
               color='red', s=120, zorder=5, label='Best CV alpha')
    ose_idx = ridge_cv_df[ridge_cv_df['alpha']==ridge_ose_alpha].index[0]
    ax.scatter(ose_idx, ridge_cv_df.loc[ose_idx,'mean'],
               color='#52B788', s=150, marker='*', zorder=6,
               label=f'1-SE selected α={ridge_ose_alpha}')
    ax.set_xticks(range(len(ridge_cv_df)))
    ax.set_xticklabels([str(a) for a in ridge_cv_df['alpha']])
    ax.set_xlabel('Ridge Alpha (λ)')
    ax.set_ylabel('CV RMSE (kg/ha)')
    ax.set_title('Ridge: 5-Fold CV RMSE vs Alpha\nOne-Standard-Error Rule Applied',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/17_ridge_cv_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 18: Lasso CV — RMSE vs Alpha with features kept ──────────────
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx()
    ax1.errorbar(range(len(lasso_cv_df)),
                 lasso_cv_df['mean'],
                 yerr=lasso_cv_df['se'],
                 fmt='o-', color='#5FA8D3', linewidth=2,
                 capsize=5, label='Mean CV RMSE ± 1 SE')
    ax1.axhline(lasso_threshold, color='#E07B39', linewidth=1.5,
                linestyle='--', label=f'1-SE threshold')
    ose_idx = lasso_cv_df[lasso_cv_df['alpha']==lasso_ose_alpha].index[0]
    ax1.scatter(ose_idx, lasso_cv_df.loc[ose_idx,'mean'],
                color='#52B788', s=150, marker='*', zorder=6,
                label=f'1-SE selected α={lasso_ose_alpha}')
    ax2.bar(range(len(lasso_cv_df)), lasso_cv_df['n_kept'],
            alpha=0.25, color='#6B4F3A', label='Features kept')
    ax1.set_xticks(range(len(lasso_cv_df)))
    ax1.set_xticklabels([str(a) for a in lasso_cv_df['alpha']])
    ax1.set_xlabel('Lasso Alpha (λ)')
    ax1.set_ylabel('CV RMSE (kg/ha)', color='#5FA8D3')
    ax2.set_ylabel('Features Retained', color='#6B4F3A')
    ax1.set_title('Lasso: CV RMSE vs Alpha + Feature Selection\nOne-Standard-Error Rule Applied',
                  fontsize=13, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=9)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/18_lasso_cv_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 19: Logistic CV — AUC vs C ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    lr_threshold = lr_cv_df.loc[lr_cv_df['mean'].idxmax(),'mean'] - \
                   lr_cv_df.loc[lr_cv_df['mean'].idxmax(),'se']
    ax.errorbar(range(len(lr_cv_df)),
                lr_cv_df['mean'],
                yerr=lr_cv_df['se'],
                fmt='o-', color='#E07B39', linewidth=2,
                capsize=5, label='Mean CV AUC-ROC ± 1 SE')
    ax.axhline(lr_threshold, color='#2D6A4F', linewidth=1.5,
               linestyle='--', label=f'1-SE threshold')
    best_idx = lr_cv_df['mean'].idxmax()
    ax.scatter(best_idx, lr_cv_df.loc[best_idx,'mean'],
               color='red', s=120, zorder=5, label='Best CV C')
    ose_idx = lr_cv_df[lr_cv_df['C']==lr_ose_C].index[0]
    ax.scatter(ose_idx, lr_cv_df.loc[ose_idx,'mean'],
               color='#52B788', s=150, marker='*', zorder=6,
               label=f'1-SE selected C={lr_ose_C}')
    ax.set_xticks(range(len(lr_cv_df)))
    ax.set_xticklabels([str(c) for c in lr_cv_df['C']])
    ax.set_xlabel('C (Inverse regularization strength)')
    ax.set_ylabel('CV AUC-ROC')
    ax.set_title('Logistic Regression: Stratified 5-Fold CV AUC vs C\nOne-Standard-Error Rule Applied',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/19_logistic_cv_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # ── Plot 20: Milestone summary — all models W3→W5 ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("CropCast — W3→W5 Baseline Milestone\nHonest Performance Summary",
                 fontsize=14, fontweight='bold')

    # Left: Regression RMSE progression
    ax = axes[0]
    models = ['W3 OLS\n(no feat eng)', 'W4 Ridge\n(feat eng)', 'W5 Ridge\n(1-SE CV)']
    rmses  = [W3_OLS_RMSE, W4_RIDGE_RMSE, ridge_rmse]
    colors = ['#E07B39', '#5FA8D3', '#2D6A4F']
    bars   = ax.bar(models, rmses, color=colors, edgecolor='white', width=0.55)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                f"{val:.0f}", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE (kg/ha) — lower is better')
    ax.set_title('Yield Regression — RMSE Across Weeks')
    ax.set_ylim(0, max(rmses)*1.2)

    # Right: Classification AUC
    ax = axes[1]
    cls_models = ['W4 Logistic\n(initial)', 'W5 Logistic\n(1-SE CV)']
    aucs       = [W4_AUC_ROC, lr_auc]
    colors_cls = ['#5FA8D3', '#2D6A4F']
    bars2      = ax.bar(cls_models, aucs, color=colors_cls,
                        edgecolor='white', width=0.45)
    for bar, val in zip(bars2, aucs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.4f}", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC-ROC — higher is better')
    ax.set_title('Disease Classification — AUC-ROC')
    ax.set_ylim(0.85, 1.0)
    ax.axhline(0.9, color='#E07B39', linewidth=1.2, linestyle='--',
               alpha=0.6, label='AUC=0.9 reference')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = f"{FIGURES_PATH}/20_milestone_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

# ── STEP 8: FINAL SUMMARY ─────────────────────────────────────────────────────
def print_summary(ridge_ose_alpha, ridge_rmse, ridge_r2,
                  lasso_ose_alpha, lasso_rmse, lasso_r2,
                  lasso_zeros, lasso_kept,
                  lr_ose_C, lr_precision, lr_recall, lr_f1, lr_auc):
    print("\n" + "█"*60)
    print("  WEEK 5 — HONEST BASELINE MILESTONE")
    print("  ✅ SEALED — these numbers go forward as the bar to beat")
    print("█"*60)

    print(f"\n  REGRESSION — Expected Yield (kg/ha)")
    print(f"  {'Model':<40} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*58}")
    print(f"  {'W3 OLS (baseline, no feature engineering)':<40} {W3_OLS_RMSE:>8.2f}  {'0.2779':>8}")
    print(f"  {'W5 Ridge (α='+str(ridge_ose_alpha)+', 1-SE, 5-fold CV)':<40} {ridge_rmse:>8.2f}  {ridge_r2:>8.4f}")
    print(f"  {'W5 Lasso (α='+str(lasso_ose_alpha)+', 1-SE, 5-fold CV)':<40} {lasso_rmse:>8.2f}  {lasso_r2:>8.4f}")
    print(f"\n  Total RMSE improvement W3→W5 : {W3_OLS_RMSE - min(ridge_rmse,lasso_rmse):+.2f} kg/ha")
    print(f"  Lasso feature selection       : {lasso_zeros} eliminated, {lasso_kept} retained")

    print(f"\n  CLASSIFICATION — Disease Outbreak")
    print(f"  {'Metric':<20} {'W4 Initial':>12} {'W5 1-SE CV':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Precision':<20} {'0.6283':>12} {lr_precision:>12.4f}")
    print(f"  {'Recall':<20} {'0.8209':>12} {lr_recall:>12.4f}")
    print(f"  {'F1 Score':<20} {'0.7118':>12} {lr_f1:>12.4f}")
    print(f"  {'AUC-ROC':<20} {W4_AUC_ROC:>12.4f} {lr_auc:>12.4f}")

    print(f"\n  4 new plots saved to: {FIGURES_PATH}")
    print(f"     17_ridge_cv_curve.png")
    print(f"     18_lasso_cv_curve.png")
    print(f"     19_logistic_cv_curve.png")
    print(f"     20_milestone_summary.png")

    print(f"\n  ⚠️  MILESTONE LOCKED — test set now SEALED")
    print(f"  W6 Decision Tree, W7 Random Forest, W8 XGBoost")
    print(f"  must beat: RMSE < {min(ridge_rmse,lasso_rmse):.0f} kg/ha  AUC > {lr_auc:.4f}")
    print("█"*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 5")
    print("  Cross-Validation + 1-SE Rule + Honest Baseline Milestone")
    print("█"*60)

    (X_train, X_test,
     y_yield_train, y_yield_test,
     y_dis_train, y_dis_test,
     feature_cols)              = load_and_engineer()

    ridge_cv_df, ridge_ose_alpha, \
        ridge_best_mean, ridge_best_se, \
        ridge_threshold         = cv_ridge(X_train, y_yield_train)

    lasso_cv_df, lasso_ose_alpha, \
        lasso_best_mean, lasso_best_se, \
        lasso_threshold         = cv_lasso(X_train, y_yield_train)

    lr_cv_df, lr_ose_C, \
        lr_best_mean, lr_best_se = cv_logistic(X_train, y_dis_train)

    (ridge, ridge_rmse, ridge_r2,
     lasso, lasso_rmse, lasso_r2,
     lasso_zeros, lasso_kept,
     lr, lr_precision, lr_recall,
     lr_f1, lr_auc, y_prob)     = evaluate_final_models(
                                        X_train, X_test,
                                        y_yield_train, y_yield_test,
                                        y_dis_train, y_dis_test,
                                        ridge_ose_alpha, lasso_ose_alpha, lr_ose_C)

    log_milestone(ridge_ose_alpha, ridge_rmse, ridge_r2,
                  lasso_ose_alpha, lasso_rmse, lasso_r2, lasso_zeros, lasso_kept,
                  lr_ose_C, lr_precision, lr_recall, lr_f1, lr_auc)

    generate_plots(ridge_cv_df, lasso_cv_df, lr_cv_df,
                   ridge_threshold, lasso_threshold,
                   ridge_ose_alpha, lasso_ose_alpha, lr_ose_C,
                   ridge_rmse, lasso_rmse, lr_auc)

    print_summary(ridge_ose_alpha, ridge_rmse, ridge_r2,
                  lasso_ose_alpha, lasso_rmse, lasso_r2,
                  lasso_zeros, lasso_kept,
                  lr_ose_C, lr_precision, lr_recall, lr_f1, lr_auc)