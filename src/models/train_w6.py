# =============================================================================
# CropCast — Week 6: Decision Tree for Disease Outbreak Risk
# File: src/models/train_w6.py
#
# Requirements:
# 1. Grow an interpretable decision tree on disease_outbreak
# 2. Read top splits on soil, weather, NDVI, sowing dates
# 3. Must beat W5 baseline: AUC > 0.9154
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report, RocCurveDisplay)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH       = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/clean_dataset.csv"
FIGURES_PATH          = "/Users/amrkalammansoori/Desktop/crop-cast/reports/figures"
MLFLOW_DIR            = "/Users/amrkalammansoori/Desktop/crop-cast/mlruns"
CLASSIFICATION_TARGET = "disease_outbreak"
REGRESSION_TARGET     = "expected_yield_kg_ha"
RANDOM_STATE          = 42

# W5 baselines to beat
W5_AUC  = 0.9154
W5_RMSE = 955.14

# ── FEATURE ENGINEERING (identical to W4/W5) ──────────────────────────────────
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
    y_disease = fe[CLASSIFICATION_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_disease, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_disease)

    # Decision trees don't need scaling but we keep it for consistency
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    if not np.isfinite(X_train_s).all():
        X_train_s = np.nan_to_num(X_train_s)
        X_test_s  = np.nan_to_num(X_test_s)

    print(f"✅ Train: {X_train_s.shape[0]:,}  Test: {X_test_s.shape[0]:,}  Features: {X_train_s.shape[1]}")
    return X_train_s, X_test_s, y_train, y_test, feature_cols

# ── STEP 2: FIND BEST MAX DEPTH ───────────────────────────────────────────────
def tune_depth(X_train, y_train):
    print("\n" + "="*60)
    print("STEP 2 — TUNE TREE DEPTH (5-FOLD STRATIFIED CV)")
    print("="*60)

    depths   = [3, 4, 5, 6, 7, 8, 10, None]
    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results  = []

    print(f"  {'Max Depth':<12} {'Mean AUC':>10} {'Std AUC':>10}")
    print(f"  {'-'*34}")

    for d in depths:
        model     = DecisionTreeClassifier(
            max_depth=d, class_weight='balanced',
            random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=skf, scoring='roc_auc')
        label = str(d) if d else 'None (full)'
        print(f"  {label:<12} {cv_scores.mean():>10.4f} {cv_scores.std():>10.4f}")
        results.append({'depth':d,'mean':cv_scores.mean(),'std':cv_scores.std()})

    best = max(results, key=lambda x: x['mean'])
    print(f"\n  Best depth: {best['depth']}  (CV AUC = {best['mean']:.4f})")
    return best['depth']

# ── STEP 3: TRAIN FINAL DECISION TREE ─────────────────────────────────────────
def train_tree(X_train, X_test, y_train, y_test, best_depth, feature_cols):
    print("\n" + "="*60)
    print("STEP 3 — TRAIN DECISION TREE & EVALUATE")
    print("="*60)

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Disease_Prediction")

    with mlflow.start_run(run_name="W6_DecisionTree_Classifier"):
        tree = DecisionTreeClassifier(
            max_depth=best_depth,
            class_weight='balanced',
            criterion='gini',
            random_state=RANDOM_STATE
        )
        tree.fit(X_train, y_train)

        y_pred = tree.predict(X_test)
        y_prob = tree.predict_proba(X_test)[:,1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_prob)
        accuracy  = (y_pred == y_test).mean()

        print(f"\n  📊 Decision Tree Metrics:")
        print(f"     Accuracy  = {accuracy:.4f}")
        print(f"     Precision = {precision:.4f}")
        print(f"     Recall    = {recall:.4f}")
        print(f"     F1 Score  = {f1:.4f}")
        print(f"     AUC-ROC   = {auc:.4f}")

        beat = auc > W5_AUC
        print(f"\n  {'✅ BEAT' if beat else '❌ DID NOT BEAT'} W5 Logistic baseline!")
        print(f"  W5 AUC: {W5_AUC:.4f}  →  W6 Tree AUC: {auc:.4f}  ({auc-W5_AUC:+.4f})")

        # Tree size
        n_leaves = tree.get_n_leaves()
        depth    = tree.get_depth()
        print(f"\n  Tree structure:")
        print(f"  Max depth set : {best_depth}")
        print(f"  Actual depth  : {depth}")
        print(f"  Number of leaves: {n_leaves}")

        # Top splits — feature importances
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n  Top 15 features by Gini importance:")
        print(imp_df.head(15).to_string(index=False))

        # Text representation of top of tree
        print(f"\n  Top 3 levels of the decision tree:")
        tree_text = export_text(tree, feature_names=feature_cols, max_depth=3)
        print(tree_text)

        mlflow.log_param("model",      "DecisionTree")
        mlflow.log_param("week",       "W6")
        mlflow.log_param("max_depth",  best_depth)
        mlflow.log_param("criterion",  "gini")
        mlflow.log_param("n_leaves",   n_leaves)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("auc_roc",   auc)
        mlflow.log_metric("w5_auc",    W5_AUC)
        mlflow.log_metric("auc_improvement", auc - W5_AUC)
        mlflow.sklearn.log_model(tree, "decision_tree_model")
        print(f"\n  ✅ MLflow run logged → W6_DecisionTree_Classifier")

    return tree, y_prob, auc, precision, recall, f1, imp_df

# ── STEP 4: VISUALISATIONS ────────────────────────────────────────────────────
def generate_plots(tree, feature_cols, y_test, y_prob, imp_df, best_depth):
    print("\n" + "="*60)
    print("STEP 4 — GENERATING VISUALISATIONS")
    print("="*60)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Plot 21: Tree visualisation (top 4 levels)
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(tree, feature_names=feature_cols,
              class_names=['No Outbreak','Outbreak'],
              filled=True, rounded=True, max_depth=4,
              ax=ax, fontsize=8)
    ax.set_title(f"Decision Tree — Disease Outbreak Risk\n"
                 f"CropCast W6 (max_depth={best_depth}, showing top 4 levels)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f"{FIGURES_PATH}/21_decision_tree.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Plot 22: Feature importance bar chart
    top20 = imp_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = ['#2D6A4F' if v > 0.05 else '#5FA8D3' for v in top20['Importance']]
    ax.barh(top20['Feature'], top20['Importance'], color=colors, edgecolor='white')
    ax.set_xlabel('Gini Importance')
    ax.set_title('Decision Tree — Feature Importance (Gini)\nCropCast W6',
                 fontsize=13, fontweight='bold')
    ax.axvline(0.05, color='#E07B39', linewidth=1.5,
               linestyle='--', label='0.05 threshold')
    ax.legend()
    plt.tight_layout()
    path = f"{FIGURES_PATH}/22_dt_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

    # Plot 23: ROC Curve
    fig, ax = plt.subplots(figsize=(8, 7))
    RocCurveDisplay.from_predictions(
        y_test, y_prob, ax=ax,
        color='#2D6A4F', name='Decision Tree (W6)')
    ax.axhline(W5_AUC, color='#E07B39', linewidth=1.5,
               linestyle='--', label=f'W5 Logistic baseline AUC={W5_AUC}')
    ax.plot([0,1],[0,1],'k--', linewidth=1)
    ax.set_title('ROC Curve — Decision Tree vs W5 Baseline\nCropCast W6',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/23_dt_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {path}")

# ── STEP 5: SUMMARY ───────────────────────────────────────────────────────────
def print_summary(auc, precision, recall, f1, best_depth, imp_df):
    print("\n" + "█"*60)
    print("  WEEK 6 RESULTS SUMMARY")
    print("█"*60)
    print(f"\n  CLASSIFICATION — Disease Outbreak (Decision Tree)")
    print(f"  {'Metric':<20} {'W5 Logistic':>12} {'W6 Tree':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Precision':<20} {'0.6278':>12} {precision:>12.4f}")
    print(f"  {'Recall':<20} {'0.8212':>12} {recall:>12.4f}")
    print(f"  {'F1 Score':<20} {'0.7116':>12} {f1:>12.4f}")
    print(f"  {'AUC-ROC':<20} {W5_AUC:>12.4f} {auc:>12.4f}")
    print(f"\n  Tree depth: {best_depth}  |  Top split feature: {imp_df.iloc[0]['Feature']}")
    print(f"\n  Top 5 most important features:")
    for _, row in imp_df.head(5).iterrows():
        print(f"    {row['Feature']:<30} {row['Importance']:.4f}")
    print(f"\n  3 plots saved to: {FIGURES_PATH}")
    print(f"     21_decision_tree.png")
    print(f"     22_dt_feature_importance.png")
    print(f"     23_dt_roc_curve.png")
    print(f"\n  Next → Week 7: Random Forest — cut variance, OOB error,")
    print(f"                  permutation importance ranking")
    print("█"*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 6: DECISION TREE")
    print("  Interpretable classifier for disease-outbreak risk")
    print("█"*60)

    X_train, X_test, y_train, y_test, feature_cols = load_and_engineer()
    best_depth = tune_depth(X_train, y_train)
    tree, y_prob, auc, precision, recall, f1, imp_df = train_tree(
        X_train, X_test, y_train, y_test, best_depth, feature_cols)
    generate_plots(tree, feature_cols, y_test, y_prob, imp_df, best_depth)
    print_summary(auc, precision, recall, f1, best_depth, imp_df)