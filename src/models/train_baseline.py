# =============================================================================
# CropCast — Week 3: Linear Regression Baseline + Gradient Descent + MLflow
# File: src/models/train_baseline.py
# Dataset: cropcast_dataset.csv (250,000 rows — real agronomic correlations)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/clean_dataset.csv"
FIGURES_PATH    = "/Users/amrkalammansoori/Desktop/crop-cast/reports/figures"
MLFLOW_DIR      = "/Users/amrkalammansoori/Desktop/crop-cast/mlruns"
REGRESSION_TARGET = "expected_yield_kg_ha"

FEATURE_COLS = [
    "sowing_delay_days", "soil_nitrogen", "soil_phosphorus", "soil_potassium",
    "soil_ph", "soil_moisture", "organic_matter", "temperature", "humidity",
    "rainfall", "wind_speed", "growing_degree_days", "ndvi", "savi",
    "ndvi_anomaly", "chlorophyll_content", "leaf_area_index"
]

RANDOM_STATE = 42

# ── HELPER ────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n  📊 {label} Metrics:")
    print(f"     MAE  = {mae:.2f} kg/ha")
    print(f"     RMSE = {rmse:.2f} kg/ha")
    print(f"     R²   = {r2:.4f}")
    return mae, rmse, r2

# ── STEP 1: LOAD ──────────────────────────────────────────────────────────────
def load_data():
    print("\n" + "="*60)
    print("STEP 1 — LOADING CLEAN DATASET")
    print("="*60)

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"Run src/data/ingest.py first.\nNot found: {CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    features = [f for f in FEATURE_COLS if f in df.columns]
    missing  = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"⚠️  Skipped (not found): {missing}")

    print(f"✅ Using {len(features)} features")
    return df, features

# ── STEP 2: PREPARE ───────────────────────────────────────────────────────────
def prepare_data(df, features):
    print("\n" + "="*60)
    print("STEP 2 — TRAIN / TEST SPLIT & SCALING")
    print("="*60)

    X = df[features].values
    y = df[REGRESSION_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)

    print(f"✅ Train: {X_train.shape[0]:,} rows  |  Test: {X_test.shape[0]:,} rows")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    print("✅ StandardScaler fitted on train only (no leakage)")

    naive_rmse = np.sqrt(mean_squared_error(
        y_test, np.full_like(y_test, y_train.mean())))
    print(f"\n  Naïve baseline RMSE : {naive_rmse:.2f} kg/ha")
    print(f"  Mean yield          : {y_train.mean():.2f} kg/ha")
    print(f"  Any model must beat : RMSE = {naive_rmse:.2f}")

    return X_train_s, X_test_s, y_train, y_test, scaler, naive_rmse

# ── STEP 3: OLS ───────────────────────────────────────────────────────────────
def run_ols(X_train, X_test, y_train, y_test, naive_rmse, features):
    print("\n" + "="*60)
    print("STEP 3 — OLS LINEAR REGRESSION (Closed-Form Normal Equation)")
    print("="*60)
    print("  Formula: w = (X'X)^-1 X'y")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Yield_Prediction")

    with mlflow.start_run(run_name="W3_OLS_Baseline"):
        ols = LinearRegression()
        ols.fit(X_train, y_train)

        y_pred_train = ols.predict(X_train)
        y_pred_test  = ols.predict(X_test)

        train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_pred_train, "TRAIN")
        test_mae,  test_rmse,  test_r2  = compute_metrics(y_test,  y_pred_test,  "TEST ")

        improvement = naive_rmse - test_rmse
        print(f"\n  {'✅ BEAT' if improvement > 0 else '❌ DID NOT BEAT'} naïve baseline!")
        print(f"  Naïve RMSE : {naive_rmse:.2f}  →  OLS RMSE : {test_rmse:.2f}  "
              f"({improvement:+.2f} kg/ha)")

        # Feature coefficients
        coef_df = pd.DataFrame({"Feature": features, "Coefficient": ols.coef_})
        coef_df = coef_df.reindex(
            coef_df["Coefficient"].abs().sort_values(ascending=False).index)
        print("\n  Feature coefficients (sorted by |magnitude|):")
        print(coef_df.to_string(index=False))

        mlflow.log_param("model",       "OLS_LinearRegression")
        mlflow.log_param("week",        "W3")
        mlflow.log_param("n_features",  X_train.shape[1])
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2",   train_r2)
        mlflow.log_metric("test_rmse",  test_rmse)
        mlflow.log_metric("test_r2",    test_r2)
        mlflow.log_metric("naive_rmse", naive_rmse)
        mlflow.log_metric("improvement_over_naive", improvement)
        mlflow.sklearn.log_model(ols, "ols_model")
        print(f"\n  ✅ MLflow run logged → W3_OLS_Baseline")

    return ols, y_pred_test, test_rmse, test_r2

# ── STEP 4: GRADIENT DESCENT ─────────────────────────────────────────────────
def gradient_descent_loop(X, y, alpha, n_iters):
    m = X.shape[0]
    w = np.zeros(X.shape[1])
    costs = []
    for _ in range(n_iters):
        error    = X @ w - y
        gradient = (1 / m) * (X.T @ error)
        w        = w - alpha * gradient
        cost     = (1 / (2 * m)) * np.sum(error ** 2)
        costs.append(cost)
        if not np.isfinite(cost):
            break
    return w, costs

def run_gradient_descent(X_train, X_test, y_train, y_test, naive_rmse):
    print("\n" + "="*60)
    print("STEP 4 — GRADIENT DESCENT (Pure NumPy, From Scratch)")
    print("="*60)
    print("  Update rule: w = w - α × (1/m) × Xᵀ(Xw − y)")

    X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_b  = np.hstack([np.ones((X_test.shape[0],  1)), X_test])

    y_mean = y_train.mean()
    y_std  = y_train.std()
    y_norm = (y_train - y_mean) / y_std

    learning_rates = [0.001, 0.01, 0.05, 0.1]
    n_iters        = 2000
    results        = {}

    os.makedirs(FIGURES_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha in learning_rates:
        w, costs = gradient_descent_loop(X_train_b, y_norm, alpha, n_iters)

        if not np.isfinite(costs[-1]):
            print(f"  α={alpha}  →  DIVERGED")
            continue

        y_pred = (X_test_b @ w) * y_std + y_mean
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
        r2     = r2_score(y_test, y_pred)
        results[alpha] = {"costs": costs, "rmse": rmse, "r2": r2}

        ax.plot(costs, label=f"α={alpha}  RMSE={rmse:.0f} kg/ha")
        print(f"  α={alpha:<5}  →  RMSE={rmse:.2f}  R²={r2:.4f}  "
              f"Final cost={costs[-1]:.6f}")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost MSE/2 (normalised y)")
    ax.set_title("Gradient Descent — Learning Rate Comparison\nCropCast W3",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    gd_plot = f"{FIGURES_PATH}/09_gradient_descent_curves.png"
    plt.savefig(gd_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✅ Saved: {gd_plot}")

    if not results:
        return 0.001, naive_rmse, 0.0

    best_alpha = min(results, key=lambda a: results[a]["rmse"])
    best       = results[best_alpha]
    print(f"\n  🏆 Best learning rate: α = {best_alpha}")
    print(f"     RMSE = {best['rmse']:.2f} kg/ha  |  R² = {best['r2']:.4f}")

    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment("CropCast_Yield_Prediction")
    with mlflow.start_run(run_name="W3_GradientDescent_Baseline"):
        mlflow.log_param("model",       "GradientDescent_NumPy")
        mlflow.log_param("week",        "W3")
        mlflow.log_param("best_alpha",  best_alpha)
        mlflow.log_param("n_iters",     n_iters)
        mlflow.log_metric("test_rmse",  best["rmse"])
        mlflow.log_metric("test_r2",    best["r2"])
        mlflow.log_metric("naive_rmse", naive_rmse)
        mlflow.log_metric("improvement", naive_rmse - best["rmse"])
        mlflow.log_artifact(gd_plot)
        print(f"  ✅ MLflow run logged → W3_GradientDescent_Baseline")

    return best_alpha, best["rmse"], best["r2"]

# ── STEP 5: DIAGNOSTIC PLOTS ─────────────────────────────────────────────────
def plot_diagnostics(y_test, y_pred_ols, ols_rmse, naive_rmse):
    print("\n" + "="*60)
    print("STEP 5 — DIAGNOSTIC PLOTS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("OLS Baseline — Diagnostic Plots  |  CropCast W3",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.scatter(y_test, y_pred_ols, alpha=0.15, s=5,
               color="#2D6A4F", rasterized=True)
    lims = [min(y_test.min(), y_pred_ols.min()),
            max(y_test.max(), y_pred_ols.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Yield (kg/ha)"); ax.set_ylabel("Predicted Yield (kg/ha)")
    ax.set_title(f"Actual vs Predicted\nRMSE = {ols_rmse:.2f} kg/ha")
    ax.legend()

    residuals = y_test - y_pred_ols
    ax = axes[1]
    ax.scatter(y_pred_ols, residuals, alpha=0.15, s=5,
               color="#5FA8D3", rasterized=True)
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Predicted Yield (kg/ha)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residual Plot\nMean residual = {residuals.mean():.2f}")

    plt.tight_layout()
    path = f"{FIGURES_PATH}/10_ols_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"✅ Saved: {path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Naïve Baseline\n(predict mean)", "OLS Linear\nRegression (W3)"],
           [naive_rmse, ols_rmse],
           color=["#E07B39", "#2D6A4F"], edgecolor="white", width=0.5)
    for i, val in enumerate([naive_rmse, ols_rmse]):
        ax.text(i, val + 5, f"{val:.2f}", ha="center",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("RMSE (kg/ha) — lower is better")
    ax.set_title("Baseline Comparison\nCropCast W3", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(naive_rmse, ols_rmse) * 1.2)
    plt.tight_layout()
    path = f"{FIGURES_PATH}/11_baseline_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(); print(f"✅ Saved: {path}")

# ── STEP 6: SUMMARY ───────────────────────────────────────────────────────────
def print_summary(naive_rmse, ols_rmse, ols_r2, gd_alpha, gd_rmse, gd_r2):
    print("\n" + "█"*60)
    print("  WEEK 3 RESULTS SUMMARY")
    print("█"*60)
    print(f"\n  {'Model':<38} {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*56}")
    print(f"  {'Naïve Baseline (predict mean)':<38} {naive_rmse:>8.2f}       —")
    print(f"  {'OLS Linear Regression':<38} {ols_rmse:>8.2f}  {ols_r2:>8.4f}")
    print(f"  {'GD Best (α='+str(gd_alpha)+')':<38} {gd_rmse:>8.2f}  {gd_r2:>8.4f}")
    print(f"\n  OLS improvement over naïve: {naive_rmse - ols_rmse:+.2f} kg/ha")
    print(f"\n  View MLflow UI:")
    print(f"  mlflow ui --backend-store-uri file://{MLFLOW_DIR}")
    print(f"  → http://127.0.0.1:5000")
    print(f"\n  Next → Week 4: Ridge + Lasso on yield,")
    print(f"                  Logistic Regression for disease-outbreak risk")
    print("█"*60 + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 3: LINEAR REGRESSION BASELINE")
    print("  OLS + Gradient Descent + MLflow")
    print("█"*60)

    df, features                        = load_data()
    X_train, X_test, y_train, y_test, \
        scaler, naive_rmse              = prepare_data(df, features)
    ols, y_pred_ols, ols_rmse, ols_r2  = run_ols(X_train, X_test,
                                                  y_train, y_test,
                                                  naive_rmse, features)
    gd_alpha, gd_rmse, gd_r2           = run_gradient_descent(X_train, X_test,
                                                               y_train, y_test,
                                                               naive_rmse)
    plot_diagnostics(y_test, y_pred_ols, ols_rmse, naive_rmse)
    print_summary(naive_rmse, ols_rmse, ols_r2, gd_alpha, gd_rmse, gd_r2)