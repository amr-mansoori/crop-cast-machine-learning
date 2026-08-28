# =============================================================================
# CropCast — Week 2: Data Ingestion, Validation & EDA
# File: src/data/ingest.py
# Dataset: cropcast_dataset.csv (250,000 rows, 24 columns)
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/cropcast_dataset.csv"
FIGURES_PATH = "/Users/amrkalammansoori/Desktop/crop-cast/reports/figures"
CLEAN_PATH  = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/clean_dataset.csv"

REGRESSION_TARGET    = "expected_yield_kg_ha"
CLASSIFICATION_TARGET = "disease_outbreak"

NUMERICAL_FEATURES = [
    "sowing_delay_days", "soil_nitrogen", "soil_phosphorus", "soil_potassium",
    "soil_ph", "soil_moisture", "organic_matter", "temperature", "humidity",
    "rainfall", "wind_speed", "growing_degree_days", "ndvi", "savi",
    "ndvi_anomaly", "chlorophyll_content", "leaf_area_index"
]

CATEGORICAL_FEATURES = ["crop_type", "agro_zone", "region", "season"]

# ── STEP 1: LOAD & VALIDATE ───────────────────────────────────────────────────
def load_and_validate(path):
    print("\n" + "="*60)
    print("STEP 1 — LOADING & VALIDATING DATA")
    print("="*60)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Verify targets
    for col in [REGRESSION_TARGET, CLASSIFICATION_TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing target column: {col}")
    print(f"✅ Regression target    : '{REGRESSION_TARGET}'")
    print(f"✅ Classification target: '{CLASSIFICATION_TARGET}'")

    return df

# ── STEP 2: SCHEMA PROFILE ────────────────────────────────────────────────────
def schema_profile(df):
    print("\n" + "="*60)
    print("STEP 2 — SCHEMA PROFILE")
    print("="*60)

    print(f"\n  Numerical features   ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    print(f"\n  Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    print(f"\n  Targets: {REGRESSION_TARGET}, {CLASSIFICATION_TARGET}")

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Numerical Summary Statistics ---")
    print(df[NUMERICAL_FEATURES].describe().round(3))

# ── STEP 3: DATA QUALITY ──────────────────────────────────────────────────────
def check_data_quality(df):
    print("\n" + "="*60)
    print("STEP 3 — DATA QUALITY CHECK")
    print("="*60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).query("`Missing Count` > 0")

    if missing_report.empty:
        print("✅ No missing values found.")
    else:
        print(f"⚠️  Missing values:\n{missing_report}")

    dupes = df.duplicated().sum()
    if dupes == 0:
        print("✅ No duplicate rows found.")
    else:
        print(f"⚠️  {dupes:,} duplicates found — dropping.")
        df = df.drop_duplicates()

    return df

# ── STEP 4: TARGET ANALYSIS ───────────────────────────────────────────────────
def analyse_targets(df):
    print("\n" + "="*60)
    print("STEP 4 — TARGET ANALYSIS")
    print("="*60)

    # Regression target
    print(f"\n📊 REGRESSION TARGET: {REGRESSION_TARGET}")
    print(df[REGRESSION_TARGET].describe().round(2))

    naive_mean = df[REGRESSION_TARGET].mean()
    naive_rmse = np.sqrt(((df[REGRESSION_TARGET] - naive_mean)**2).mean())
    print(f"\n  Naïve baseline (predict mean always):")
    print(f"  Mean yield  : {naive_mean:.2f} kg/ha")
    print(f"  Naïve RMSE  : {naive_rmse:.2f} kg/ha  ← irreducible-error floor estimate")
    print(f"  Any model must beat RMSE = {naive_rmse:.2f} to be useful.")

    # Classification target
    print(f"\n🦠 CLASSIFICATION TARGET: {CLASSIFICATION_TARGET}")
    counts = df[CLASSIFICATION_TARGET].value_counts()
    pcts   = df[CLASSIFICATION_TARGET].value_counts(normalize=True) * 100
    print(pd.DataFrame({"Count": counts, "Percentage %": pcts.round(2)}))
    majority_pct = pcts.max()
    print(f"\n  Majority-class baseline accuracy: {majority_pct:.1f}%")
    print(f"  Minority class (outbreak=1)     : {pcts.min():.1f}%")
    if pcts.min() < 30:
        print(f"  ⚠️  Class imbalance — use class_weight='balanced' in W4")

    # Correlations — the key check
    print(f"\n🔗 FEATURE CORRELATIONS WITH TARGETS:")
    num_df = df[NUMERICAL_FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]]
    corr   = num_df.corr()

    print(f"\n  Top 8 correlations with {REGRESSION_TARGET}:")
    yc = (corr[REGRESSION_TARGET].drop(REGRESSION_TARGET)
          .sort_values(key=abs, ascending=False).head(8))
    print(yc.round(4).to_string())

    print(f"\n  Top 8 correlations with {CLASSIFICATION_TARGET}:")
    dc = (corr[CLASSIFICATION_TARGET].drop(CLASSIFICATION_TARGET)
          .sort_values(key=abs, ascending=False).head(8))
    print(dc.round(4).to_string())

# ── STEP 5: VISUALISATIONS ────────────────────────────────────────────────────
def generate_visualisations(df):
    print("\n" + "="*60)
    print("STEP 5 — GENERATING VISUALISATIONS")
    print("="*60)

    os.makedirs(FIGURES_PATH, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1: Yield distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Regression Target — Yield Distribution", fontsize=14, fontweight="bold")
    axes[0].hist(df[REGRESSION_TARGET], bins=60, color="#2D6A4F", edgecolor="white", alpha=0.85)
    axes[0].axvline(df[REGRESSION_TARGET].mean(), color="#E07B39", linewidth=2,
                    label=f"Mean={df[REGRESSION_TARGET].mean():.0f}")
    axes[0].set_xlabel("Yield (kg/ha)"); axes[0].set_ylabel("Frequency")
    axes[0].set_title("Yield Histogram"); axes[0].legend()
    sns.boxplot(y=df[REGRESSION_TARGET], ax=axes[1], color="#52B788")
    axes[1].set_ylabel("Yield (kg/ha)"); axes[1].set_title("Yield Boxplot")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/01_yield_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 01_yield_distribution.png")

    # Plot 2: Class balance
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[CLASSIFICATION_TARGET].value_counts()
    colors = ["#2D6A4F", "#E07B39"]
    bars = ax.bar(["No Outbreak (0)", "Outbreak (1)"], counts.values,
                  color=colors, edgecolor="white")
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{count:,}\n({count/len(df)*100:.1f}%)",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Classification Target — Class Balance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count"); ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/02_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 02_class_balance.png")

    # Plot 3: Correlation heatmap
    num_df = df[NUMERICAL_FEATURES + [REGRESSION_TARGET, CLASSIFICATION_TARGET]]
    corr   = num_df.corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 03_correlation_heatmap.png")

    # Plot 4: Top correlations with yield
    yield_corr = (corr[REGRESSION_TARGET]
                  .drop([REGRESSION_TARGET, CLASSIFICATION_TARGET])
                  .sort_values(key=abs, ascending=False).head(10))
    colors_corr = ["#2D6A4F" if v > 0 else "#E07B39" for v in yield_corr.values]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(yield_corr.index, yield_corr.values, color=colors_corr, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson Correlation with Yield")
    ax.set_title("Top 10 Features Correlated with Yield", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/04_yield_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 04_yield_correlations.png")

    # Plot 5: NDVI vs Yield coloured by disease
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(df["ndvi"], df[REGRESSION_TARGET],
                         c=df[CLASSIFICATION_TARGET], cmap="RdYlGn",
                         alpha=0.2, s=4, rasterized=True)
    plt.colorbar(scatter, ax=ax, label="Disease Outbreak (0=No, 1=Yes)")
    ax.set_xlabel("NDVI"); ax.set_ylabel("Yield (kg/ha)")
    ax.set_title("NDVI vs Yield — coloured by Disease Outbreak",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/05_ndvi_vs_yield.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 05_ndvi_vs_yield.png")

    # Plot 6: Yield by crop type
    fig, ax = plt.subplots(figsize=(10, 6))
    order = (df.groupby("crop_type")[REGRESSION_TARGET]
               .median().sort_values(ascending=False).index)
    sns.boxplot(data=df, x="crop_type", y=REGRESSION_TARGET,
                order=order, palette="Set2", ax=ax)
    ax.set_title("Yield Distribution by Crop Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Crop Type"); ax.set_ylabel("Yield (kg/ha)")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/06_yield_by_crop_type.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 06_yield_by_crop_type.png")

    # Plot 7: Key feature distributions
    key_feats = ["temperature", "humidity", "rainfall", "soil_moisture",
                 "soil_ph", "organic_matter", "ndvi", "savi"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Key Feature Distributions", fontsize=14, fontweight="bold")
    for i, feat in enumerate(key_feats):
        axes.flatten()[i].hist(df[feat], bins=40, color="#5FA8D3",
                               edgecolor="white", alpha=0.85)
        axes.flatten()[i].set_title(feat, fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/07_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 07_feature_distributions.png")

    # Plot 8: Outbreak rate by agro zone
    fig, ax = plt.subplots(figsize=(10, 5))
    zone_outbreak = (df.groupby("agro_zone")[CLASSIFICATION_TARGET].mean() * 100
                     ).sort_values(ascending=False)
    ax.bar(zone_outbreak.index, zone_outbreak.values,
           color="#E07B39", edgecolor="white")
    ax.set_ylabel("Outbreak Rate (%)"); ax.set_xlabel("Agro-Climatic Zone")
    ax.set_title("Disease Outbreak Rate by Agro-Climatic Zone",
                 fontsize=13, fontweight="bold")
    for i, v in enumerate(zone_outbreak.values):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_PATH}/08_outbreak_by_zone.png", dpi=150, bbox_inches="tight")
    plt.close(); print("✅ Saved: 08_outbreak_by_zone.png")

# ── STEP 6: SAVE CLEAN DATASET ────────────────────────────────────────────────
def save_clean_dataset(df):
    print("\n" + "="*60)
    print("STEP 6 — SAVING CLEAN DATASET")
    print("="*60)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Saved to: {CLEAN_PATH}")
    print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Ready for Week 3 model training.")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  CROPCAST — WEEK 2: INGESTION, VALIDATION & EDA")
    print("█"*60)

    df = load_and_validate(DATA_PATH)
    schema_profile(df)
    df = check_data_quality(df)
    analyse_targets(df)
    generate_visualisations(df)
    save_clean_dataset(df)

    print("\n" + "█"*60)
    print("  WEEK 2 COMPLETE ✅")
    print(f"  8 plots saved to: {FIGURES_PATH}")
    print(f"  Clean dataset   : src/data/clean_dataset.csv")
    print(f"  Next → Week 3: Linear Regression on expected_yield_kg_ha")
    print("█"*60 + "\n")