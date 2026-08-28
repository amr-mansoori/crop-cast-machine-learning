import numpy as np
import pandas as pd
import os

OUTPUT_PATH = "/Users/amrkalammansoori/Desktop/crop-cast/src/data/cropcast_dataset.csv"
N_ROWS      = 250_000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("\n" + "█"*60)
print("  CROPCAST — DATASET GENERATOR")
print("  Generating 250,000 rows with real agronomic relationships")
print("█"*60)

# ── CROP TYPES ────────────────────────────────────────────────────────────────
crop_types    = ["Wheat", "Rice", "Maize", "Soybean", "Barley"]
crop_baseline = {"Wheat": 3500, "Rice": 4500, "Maize": 5500,
                 "Soybean": 2800, "Barley": 3200}
crop_noise    = {"Wheat": 400,  "Rice": 450,  "Maize": 550,
                 "Soybean": 300, "Barley": 350}
crop          = np.random.choice(crop_types, N_ROWS,
                                  p=[0.25, 0.25, 0.25, 0.15, 0.10])

zones   = ["Tropical", "Subtropical", "Temperate", "Semi-Arid", "Humid"]
regions = ["North", "South", "East", "West", "Central"]
seasons = ["Kharif", "Rabi", "Zaid"]

agro_zone = np.random.choice(zones, N_ROWS, p=[0.20, 0.25, 0.25, 0.15, 0.15])
region    = np.random.choice(regions, N_ROWS)
season    = np.random.choice(seasons, N_ROWS, p=[0.45, 0.40, 0.15])

print("[1/5] Base features done")

# ── SOIL FEATURES ─────────────────────────────────────────────────────────────
soil_nitrogen   = np.random.normal(80, 28, N_ROWS).clip(10, 145)
soil_phosphorus = np.random.normal(45, 18, N_ROWS).clip(5, 85)
soil_potassium  = np.random.normal(55, 20, N_ROWS).clip(10, 105)
soil_ph         = np.random.normal(6.5, 0.85, N_ROWS).clip(4.2, 8.8)
soil_moisture   = np.random.normal(45, 15, N_ROWS).clip(15, 85)
organic_matter  = np.random.normal(2.5, 1.1, N_ROWS).clip(0.3, 6.0)
sowing_delay    = np.random.gamma(shape=2, scale=7, size=N_ROWS).clip(0, 45)
crop_growth_stage = np.random.choice([1,2,3,4], N_ROWS, p=[0.20,0.35,0.30,0.15])

print("[2/5] Soil features done")

# ── WEATHER FEATURES ──────────────────────────────────────────────────────────
temperature = np.random.normal(24, 7, N_ROWS).clip(4, 46)
humidity    = np.random.normal(62, 19, N_ROWS).clip(25, 98)
rainfall    = np.random.gamma(shape=3, scale=42, size=N_ROWS).clip(0, 380)
wind_speed  = np.random.exponential(scale=3, size=N_ROWS).clip(0, 16)
gdd         = np.random.normal(1800, 320, N_ROWS).clip(700, 3200)

print("[3/5] Weather features done")

# ── NDVI — genuinely correlated with soil and weather ────────────────────────
ndvi_base = (
    0.25 * (soil_nitrogen / 145) +
    0.20 * (soil_phosphorus / 85) +
    0.15 * (soil_potassium / 105) +
    0.20 * np.clip((rainfall - 30) / 280, 0, 1) +
    0.10 * np.clip((temperature - 8) / 22, 0, 1) +
    0.10 * (organic_matter / 6.0)
)
ndvi             = (ndvi_base * 0.75 + 0.12 +
                   np.random.normal(0, 0.07, N_ROWS)).clip(-0.1, 0.98)
savi             = (ndvi * 0.88 + np.random.normal(0, 0.025, N_ROWS)).clip(-0.1, 0.92)
ndvi_baseline    = (ndvi + np.random.normal(0.06, 0.06, N_ROWS)).clip(0.05, 1.0)
ndvi_anomaly     = (ndvi - ndvi_baseline).clip(-0.5, 0.3)
chlorophyll      = (ndvi * 50 + np.random.normal(12, 7, N_ROWS)).clip(3, 90)
leaf_area_index  = (ndvi * 4.5 + np.random.normal(0.8, 0.7, N_ROWS)).clip(0.3, 9)

print("[4/5] NDVI and satellite features done")

# ── TARGET 1: YIELD — strong agronomic relationships ─────────────────────────
print("[5/5] Computing targets...")

base_yield = np.array([crop_baseline[c] for c in crop], dtype=float)

# Each factor is a multiplier — all grounded in real agronomy
# NPK: primary yield drivers — stronger weight
npk_score  = (0.45*(soil_nitrogen/145) +
              0.30*(soil_phosphorus/85) +
              0.25*(soil_potassium/105))
npk_factor = 0.4 + 1.2 * npk_score                         # range 0.4-1.6

# pH: Gaussian peak at 6.5 (optimal for most crops)
ph_factor  = np.exp(-0.25 * (soil_ph - 6.5)**2)
ph_factor  = 0.55 + 0.55 * ph_factor                       # range 0.55-1.10

# NDVI: direct proxy for crop health and biomass
ndvi_factor = 0.3 + 1.4 * ndvi.clip(0, 1)                  # range 0.3-1.7

# Rainfall: optimal 80-150mm/month
rainfall_factor = np.exp(-0.00006 * (rainfall - 115)**2)
rainfall_factor = 0.45 + 0.70 * rainfall_factor             # range 0.45-1.15

# Temperature: optimal 22-26°C
temp_factor = np.exp(-0.015 * (temperature - 24)**2)
temp_factor = 0.45 + 0.70 * temp_factor                     # range 0.45-1.15

# Sowing delay: each day late = ~1% yield loss
sowing_factor = np.exp(-0.010 * sowing_delay)               # range 0.64-1.0

# Organic matter: slow-release nutrients improve yield
om_factor = 0.82 + 0.18 * (organic_matter / 6.0)           # range 0.82-1.0

# Combine multiplicatively — this creates strong realistic correlations
yield_raw = (base_yield *
             npk_factor *
             ph_factor *
             ndvi_factor *
             rainfall_factor *
             temp_factor *
             sowing_factor *
             om_factor)

# Realistic field-level noise (~10% of base yield)
noise_std      = np.array([crop_noise[c] for c in crop])
expected_yield = (yield_raw + np.random.normal(0, noise_std, N_ROWS)).clip(150, 13000).round(2)

# ── TARGET 2: DISEASE OUTBREAK — plant pathology relationships ───────────────
# Primary drivers from published disease epidemiology:
# 1. Humidity >70% + Temperature >25°C → fungal disease (most important)
# 2. Rainfall >150mm/month → spore spread, waterlogging, root rot
# 3. NDVI <0.4 → stressed/weakened plant → more susceptible
# 4. Negative NDVI anomaly → health declining → early disease signal

humidity_risk   = np.clip((humidity - 55) / 40, 0, 1)
temp_risk       = np.clip((temperature - 18) / 20, 0, 1)
ht_interaction  = humidity_risk * temp_risk                 # must BOTH be high

rainfall_risk   = np.clip((rainfall - 80) / 220, 0, 1)
ndvi_weak_risk  = np.clip((0.45 - ndvi) / 0.45, 0, 1)
anomaly_risk    = np.clip(-ndvi_anomaly / 0.25, 0, 1)

# Weighted probability
outbreak_logit = (
    0.40 * ht_interaction +
    0.25 * rainfall_risk +
    0.20 * ndvi_weak_risk +
    0.15 * anomaly_risk
)

# Add noise and apply threshold to get ~20-25% outbreak rate
outbreak_prob  = (outbreak_logit + np.random.normal(0, 0.07, N_ROWS)).clip(0, 1)
disease_outbreak = (outbreak_prob > 0.22).astype(int)

outbreak_rate = disease_outbreak.mean() * 100
print(f"    Disease outbreak rate : {outbreak_rate:.1f}%")
print(f"    Mean yield            : {expected_yield.mean():.0f} kg/ha")
print(f"    Yield std             : {expected_yield.std():.0f} kg/ha")

# ── ASSEMBLE ─────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "crop_type":             crop,
    "agro_zone":             agro_zone,
    "region":                region,
    "season":                season,
    "crop_growth_stage":     crop_growth_stage,
    "sowing_delay_days":     sowing_delay.round(1),
    "soil_nitrogen":         soil_nitrogen.round(2),
    "soil_phosphorus":       soil_phosphorus.round(2),
    "soil_potassium":        soil_potassium.round(2),
    "soil_ph":               soil_ph.round(3),
    "soil_moisture":         soil_moisture.round(2),
    "organic_matter":        organic_matter.round(3),
    "temperature":           temperature.round(2),
    "humidity":              humidity.round(2),
    "rainfall":              rainfall.round(2),
    "wind_speed":            wind_speed.round(2),
    "growing_degree_days":   gdd.round(0).astype(int),
    "ndvi":                  ndvi.round(4),
    "savi":                  savi.round(4),
    "ndvi_anomaly":          ndvi_anomaly.round(4),
    "chlorophyll_content":   chlorophyll.round(2),
    "leaf_area_index":       leaf_area_index.round(3),
    "expected_yield_kg_ha":  expected_yield,
    "disease_outbreak":      disease_outbreak,
})

# ── VERIFY CORRELATIONS ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("CORRELATION VERIFICATION")
print("="*60)
num_df = df.select_dtypes(include="number")

print("\nTop correlations with expected_yield_kg_ha:")
yc = (num_df.corr()["expected_yield_kg_ha"]
      .drop("expected_yield_kg_ha")
      .sort_values(key=abs, ascending=False).head(10))
print(yc.round(4).to_string())

print("\nTop correlations with disease_outbreak:")
dc = (num_df.corr()["disease_outbreak"]
      .drop("disease_outbreak")
      .sort_values(key=abs, ascending=False).head(10))
print(dc.round(4).to_string())

# ── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "█"*60)
print("  DATASET GENERATION COMPLETE ✅")
print("█"*60)
print(f"\n  Rows     : {len(df):,}")
print(f"  Columns  : {len(df.columns)}")
print(f"  Saved to : {OUTPUT_PATH}")
print(f"\n  Regression  target : expected_yield_kg_ha")
print(f"  Classification target : disease_outbreak ({outbreak_rate:.1f}% outbreak rate)")
print(f"\n  Next -> run: python src/data/ingest.py")
print("█"*60 + "\n")