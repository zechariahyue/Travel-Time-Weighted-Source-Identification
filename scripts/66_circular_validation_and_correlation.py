"""
Circular Validation and Anomaly Correlation (Exponential Propagation)

Two analyses in one script:

1. CIRCULAR VALIDATION with exponential propagation
   - Finds global optimal weights on all 44 events simultaneously
   - Applies those weights to the same 44 events (train = test)
   - Provides apples-to-apples comparison with LOOCV from script 65

2. ANOMALY CORRELATION
   - Point-biserial correlation between AMPAC's event-level anomaly score
     and LOOCV success/failure (from script 65 loocv_results.csv)
   - Summary statistics: anomaly score in success vs failed events
   - Fills the placeholder in manuscript Section 3.3.2

Outputs
-------
results/exponential_propagation/
    circular_validation.json  -- circular performance by station
    anomaly_correlation.json  -- correlation + success/fail stats
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

OUT = Path("results/exponential_propagation")
OUT.mkdir(parents=True, exist_ok=True)

TRI_REGISTRY = 110028001187
DEFAULT_VEL   = 5.0   # km/h
DEFAULT_DECAY = 6.0   # hours

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "results/component_redesign/rankings_with_redesigned_components.csv",
    encoding="utf-8-sig"
)

confirmed_events = df[
    (df["REGISTRY_ID"] == TRI_REGISTRY) &
    (df["affected_station"].isin([11447650, 11447890]))
]["event_id"].unique()

data = df[df["event_id"].isin(confirmed_events)].copy()

print("=" * 70)
print("SCRIPT 66: CIRCULAR VALIDATION + ANOMALY CORRELATION")
print("=" * 70)
print(f"\nEvents: {len(confirmed_events)}")

# ── Add exponential propagation (identical to script 65) ──────────────────────
def add_exp_propagation(df_in, velocity_kmh, decay_h):
    d = df_in.copy()
    travel_time = d["distance_to_station_km"] / velocity_kmh
    d["prop_raw"] = np.exp(-travel_time / decay_h)
    g = d.groupby("event_id")["prop_raw"]
    mn = g.transform("min")
    mx = g.transform("max")
    rng = mx - mn
    d["prop_exp"] = np.where(rng > 0, (d["prop_raw"] - mn) / rng, 0.5)
    return d

data = add_exp_propagation(data, DEFAULT_VEL, DEFAULT_DECAY)


def score_and_rank(df_ev, w_dist, w_ind, w_anom, w_prop):
    d = df_ev.copy()
    d["score"] = (
        w_dist * d["distance_score_v2"] +
        w_ind  * d["industry_score_v2"] +
        w_anom * d["anomaly_score_v2"]  +
        w_prop * d["prop_exp"]
    )
    d["rank"] = d.groupby("event_id")["score"].rank(ascending=False, method="min")
    return d


# ── PART 1: Circular Validation ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 1: CIRCULAR VALIDATION (global weight optimisation)")
print("=" * 70)

# Grid search on ALL 44 events
best_weights_circ = [0.0, 0.0, 0.55, 0.45]
best_obj_circ     = -np.inf

for w_dist in [0.00, 0.05, 0.10]:
    for w_ind in [0.00, 0.05, 0.10]:
        for w_anom in [0.45, 0.50, 0.55, 0.60]:
            for w_prop in [0.30, 0.35, 0.40, 0.45]:
                if abs(w_dist + w_ind + w_anom + w_prop - 1.0) > 0.01:
                    continue

                ranked = score_and_rank(data, w_dist, w_ind, w_anom, w_prop)
                tri = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY][
                    ["event_id", "rank"]
                ].drop_duplicates("event_id")

                if len(tri) == 0:
                    continue

                top10    = (tri["rank"] <= 10).mean()
                mean_r   = tri["rank"].mean()
                obj      = top10 * 1000 - mean_r

                if obj > best_obj_circ:
                    best_obj_circ     = obj
                    best_weights_circ = [w_dist, w_ind, w_anom, w_prop]

print(f"\nGlobal optimal weights: {best_weights_circ}")

# Apply global weights to all events (train = test)
ranked_all = score_and_rank(data, *best_weights_circ)
tri_all = ranked_all[ranked_all["REGISTRY_ID"] == TRI_REGISTRY].copy()
# One row per event: take the minimum rank (best rank for AMPAC in event)
tri_ev = tri_all.groupby("event_id").agg(
    rank=("rank", "min"),
    station=("affected_station", "first"),
    parameter=("event_parameter", "first")
).reset_index()
tri_ev["top10"] = (tri_ev["rank"] <= 10).astype(int)

circ_overall = tri_ev["top10"].mean() * 100
circ_n       = len(tri_ev)
circ_k       = tri_ev["top10"].sum()
circ_s650    = tri_ev[tri_ev["station"] == 11447650]["top10"].mean() * 100
circ_s890    = tri_ev[tri_ev["station"] == 11447890]["top10"].mean() * 100
n_s650       = (tri_ev["station"] == 11447650).sum()
n_s890       = (tri_ev["station"] == 11447890).sum()

print(f"\nCircular validation results (exponential propagation):")
print(f"  Overall:          {circ_overall:.1f}%  ({int(circ_k)}/{circ_n})")
print(f"  Station 11447650: {circ_s650:.1f}%  (n={n_s650})")
print(f"  Station 11447890: {circ_s890:.1f}%  (n={n_s890})")
print(f"\nRank distribution:")
print(f"  Mean rank:   {tri_ev['rank'].mean():.1f}")
print(f"  Median rank: {tri_ev['rank'].median():.1f}")
print(f"  Unique ranks: {sorted(tri_ev['rank'].unique())}")

circ_results = {
    "velocity_kmh":           DEFAULT_VEL,
    "decay_h":                DEFAULT_DECAY,
    "global_optimal_weights": best_weights_circ,
    "overall_top10_pct":      round(circ_overall, 1),
    "overall_n":              int(circ_n),
    "overall_k":              int(circ_k),
    "mean_rank":              round(float(tri_ev["rank"].mean()), 2),
    "median_rank":            float(tri_ev["rank"].median()),
    "station_11447650_top10_pct": round(circ_s650, 1),
    "station_11447890_top10_pct": round(circ_s890, 1),
    "n_11447650":             int(n_s650),
    "n_11447890":             int(n_s890),
}
with open(OUT / "circular_validation.json", "w") as f:
    json.dump(circ_results, f, indent=2)
print(f"\nSaved: {OUT}/circular_validation.json")


# ── PART 2: Anomaly Correlation ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2: ANOMALY CORRELATION WITH LOOCV SUCCESS")
print("=" * 70)

# Load LOOCV results from script 65
loocv_df = pd.read_csv(OUT / "loocv_results.csv")
print(f"\nLoaded LOOCV results: {len(loocv_df)} events")

# Get AMPAC's anomaly_score_v2 for each event (it's event-level constant)
ampac_data = data[data["REGISTRY_ID"] == TRI_REGISTRY][
    ["event_id", "anomaly_score_v2", "distance_score_v2",
     "industry_score_v2", "prop_exp", "affected_station"]
].drop_duplicates("event_id").copy()

# Merge with LOOCV top10 outcome
merged = loocv_df.merge(ampac_data, on="event_id", how="left")

print(f"\nAnomalyscore_v2 for AMPAC (event-level constant within event):")
print(f"  Range: [{merged['anomaly_score_v2'].min():.4f}, {merged['anomaly_score_v2'].max():.4f}]")
print(f"  Mean:  {merged['anomaly_score_v2'].mean():.4f}")
print(f"  Std:   {merged['anomaly_score_v2'].std():.4f}")

# Point-biserial correlation: anomaly_score_v2 vs top10
r_anom, p_anom = scipy_stats.pointbiserialr(
    merged["top10"], merged["anomaly_score_v2"]
)
print(f"\nPoint-biserial correlation (anomaly vs success):")
print(f"  r = {r_anom:.4f},  p = {p_anom:.4f}")

# Success vs failed event anomaly scores
success_anom = merged[merged["top10"] == 1]["anomaly_score_v2"]
failed_anom  = merged[merged["top10"] == 0]["anomaly_score_v2"]

print(f"\nAnomaly score by outcome:")
print(f"  Success events (n={len(success_anom)}): {success_anom.mean():.4f} +/- {success_anom.std():.4f}")
print(f"  Failed events  (n={len(failed_anom)}):  {failed_anom.mean():.4f} +/- {failed_anom.std():.4f}")
print(f"  Difference: {success_anom.mean() - failed_anom.mean():+.4f}")

# t-test
t_stat, t_p = scipy_stats.ttest_ind(success_anom, failed_anom)
print(f"  t-test: t={t_stat:.3f}, p={t_p:.4f}")

# Also: distance_score_v2 vs top10 (for completeness)
r_dist, p_dist = scipy_stats.pointbiserialr(
    merged["top10"], merged["distance_score_v2"]
)
print(f"\nPoint-biserial correlation (distance vs success):")
print(f"  r = {r_dist:.4f},  p = {p_dist:.4f}")

# Also: prop_exp vs top10
r_prop, p_prop = scipy_stats.pointbiserialr(
    merged["top10"], merged["prop_exp"]
)
print(f"\nPoint-biserial correlation (prop_exp vs success):")
print(f"  r = {r_prop:.4f},  p = {p_prop:.4f}")

# Per-station anomaly analysis
print(f"\nPer-station anomaly stats (success vs failed):")
for st in [11447650, 11447890]:
    st_df = merged[merged["station"] == st]
    s_anom = st_df[st_df["top10"] == 1]["anomaly_score_v2"]
    f_anom = st_df[st_df["top10"] == 0]["anomaly_score_v2"]
    print(f"\n  Station {st}:")
    print(f"    Success (n={len(s_anom)}): {s_anom.mean():.4f} +/- {s_anom.std():.4f}")
    print(f"    Failed  (n={len(f_anom)}): {f_anom.mean():.4f} +/- {f_anom.std():.4f}")

# Save
corr_results = {
    "anomaly_vs_success": {
        "r":       round(float(r_anom), 4),
        "p":       round(float(p_anom), 4),
        "success_mean": round(float(success_anom.mean()), 4),
        "success_std":  round(float(success_anom.std()),  4),
        "failed_mean":  round(float(failed_anom.mean()),  4),
        "failed_std":   round(float(failed_anom.std()),   4),
        "t_stat":  round(float(t_stat), 3),
        "t_p":     round(float(t_p),    4),
    },
    "distance_vs_success": {
        "r": round(float(r_dist), 4),
        "p": round(float(p_dist), 4),
    },
    "propagation_vs_success": {
        "r": round(float(r_prop), 4),
        "p": round(float(p_prop), 4),
    },
}
with open(OUT / "anomaly_correlation.json", "w") as f:
    json.dump(corr_results, f, indent=2)
print(f"\nSaved: {OUT}/anomaly_correlation.json")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
