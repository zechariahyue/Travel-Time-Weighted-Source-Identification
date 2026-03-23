"""
Sensitivity: Mis-attribution, Top-k Thresholds, and sigma_d Sweep

Addresses three peer reviewer concerns simultaneously:

  M6  — mis-attribution sensitivity: what if some of the 12 failed events
         are not actually AMPAC events?
  m2  — sigma_d = 20 km spatial decay in the anomaly score is uncalibrated;
         sweep sigma_d in {5, 10, 15, 20, 30, 40, 60} km.
  m3  — only top-10 was reported; add top-5, top-20, top-50.

Parts
-----
A — Top-k performance at multiple thresholds (m3)
    Loads existing LOOCV results and computes top-5 / top-10 / top-20 / top-50
    success rates overall and per station with Wilson 95% CIs.

B — Mis-attribution sensitivity (M6)
    Assumes k of the 12 failed events might not be AMPAC events and removes
    them from the denominator; computes adjusted success rate for
    k in {0, 1, 2, 3, 5, 10}.

C — sigma_d sensitivity (m2)
    Recomputes anomaly_score_v2 from scratch for each sigma_d value using only
    the distance-to-station field (g(e) cancels in normalisation); runs scoring
    with fixed optimal weights [0, 0, 0.60, 0.40] and velocity=5 km/h,
    decay=6 h; reports top-10 rate for each sigma_d.

Outputs  (results/sensitivity_misattribution/)
---------
  topk_performance.csv
  misattribution_sensitivity.csv
  sigma_d_sensitivity.csv
  summary.json

Data
----
  results/component_redesign/rankings_with_redesigned_components.csv
  results/exponential_propagation/loocv_results.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

# ── Constants ─────────────────────────────────────────────────────────────────

TRI_REGISTRY    = 110028001187
STATIONS        = [11447650, 11447890]
OPT_WEIGHTS     = [0.0, 0.0, 0.60, 0.40]   # [w_dist, w_ind, w_anom, w_prop]
DEFAULT_VEL     = 5.0    # km/h
DEFAULT_DECAY   = 6.0    # hours
DEFAULT_SIGMA_D = 20.0   # km  (sigma_d used when building anomaly_score_v2)

OUT = Path("results/sensitivity_misattribution")
OUT.mkdir(parents=True, exist_ok=True)

# ── Helper: Wilson 95% CI ─────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Return (lo, hi) Wilson 95% CI as fractions (not percentages)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre - margin) / denom, (centre + margin) / denom


# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

loocv = pd.read_csv(
    "results/exponential_propagation/loocv_results.csv",
    encoding="utf-8-sig",
)
print(f"LOOCV results: {len(loocv)} events")
print(f"  Overall top-10: {loocv['top10'].sum()}/{len(loocv)}")

df_raw = pd.read_csv(
    "results/component_redesign/rankings_with_redesigned_components.csv",
    encoding="utf-8-sig",
)

# Restrict to the 44 confirmed AMPAC events across both stations
confirmed_event_ids = set(loocv["event_id"].unique())
data = df_raw[df_raw["event_id"].isin(confirmed_event_ids)].copy()
print(f"Main CSV rows for confirmed events: {len(data)}")


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Top-k at multiple thresholds
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART A — TOP-k PERFORMANCE AT MULTIPLE THRESHOLDS")
print("=" * 70)

thresholds = [5, 10, 20, 50]
topk_records = []

n_overall = len(loocv)
sub_650   = loocv[loocv["station"] == 11447650]
sub_890   = loocv[loocv["station"] == 11447890]

header = f"{'Threshold':<12} {'Overall (%)':<14} {'95% CI':<18} {'S11447650 (%)':<16} {'S11447890 (%)'}"
print("\n" + header)
print("-" * len(header))

for k in thresholds:
    k_overall = (loocv["rank"] <= k).sum()
    rate_ov   = k_overall / n_overall * 100
    ci_lo, ci_hi = wilson_ci(k_overall, n_overall)

    k_650  = (sub_650["rank"] <= k).sum()
    rate_650 = k_650 / len(sub_650) * 100

    k_890  = (sub_890["rank"] <= k).sum()
    rate_890 = k_890 / len(sub_890) * 100

    ci_str = f"[{ci_lo*100:.1f}, {ci_hi*100:.1f}]"
    print(
        f"Top-{k:<7} {rate_ov:<14.1f} {ci_str:<18} {rate_650:<16.1f} {rate_890:.1f}"
    )

    topk_records.append({
        "threshold":         k,
        "overall_pct":       round(rate_ov, 1),
        "overall_k":         int(k_overall),
        "overall_n":         int(n_overall),
        "ci_lo_pct":         round(ci_lo * 100, 1),
        "ci_hi_pct":         round(ci_hi * 100, 1),
        "s11447650_pct":     round(rate_650, 1),
        "s11447650_k":       int(k_650),
        "s11447650_n":       int(len(sub_650)),
        "s11447890_pct":     round(rate_890, 1),
        "s11447890_k":       int(k_890),
        "s11447890_n":       int(len(sub_890)),
    })

topk_df = pd.DataFrame(topk_records)
topk_df.to_csv(OUT / "topk_performance.csv", index=False)
print(f"\nSaved: {OUT / 'topk_performance.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Mis-attribution sensitivity
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART B — MIS-ATTRIBUTION SENSITIVITY")
print("=" * 70)

n_total    = len(loocv)                        # 44
k_success  = int(loocv["top10"].sum())         # 32
n_failures = n_total - k_success               # 12

print(f"\nBaseline: {k_success}/{n_total} successes, {n_failures} failures")
print("If k of the 12 failed events are mis-attributed (not AMPAC), remove")
print("them from the denominator (numerator stays at 32).\n")

# Per-station failure counts (needed for proportional per-station adjustment)
fail_650 = int((sub_650["top10"] == 0).sum())
fail_890 = int((sub_890["top10"] == 0).sum())
succ_650 = int(sub_650["top10"].sum())
succ_890 = int(sub_890["top10"].sum())
n_650    = int(len(sub_650))
n_890    = int(len(sub_890))

print(f"Station 11447650: {succ_650}/{n_650} successes, {fail_650} failures")
print(f"Station 11447890: {succ_890}/{n_890} successes, {fail_890} failures")

k_vals = [0, 1, 2, 3, 5, 10]
misattrib_records = []

header2 = (
    f"{'k removed':<12} {'Adj. overall':<16} {'95% CI':<20}"
    f"{'S11447650':<14} {'S11447890':<14} Interpretation"
)
print("\n" + header2)
print("-" * len(header2))

for k in k_vals:
    n_adj = n_total - k
    rate_adj = k_success / n_adj * 100
    ci_lo, ci_hi = wilson_ci(k_success, n_adj)
    ci_str = f"[{ci_lo*100:.1f}, {ci_hi*100:.1f}]"

    # Proportional split of k across stations (round to nearest int)
    k_650_removed = round(k * fail_650 / n_failures) if n_failures > 0 else 0
    k_890_removed = k - k_650_removed

    n_650_adj = n_650 - k_650_removed
    n_890_adj = n_890 - k_890_removed

    rate_650_adj = succ_650 / n_650_adj * 100 if n_650_adj > 0 else float("nan")
    rate_890_adj = succ_890 / n_890_adj * 100 if n_890_adj > 0 else float("nan")

    if k == 0:
        interp = "Baseline (all 44 attributed)"
    elif k == n_failures:
        interp = "All failures removed"
    else:
        interp = f"{k} of {n_failures} failures mis-attributed"

    print(
        f"{k:<12} {rate_adj:<16.1f} {ci_str:<20}"
        f"{rate_650_adj:<14.1f} {rate_890_adj:<14.1f} {interp}"
    )

    misattrib_records.append({
        "k_removed":           k,
        "n_adjusted":          n_adj,
        "k_success":           k_success,
        "adj_overall_pct":     round(rate_adj, 1),
        "ci_lo_pct":           round(ci_lo * 100, 1),
        "ci_hi_pct":           round(ci_hi * 100, 1),
        "k_650_removed":       k_650_removed,
        "n_650_adj":           n_650_adj,
        "adj_s11447650_pct":   round(rate_650_adj, 1),
        "k_890_removed":       k_890_removed,
        "n_890_adj":           n_890_adj,
        "adj_s11447890_pct":   round(rate_890_adj, 1),
        "interpretation":      interp,
    })

misattrib_df = pd.DataFrame(misattrib_records)
misattrib_df.to_csv(OUT / "misattribution_sensitivity.csv", index=False)
print(f"\nSaved: {OUT / 'misattribution_sensitivity.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART C — sigma_d sensitivity
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART C — sigma_d SENSITIVITY")
print("=" * 70)

# ── Recompute exponential propagation score (same as script 65) ───────────────

def add_exp_propagation(df_in, velocity_kmh, decay_h):
    """Exponential decay propagation score, min-max normalised per event."""
    d = df_in.copy()
    travel_time   = d["distance_to_station_km"] / velocity_kmh
    d["prop_raw"] = np.exp(-travel_time / decay_h)
    g    = d.groupby("event_id")["prop_raw"]
    mn   = g.transform("min")
    mx   = g.transform("max")
    rng  = mx - mn
    d["prop_exp"] = np.where(rng > 0, (d["prop_raw"] - mn) / rng, 0.5)
    return d


def recompute_anomaly(df_in, sigma_d_km):
    """
    Recompute anomaly score with an alternative sigma_d.

    Within each event, g(e) (the event-level anomaly magnitude) is constant
    across all facilities, so it cancels in the normalisation step:

        tilde_a(f,e) = exp(-d_f / sigma_d) / max_f exp(-d_f / sigma_d)

    For events where the original anomaly_score_v2 is uniformly 1.0 for all
    facilities (i.e., g(e) ~ 0 — no anomaly signal), tilde_a is irrelevant
    because the component contributes nothing discriminatory regardless of
    sigma_d.  We detect these events and set a_new = 1.0 (same as original).

    For all other events, tilde_a_new is computed from distance alone.
    a_new = 1 - tilde_a_new  (higher score = farther away = less suspicious).
    """
    d = df_in.copy()

    # Identify zero-anomaly events: all facilities have anomaly_score_v2 == 1.0
    ev_anom_std = d.groupby("event_id")["anomaly_score_v2"].std().fillna(0)
    zero_anom_events = set(ev_anom_std[ev_anom_std == 0].index)

    # Compute raw spatial weight for new sigma_d
    d["exp_neg_d"] = np.exp(-d["distance_to_station_km"] / sigma_d_km)

    # Normalise within event
    g_max = d.groupby("event_id")["exp_neg_d"].transform("max")
    tilde_a_new = d["exp_neg_d"] / g_max.where(g_max > 0, other=1.0)

    a_new = 1.0 - tilde_a_new

    # For zero-anomaly events, preserve original anomaly_score_v2 (= 1.0 everywhere)
    is_zero_anom = d["event_id"].isin(zero_anom_events)
    a_new = a_new.where(~is_zero_anom, other=d["anomaly_score_v2"])

    d["anomaly_score_v2"] = a_new
    return d


def score_and_rank_fixed(df_ev, w_dist, w_ind, w_anom, w_prop):
    """Apply fixed weights and rank within event."""
    d = df_ev.copy()
    d["score"] = (
        w_dist * d["distance_score_v2"] +
        w_ind  * d["industry_score_v2"] +
        w_anom * d["anomaly_score_v2"]  +
        w_prop * d["prop_exp"]
    )
    d["rank"] = d.groupby("event_id")["score"].rank(ascending=False, method="min")
    return d


# Pre-compute exponential propagation (fixed vel=5, decay=6)
data_with_prop = add_exp_propagation(data, DEFAULT_VEL, DEFAULT_DECAY)

w_d, w_i, w_a, w_p = OPT_WEIGHTS

sigma_d_vals = [5, 10, 15, 20, 30, 40, 60]

print(f"\nFixed weights: w_dist={w_d}, w_ind={w_i}, w_anom={w_a}, w_prop={w_p}")
print(f"Fixed velocity={DEFAULT_VEL} km/h, decay={DEFAULT_DECAY} h")
print(f"Sweeping sigma_d: {sigma_d_vals} km\n")

header3 = (
    f"{'sigma_d (km)':<14} {'Overall (%)':<14} {'95% CI':<20}"
    f"{'S11447650 (%)':<16} {'S11447890 (%)':<16} {'Median rank'}"
)
print(header3)
print("-" * len(header3))

sigma_records = []

for sigma_d in sigma_d_vals:
    # Recompute anomaly score for this sigma_d
    tmp = recompute_anomaly(data_with_prop, sigma_d)

    ev_results = []
    for st in STATIONS:
        st_df  = tmp[tmp["affected_station"] == st]
        events = st_df[st_df["REGISTRY_ID"] == TRI_REGISTRY]["event_id"].unique()
        for ev in events:
            test   = st_df[st_df["event_id"] == ev].copy()
            ranked = score_and_rank_fixed(test, w_d, w_i, w_a, w_p)
            rank   = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY]["rank"].min()
            ev_results.append({"station": st, "rank": rank, "top10": int(rank <= 10)})

    r         = pd.DataFrame(ev_results)
    n_tot     = len(r)
    k_tot     = r["top10"].sum()
    overall   = k_tot / n_tot * 100
    ci_lo, ci_hi = wilson_ci(k_tot, n_tot)
    ci_str    = f"[{ci_lo*100:.1f}, {ci_hi*100:.1f}]"

    r_650     = r[r["station"] == 11447650]
    r_890     = r[r["station"] == 11447890]
    s650      = r_650["top10"].mean() * 100
    s890      = r_890["top10"].mean() * 100
    med_rank  = r["rank"].median()

    mark      = " <-- default" if sigma_d == DEFAULT_SIGMA_D else ""
    print(
        f"{sigma_d:<14} {overall:<14.1f} {ci_str:<20}"
        f"{s650:<16.1f} {s890:<16.1f} {med_rank:.1f}{mark}"
    )

    sigma_records.append({
        "sigma_d_km":      sigma_d,
        "overall_pct":     round(overall, 1),
        "overall_k":       int(k_tot),
        "overall_n":       int(n_tot),
        "ci_lo_pct":       round(ci_lo * 100, 1),
        "ci_hi_pct":       round(ci_hi * 100, 1),
        "s11447650_pct":   round(s650, 1),
        "s11447890_pct":   round(s890, 1),
        "median_rank":     float(med_rank),
        "is_default":      sigma_d == DEFAULT_SIGMA_D,
    })

sigma_df = pd.DataFrame(sigma_records)
sigma_df.to_csv(OUT / "sigma_d_sensitivity.csv", index=False)

print(f"\nOverall range across sigma_d: "
      f"{sigma_df['overall_pct'].min():.1f}% - {sigma_df['overall_pct'].max():.1f}%")
print(f"S11447650 range: "
      f"{sigma_df['s11447650_pct'].min():.1f}% - {sigma_df['s11447650_pct'].max():.1f}%")
print(f"S11447890 range: "
      f"{sigma_df['s11447890_pct'].min():.1f}% - {sigma_df['s11447890_pct'].max():.1f}%")
print(f"Median rank range: "
      f"{sigma_df['median_rank'].min():.1f} - {sigma_df['median_rank'].max():.1f}")
print(f"\nSaved: {OUT / 'sigma_d_sensitivity.csv'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Save summary JSON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART D — SAVING SUMMARY")
print("=" * 70)

# Grab the top-10 row for the main CI (matches baseline LOOCV)
top10_row = topk_df[topk_df["threshold"] == 10].iloc[0]
baseline_row = misattrib_df[misattrib_df["k_removed"] == 0].iloc[0]
default_sigma_row = sigma_df[sigma_df["sigma_d_km"] == DEFAULT_SIGMA_D].iloc[0]

summary = {
    # Part A
    "topk": {
        str(k): {
            "overall_pct":   float(topk_df[topk_df["threshold"] == k]["overall_pct"].iloc[0]),
            "ci_lo_pct":     float(topk_df[topk_df["threshold"] == k]["ci_lo_pct"].iloc[0]),
            "ci_hi_pct":     float(topk_df[topk_df["threshold"] == k]["ci_hi_pct"].iloc[0]),
            "s11447650_pct": float(topk_df[topk_df["threshold"] == k]["s11447650_pct"].iloc[0]),
            "s11447890_pct": float(topk_df[topk_df["threshold"] == k]["s11447890_pct"].iloc[0]),
        }
        for k in thresholds
    },
    # Part B
    "misattribution": {
        "n_total":    int(n_total),
        "k_success":  int(k_success),
        "n_failures": int(n_failures),
        "scenarios":  [
            {
                "k_removed":       int(r["k_removed"]),
                "adj_overall_pct": float(r["adj_overall_pct"]),
                "ci_lo_pct":       float(r["ci_lo_pct"]),
                "ci_hi_pct":       float(r["ci_hi_pct"]),
            }
            for _, r in misattrib_df.iterrows()
        ],
    },
    # Part C
    "sigma_d": {
        "default_sigma_d_km": DEFAULT_SIGMA_D,
        "sweep_km": sigma_d_vals,
        "overall_range_pct": [
            float(sigma_df["overall_pct"].min()),
            float(sigma_df["overall_pct"].max()),
        ],
        "results": [
            {
                "sigma_d_km":    float(r["sigma_d_km"]),
                "overall_pct":   float(r["overall_pct"]),
                "ci_lo_pct":     float(r["ci_lo_pct"]),
                "ci_hi_pct":     float(r["ci_hi_pct"]),
                "s11447650_pct": float(r["s11447650_pct"]),
                "s11447890_pct": float(r["s11447890_pct"]),
                "median_rank":   float(r["median_rank"]),
            }
            for _, r in sigma_df.iterrows()
        ],
    },
}

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to: {OUT}/")
print("  topk_performance.csv")
print("  misattribution_sensitivity.csv")
print("  sigma_d_sensitivity.csv")
print("  summary.json")
print("\nDone.")
