"""
Fair Comparison: Binary vs Exponential Propagation (Independently Optimised Weights)

Addresses peer reviewer concern M4: the original binary-vs-exponential comparison
used weights optimised for the exponential model to evaluate the binary model.
This script gives each model its own independently LOOCV-optimised non-degenerate
weights so the comparison is on equal footing.

Two propagation scores are computed:

  Exponential (same as script 65):
      travel_time_h = distance_to_station_km / 5.0
      prop_raw      = exp(-travel_time_h / 6.0)
      prop_exp      = min-max normalised per event

  Binary (faithful to original design):
      prop_binary_raw = 0.7 if distance_to_station_km > 1.0 else 0.3
      prop_binary     = min-max normalised per event
                        (collapses to 0.5 when all facilities share the same value)

Each model is then subjected to LOOCV with per-fold weight grid-search on the
same narrow grid used in script 65.  Degenerate single-component solutions are
excluded by the grid bounds (w_p <= 0.45 already excludes w_p = 1.0).

Outputs
-------
results/fair_binary_comparison/
    fair_comparison_results.csv   -- per-event ranks and top-10 flags for both models
    fair_comparison_summary.json  -- aggregate statistics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/fair_binary_comparison")
OUT.mkdir(parents=True, exist_ok=True)

TRI_REGISTRY   = 110028001187
DEFAULT_VEL    = 5.0   # km/h
DEFAULT_DECAY  = 6.0   # hours
UPSTREAM_KM    = 1.0   # distance threshold for binary upstream classification

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
print("FAIR BINARY vs EXPONENTIAL COMPARISON (independently optimised weights)")
print("=" * 70)
print(f"\nTotal events: {len(confirmed_events)}")
by_st = data[data["REGISTRY_ID"] == TRI_REGISTRY].groupby("affected_station")["event_id"].nunique()
print(f"Events by station: {by_st.to_dict()}")
by_par = data[data["REGISTRY_ID"] == TRI_REGISTRY].groupby("event_parameter")["event_id"].nunique()
print(f"Events by parameter: {by_par.to_dict()}")


# ── Propagation score computation ─────────────────────────────────────────────

def add_exp_propagation(df_in):
    """Add exponential decay propagation score (min-max normalised per event)."""
    d = df_in.copy()
    travel_time = d["distance_to_station_km"] / DEFAULT_VEL
    d["prop_raw"] = np.exp(-travel_time / DEFAULT_DECAY)
    g  = d.groupby("event_id")["prop_raw"]
    mn = g.transform("min")
    mx = g.transform("max")
    rng = mx - mn
    d["prop_exp"] = np.where(rng > 0, (d["prop_raw"] - mn) / rng, 0.5)
    return d


def add_binary_propagation(df_in):
    """Add binary upstream propagation score (min-max normalised per event).

    Upstream  (distance > UPSTREAM_KM): raw score = 0.7
    Downstream (distance <= UPSTREAM_KM): raw score = 0.3
    All 178 facilities in this dataset are upstream, so all raw values equal 0.7,
    the min-max range collapses to 0, and every facility receives 0.5 -- faithfully
    replicating the original binary design's inability to distinguish within-station
    upstream candidates.
    """
    d = df_in.copy()
    d["prop_binary_raw"] = np.where(
        d["distance_to_station_km"] > UPSTREAM_KM, 0.7, 0.3
    )
    g  = d.groupby("event_id")["prop_binary_raw"]
    mn = g.transform("min")
    mx = g.transform("max")
    rng = mx - mn
    d["prop_binary"] = np.where(rng > 0, (d["prop_binary_raw"] - mn) / rng, 0.5)
    return d


# ── Scoring and ranking ────────────────────────────────────────────────────────

def score_and_rank_exp(df_ev, w_dist, w_ind, w_anom, w_prop):
    d = df_ev.copy()
    d["score"] = (
        w_dist * d["distance_score_v2"] +
        w_ind  * d["industry_score_v2"] +
        w_anom * d["anomaly_score_v2"]  +
        w_prop * d["prop_exp"]
    )
    d["rank"] = d.groupby("event_id")["score"].rank(ascending=False, method="min")
    return d


def score_and_rank_bin(df_ev, w_dist, w_ind, w_anom, w_prop):
    d = df_ev.copy()
    d["score"] = (
        w_dist * d["distance_score_v2"] +
        w_ind  * d["industry_score_v2"] +
        w_anom * d["anomaly_score_v2"]  +
        w_prop * d["prop_binary"]
    )
    d["rank"] = d.groupby("event_id")["score"].rank(ascending=False, method="min")
    return d


# ── Weight grid search ────────────────────────────────────────────────────────

WEIGHT_GRID = [
    (w_dist, w_ind, w_anom, w_prop)
    for w_dist in [0.00, 0.05, 0.10]
    for w_ind  in [0.00, 0.05, 0.10]
    for w_anom in [0.45, 0.50, 0.55, 0.60]
    for w_prop in [0.30, 0.35, 0.40, 0.45]
    if abs(w_dist + w_ind + w_anom + w_prop - 1.0) <= 0.01
]


def optimize_weights(train_df, score_fn):
    """Grid search for best non-degenerate weights on training data."""
    best_weights = [0.0, 0.0, 0.55, 0.45]
    best_score   = -np.inf

    for w_dist, w_ind, w_anom, w_prop in WEIGHT_GRID:
        ranked = score_fn(train_df, w_dist, w_ind, w_anom, w_prop)
        tri = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY][
            ["event_id", "rank"]
        ].drop_duplicates("event_id")

        if len(tri) == 0:
            continue

        top10  = (tri["rank"] <= 10).mean()
        mean_r = tri["rank"].mean()
        obj    = top10 * 1000 - mean_r     # same objective as original LOOCV

        if obj > best_score:
            best_score   = obj
            best_weights = [w_dist, w_ind, w_anom, w_prop]

    return best_weights


# ── LOOCV runner ──────────────────────────────────────────────────────────────

def run_loocv(station_id, score_fn, optimize_fn, model_label, verbose=True):
    """Full LOOCV with per-fold weight optimisation for one station and one model."""
    station_df = data[data["affected_station"] == station_id].copy()

    # Add the appropriate propagation score column to the entire station slice
    if model_label == "exponential":
        station_df = add_exp_propagation(station_df)
    else:
        station_df = add_binary_propagation(station_df)

    events = station_df[station_df["REGISTRY_ID"] == TRI_REGISTRY]["event_id"].unique()
    n      = len(events)

    if verbose:
        print(f"\n  Station {station_id}: {n} events")

    fold_results = []
    fold_weights = []

    for i, test_ev in enumerate(events, 1):
        train_df = station_df[station_df["event_id"] != test_ev]
        test_df  = station_df[station_df["event_id"] == test_ev].copy()

        weights = optimize_fn(train_df)
        fold_weights.append(weights)

        ranked   = score_fn(test_df, *weights)
        tri_rows = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY]
        rank     = tri_rows["rank"].min()
        param    = tri_rows["event_parameter"].iloc[0]

        fold_results.append({
            "event_id":  test_ev,
            "station":   station_id,
            "model":     model_label,
            "rank":      rank,
            "top10":     int(rank <= 10),
            "parameter": param,
            "weights":   weights,
        })

        if verbose and i % 5 == 0:
            so_far = np.mean([r["top10"] for r in fold_results])
            print(f"    fold {i:>2}/{n}  running top-10: {so_far:.1%}")

    fw = np.array(fold_weights)
    if verbose:
        print(f"  Weight means (dist, ind, anom, prop): {fw.mean(0).round(3)}")
        print(f"  Weight stds:                          {fw.std(0).round(3)}")

    return fold_results, fw


# ── Run LOOCV — EXPONENTIAL model ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 1/2: LOOCV — EXPONENTIAL model (independent weight optimisation)")
print("=" * 70)

exp_results = []
exp_weights = []

for station in [11447650, 11447890]:
    res, wts = run_loocv(
        station,
        score_fn=score_and_rank_exp,
        optimize_fn=lambda tr: optimize_weights(tr, score_and_rank_exp),
        model_label="exponential",
        verbose=True,
    )
    exp_results.extend(res)
    exp_weights.append(wts)

exp_df = pd.DataFrame(exp_results)


# ── Run LOOCV — BINARY model ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2/2: LOOCV — BINARY model (independent weight optimisation)")
print("=" * 70)

bin_results = []
bin_weights = []

for station in [11447650, 11447890]:
    res, wts = run_loocv(
        station,
        score_fn=score_and_rank_bin,
        optimize_fn=lambda tr: optimize_weights(tr, score_and_rank_bin),
        model_label="binary",
        verbose=True,
    )
    bin_results.extend(res)
    bin_weights.append(wts)

bin_df = pd.DataFrame(bin_results)


# ── Aggregate statistics ──────────────────────────────────────────────────────

def summarise(results_df, label):
    overall_n = len(results_df)
    overall_k = results_df["top10"].sum()
    overall_r = results_df["top10"].mean() * 100
    s650 = results_df[results_df["station"] == 11447650]["top10"].mean() * 100
    s890 = results_df[results_df["station"] == 11447890]["top10"].mean() * 100
    n650 = (results_df["station"] == 11447650).sum()
    n890 = (results_df["station"] == 11447890).sum()
    k650 = results_df[results_df["station"] == 11447650]["top10"].sum()
    k890 = results_df[results_df["station"] == 11447890]["top10"].sum()
    mean_r = results_df["rank"].mean()
    med_r  = results_df["rank"].median()

    # modal weights (most frequent across all folds)
    all_w = np.array([r for r in results_df["weights"]])
    modal_w = [float(np.median(all_w[:, i])) for i in range(4)]

    return {
        "model":              label,
        "overall_top10_pct":  round(overall_r, 1),
        "overall_n":          int(overall_n),
        "overall_k":          int(overall_k),
        "mean_rank":          round(mean_r, 2),
        "median_rank":        float(med_r),
        "s11447650_top10_pct": round(s650, 1),
        "s11447890_top10_pct": round(s890, 1),
        "n_11447650":         int(n650),
        "n_11447890":         int(n890),
        "k_11447650":         int(k650),
        "k_11447890":         int(k890),
        "modal_weights_dist_ind_anom_prop": modal_w,
    }


exp_summary = summarise(exp_df, "exponential")
bin_summary = summarise(bin_df, "binary_fair")

# Reference row: binary model with fixed exponential-optimal weights (script 65 values)
ref_summary = {
    "model":               "binary_original_fixed_weights",
    "overall_top10_pct":   65.9,
    "overall_n":           44,
    "overall_k":           29,
    "s11447650_top10_pct": 100.0,
    "s11447890_top10_pct": 28.6,
    "n_11447650":          23,
    "n_11447890":          21,
    "note": "Binary model evaluated with exponential-optimal weights [0,0,0.55,0.45] (unfair baseline)",
}

# ── Print comparison table ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FAIR COMPARISON RESULTS")
print("=" * 70)

ew = exp_summary["modal_weights_dist_ind_anom_prop"]
bw = bin_summary["modal_weights_dist_ind_anom_prop"]

exp_wt_str = f"[{ew[0]:.2f},{ew[1]:.2f},{ew[2]:.2f},{ew[3]:.2f}]"
bin_wt_str = f"[{bw[0]:.2f},{bw[1]:.2f},{bw[2]:.2f},{bw[3]:.2f}]"
ref_wt_str = "[0,0,0.55,0.45]"

hdr = f"{'Model':<22}  {'Weights':<22}  {'Overall':>9}  {'S11447650':>10}  {'S11447890':>10}"
sep = "-" * len(hdr)
print(f"\n{hdr}")
print(sep)
print(
    f"{'Exponential':<22}  {exp_wt_str:<22}  "
    f"{exp_summary['overall_top10_pct']:>8.1f}%  "
    f"{exp_summary['s11447650_top10_pct']:>9.1f}%  "
    f"{exp_summary['s11447890_top10_pct']:>9.1f}%"
)
print(
    f"{'Binary (fair)':<22}  {bin_wt_str:<22}  "
    f"{bin_summary['overall_top10_pct']:>8.1f}%  "
    f"{bin_summary['s11447650_top10_pct']:>9.1f}%  "
    f"{bin_summary['s11447890_top10_pct']:>9.1f}%"
)
print(
    f"{'Binary (orig, unfair)':<22}  {ref_wt_str:<22}  "
    f"{'65.9':>9}%  "
    f"{'100.0':>9}%  "
    f"{'28.6':>9}%   (fixed weights, for reference)"
)
print(sep)

print(f"\nExponential  mean rank: {exp_summary['mean_rank']:.1f},  "
      f"median rank: {exp_summary['median_rank']:.1f}")
print(f"Binary(fair) mean rank: {bin_summary['mean_rank']:.1f},  "
      f"median rank: {bin_summary['median_rank']:.1f}")

# ── Save results ──────────────────────────────────────────────────────────────
combined = pd.concat(
    [exp_df.drop(columns=["weights"]), bin_df.drop(columns=["weights"])],
    ignore_index=True,
)
combined.to_csv(OUT / "fair_comparison_results.csv", index=False)

summary_out = {
    "description": (
        "Fair comparison: each model uses independently LOOCV-optimised "
        "non-degenerate weights. Binary model uses the same narrow weight grid "
        "as the exponential model."
    ),
    "exponential": exp_summary,
    "binary_fair": bin_summary,
    "binary_original_fixed_weights_reference": ref_summary,
}
with open(OUT / "fair_comparison_summary.json", "w") as f:
    json.dump(summary_out, f, indent=2)

print(f"\nResults saved to: {OUT}/")
print("  fair_comparison_results.csv")
print("  fair_comparison_summary.json")
