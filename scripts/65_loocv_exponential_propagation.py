"""
LOOCV with Proper Exponential Decay Propagation Score

Replaces the binary upstream-direction propagation score with the formula
described in the manuscript:

    travel_time_h  = distance_to_station_km / velocity_kmh
    prop_raw       = exp(-travel_time_h / decay_h)
    prop_score     = min-max normalised per event

Then runs LOOCV with per-fold weight grid-search, exactly as in script 44,
but using the recomputed propagation scores.

Default parameters: velocity = 5.0 km/h, decay = 6.0 h

Then runs a sensitivity grid over velocity ∈ {3,4,5,6,7,8} km/h
and decay ∈ {3,4.5,6,9,12} h, fixing weights at their LOOCV-optimal values.

Outputs
-------
results/exponential_propagation/
    loocv_results.csv        — per-event rank, station, success
    loocv_summary.json       — aggregate statistics
    sensitivity_grid.csv     — top-10 rate for each (velocity, decay)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/exponential_propagation")
OUT.mkdir(parents=True, exist_ok=True)

TRI_REGISTRY   = 110028001187
DEFAULT_VEL    = 5.0   # km/h  (typical Sacramento River flow velocity)
DEFAULT_DECAY  = 6.0   # hours (weights facilities within ~±12 h of expected arrival)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "results/component_redesign/rankings_with_redesigned_components.csv",
    encoding="utf-8-sig"
)

# Keep only the two stations used in the original analysis
# (station 11455478 has only 2 AMPAC events and was excluded)
confirmed_events = df[
    (df["REGISTRY_ID"] == TRI_REGISTRY) &
    (df["affected_station"].isin([11447650, 11447890]))
]["event_id"].unique()

data = df[df["event_id"].isin(confirmed_events)].copy()

print("=" * 70)
print("LOOCV WITH EXPONENTIAL DECAY PROPAGATION")
print("=" * 70)
print(f"\nTotal events: {len(confirmed_events)}")
by_st = data[data["REGISTRY_ID"]==TRI_REGISTRY].groupby("affected_station")["event_id"].nunique()
print(f"Events by station: {by_st.to_dict()}")
by_par = data[data["REGISTRY_ID"]==TRI_REGISTRY].groupby("event_parameter")["event_id"].nunique()
print(f"Events by parameter: {by_par.to_dict()}")


# ── Core functions ────────────────────────────────────────────────────────────

def add_exp_propagation(df_in, velocity_kmh, decay_h):
    """Add exponential decay propagation score (normalised per event)."""
    d = df_in.copy()
    travel_time = d["distance_to_station_km"] / velocity_kmh
    d["prop_raw"] = np.exp(-travel_time / decay_h)

    # Min-max normalise within each event
    g = d.groupby("event_id")["prop_raw"]
    mn = g.transform("min")
    mx = g.transform("max")
    rng = mx - mn
    d["prop_exp"] = np.where(rng > 0, (d["prop_raw"] - mn) / rng, 0.5)
    return d


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


def optimize_weights(train_df):
    """Grid search for best weights on training data."""
    best_weights = [0.0, 0.0, 0.55, 0.45]
    best_score   = -np.inf

    for w_dist in [0.00, 0.05, 0.10]:
        for w_ind in [0.00, 0.05, 0.10]:
            for w_anom in [0.45, 0.50, 0.55, 0.60]:
                for w_prop in [0.30, 0.35, 0.40, 0.45]:
                    if abs(w_dist + w_ind + w_anom + w_prop - 1.0) > 0.01:
                        continue

                    ranked = score_and_rank(train_df, w_dist, w_ind, w_anom, w_prop)
                    tri = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY][
                        ["event_id", "rank"]
                    ].drop_duplicates("event_id")

                    if len(tri) == 0:
                        continue

                    top10 = (tri["rank"] <= 10).mean()
                    mean_r = tri["rank"].mean()
                    obj = top10 * 1000 - mean_r     # same objective as original LOOCV

                    if obj > best_score:
                        best_score   = obj
                        best_weights = [w_dist, w_ind, w_anom, w_prop]

    return best_weights


def run_loocv(station_id, velocity_kmh, decay_h, verbose=True):
    """Full LOOCV with per-fold weight optimisation for one station."""
    station_df = data[data["affected_station"] == station_id].copy()
    station_df = add_exp_propagation(station_df, velocity_kmh, decay_h)

    events = station_df[station_df["REGISTRY_ID"] == TRI_REGISTRY]["event_id"].unique()
    n      = len(events)

    if verbose:
        print(f"\n  Station {station_id}: {n} events")

    fold_results = []
    fold_weights = []

    for i, test_ev in enumerate(events, 1):
        train_df = station_df[station_df["event_id"] != test_ev]
        test_df  = station_df[station_df["event_id"] == test_ev].copy()

        weights = optimize_weights(train_df)
        fold_weights.append(weights)

        ranked = score_and_rank(test_df, *weights)
        tri_rows = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY]
        rank = tri_rows["rank"].min()
        param = tri_rows["event_parameter"].iloc[0]

        fold_results.append({
            "event_id": test_ev,
            "station":  station_id,
            "rank":     rank,
            "top10":    int(rank <= 10),
            "parameter": param,
            "weights":  weights,
        })

        if verbose and i % 5 == 0:
            so_far = np.mean([r["top10"] for r in fold_results])
            print(f"    fold {i:>2}/{n}  running top-10: {so_far:.1%}")

    # Weight stability
    fw = np.array(fold_weights)
    if verbose:
        print(f"  Weight means (dist, ind, anom, prop): {fw.mean(0).round(3)}")
        print(f"  Weight stds:                          {fw.std(0).round(3)}")

    return fold_results, fw


# ── Run LOOCV for default parameters ─────────────────────────────────────────
print(f"\nDefault parameters: velocity={DEFAULT_VEL} km/h, decay={DEFAULT_DECAY} h")

all_results = []
all_weights = []

for station in [11447650, 11447890]:
    res, wts = run_loocv(station, DEFAULT_VEL, DEFAULT_DECAY, verbose=True)
    all_results.extend(res)
    all_weights.append(wts)

results_df = pd.DataFrame(all_results)

# ── Aggregate statistics ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

overall_rate = results_df["top10"].mean() * 100
overall_n    = len(results_df)
overall_k    = results_df["top10"].sum()
mean_rank    = results_df["rank"].mean()
med_rank     = results_df["rank"].median()

print(f"\nOverall:  {overall_rate:.1f}%  ({overall_k}/{overall_n})")
print(f"Mean rank: {mean_rank:.1f},  Median rank: {med_rank:.1f}")

for st in [11447650, 11447890]:
    sub = results_df[results_df["station"] == st]
    rate = sub["top10"].mean() * 100
    n    = len(sub)
    k    = sub["top10"].sum()
    print(f"Station {st}: {rate:.1f}%  ({k}/{n})")

# Binomial p-value vs 50% null
from scipy import stats as scipy_stats
binom_p = scipy_stats.binom_test(overall_k, overall_n, 0.5, alternative="greater") \
    if hasattr(scipy_stats, "binom_test") \
    else scipy_stats.binomtest(overall_k, overall_n, 0.5, alternative="greater").pvalue
print(f"\np-value (one-sided binomial vs 50%): {binom_p:.4f}")

# Wilson 95% CI
z = 1.96
p_hat = overall_k / overall_n
wilson_lo = (p_hat + z**2/(2*overall_n) - z*np.sqrt(p_hat*(1-p_hat)/overall_n + z**2/(4*overall_n**2))) / (1 + z**2/overall_n)
wilson_hi = (p_hat + z**2/(2*overall_n) + z*np.sqrt(p_hat*(1-p_hat)/overall_n + z**2/(4*overall_n**2))) / (1 + z**2/overall_n)
print(f"95% CI (Wilson): [{wilson_lo*100:.1f}%, {wilson_hi*100:.1f}%]")

# ── Save LOOCV results ────────────────────────────────────────────────────────
results_df.drop(columns=["weights"]).to_csv(OUT / "loocv_results.csv", index=False)

summary = {
    "velocity_kmh":   DEFAULT_VEL,
    "decay_h":        DEFAULT_DECAY,
    "overall_top10_pct": round(overall_rate, 1),
    "overall_n":      int(overall_n),
    "overall_k":      int(overall_k),
    "mean_rank":      round(mean_rank, 2),
    "median_rank":    med_rank,
    "p_value":        round(float(binom_p), 4),
    "ci_lo_pct":      round(wilson_lo * 100, 1),
    "ci_hi_pct":      round(wilson_hi * 100, 1),
    "station_11447650_top10_pct": round(results_df[results_df["station"]==11447650]["top10"].mean()*100, 1),
    "station_11447890_top10_pct": round(results_df[results_df["station"]==11447890]["top10"].mean()*100, 1),
    "n_11447650": int((results_df["station"]==11447650).sum()),
    "n_11447890": int((results_df["station"]==11447890).sum()),
}
with open(OUT / "loocv_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── Sensitivity grid ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SENSITIVITY ANALYSIS: velocity × decay")
print("=" * 70)

# Use the optimal weights from the default run as fixed weights for sensitivity
default_weights = np.vstack(all_weights).mean(0)
print(f"Fixed weights for sensitivity: {default_weights.round(3)}")

velocities  = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
decay_vals  = [3.0, 4.5, 6.0, 9.0, 12.0]

sens_records = []
print(f"\n{'Velocity':>10} {'Decay':>7}  {'Overall':>9}  {'S650':>7}  {'S890':>7}")
print("-" * 50)

for vel in velocities:
    for decay in decay_vals:
        # Recompute prop scores with these parameters
        tmp = data.copy()
        tmp = add_exp_propagation(tmp, vel, decay)

        # Use fixed default weights (no re-optimisation for sensitivity)
        w_d, w_i, w_a, w_p = default_weights
        all_ev_res = []
        for st in [11447650, 11447890]:
            st_df = tmp[tmp["affected_station"] == st].copy()
            events_st = st_df[st_df["REGISTRY_ID"]==TRI_REGISTRY]["event_id"].unique()
            for ev in events_st:
                test = st_df[st_df["event_id"] == ev].copy()
                ranked = score_and_rank(test, w_d, w_i, w_a, w_p)
                rank = ranked[ranked["REGISTRY_ID"]==TRI_REGISTRY]["rank"].min()
                all_ev_res.append({"station": st, "top10": int(rank <= 10)})

        r = pd.DataFrame(all_ev_res)
        overall  = r["top10"].mean() * 100
        s650 = r[r["station"]==11447650]["top10"].mean() * 100
        s890 = r[r["station"]==11447890]["top10"].mean() * 100
        is_default = (vel == DEFAULT_VEL and decay == DEFAULT_DECAY)
        mark = " <-- default" if is_default else ""
        print(f"{vel:>10.1f} {decay:>7.1f}  {overall:>8.1f}%  {s650:>6.1f}%  {s890:>6.1f}%{mark}")

        sens_records.append({
            "velocity_kmh": vel, "decay_h": decay,
            "overall_pct": round(overall, 1),
            "s11447650_pct": round(s650, 1),
            "s11447890_pct": round(s890, 1),
            "is_default": is_default,
        })

sens_df = pd.DataFrame(sens_records)
sens_df.to_csv(OUT / "sensitivity_grid.csv", index=False)

print(f"\nOverall range: {sens_df['overall_pct'].min():.1f}% – {sens_df['overall_pct'].max():.1f}%")
print(f"S11447650 range: {sens_df['s11447650_pct'].min():.1f}% – {sens_df['s11447650_pct'].max():.1f}%")
print(f"S11447890 range: {sens_df['s11447890_pct'].min():.1f}% – {sens_df['s11447890_pct'].max():.1f}%")

print(f"\nResults saved to: {OUT}/")
print("  loocv_results.csv")
print("  loocv_summary.json")
print("  sensitivity_grid.csv")
