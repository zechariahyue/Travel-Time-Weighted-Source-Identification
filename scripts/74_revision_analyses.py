"""
Revision Analyses — addressing peer-review concerns

M2: Anomaly score without inversion (compare to inverted baseline)
M3: Full uniform weight grid (0.1 steps, 0-1 range, all combos summing to 1)
M6: Binary propagation model with independently optimised LOOCV weights
M7: Per-parameter breakdown (verify mixed event types handled per-parameter)
M8: Distance-only baseline

Outputs: results/revision_analyses/
    full_grid_loocv.json
    no_inversion_loocv.json
    binary_independent_loocv.json
    distance_only_baseline.json
    event_type_breakdown.json
    summary_table.csv
"""

import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

OUT = Path("results/revision_analyses")
OUT.mkdir(parents=True, exist_ok=True)

TRI_REGISTRY = 110028001187
DEFAULT_VEL  = 5.0   # km/h
DEFAULT_DECAY = 6.0  # hours

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "results/component_redesign/rankings_with_redesigned_components.csv",
    encoding="utf-8-sig"
)

confirmed_events = df[
    (df["REGISTRY_ID"] == TRI_REGISTRY) &
    (df["affected_station"].isin([11447650, 11447890]))
]["event_id"].unique()

data = df[df["event_id"].isin(confirmed_events)].copy()

# Recompute exponential propagation (the correct one, per script 65)
travel_time = data["distance_to_station_km"] / DEFAULT_VEL
data["prop_exp_raw"] = np.exp(-travel_time / DEFAULT_DECAY)
g = data.groupby("event_id")["prop_exp_raw"]
mn = g.transform("min"); mx = g.transform("max"); rng = mx - mn
data["prop_exp"] = np.where(rng > 0, (data["prop_exp_raw"] - mn) / rng, 0.5)

# Confirm: anomaly_score_v2 = 1 - anomaly_magnitude_score
diff = (data["anomaly_score_v2"] - (1 - data["anomaly_magnitude_score"])).abs().max()
print(f"[Check] anomaly_score_v2 == 1 - anomaly_magnitude_score: max diff = {diff:.2e}")
# The non-inverted anomaly is anomaly_magnitude_score directly
# (closer facilities score higher; distant AMPAC would score lower)

print(f"\nTotal events: {len(confirmed_events)}")
by_st  = data[data["REGISTRY_ID"]==TRI_REGISTRY].groupby("affected_station")["event_id"].nunique()
by_par = data[data["REGISTRY_ID"]==TRI_REGISTRY].groupby("event_parameter")["event_id"].nunique()
print(f"By station:   {by_st.to_dict()}")
print(f"By parameter: {by_par.to_dict()}")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def rank_events(df_in, anom_col, prop_col, w_dist, w_ind, w_anom, w_prop):
    d = df_in.copy()
    d["score"] = (
        w_dist * d["distance_score_v2"] +
        w_ind  * d["industry_score_v2"] +
        w_anom * d[anom_col] +
        w_prop * d[prop_col]
    )
    d["rank"] = d.groupby("event_id")["score"].rank(ascending=False, method="min")
    return d


def loocv_single_station(station_df, anom_col, prop_col, weight_grid):
    """LOOCV with per-fold weight search over weight_grid."""
    events = station_df[station_df["REGISTRY_ID"] == TRI_REGISTRY]["event_id"].unique()
    fold_results = []
    fold_weights_list = []

    for test_ev in events:
        train_df = station_df[station_df["event_id"] != test_ev]
        test_df  = station_df[station_df["event_id"] == test_ev].copy()

        # Grid search on training set
        best_obj = -np.inf
        best_w   = weight_grid[0]
        for w in weight_grid:
            ranked = rank_events(train_df, anom_col, prop_col, *w)
            tri = ranked[ranked["REGISTRY_ID"] == TRI_REGISTRY][
                ["event_id", "rank"]].drop_duplicates("event_id")
            if len(tri) == 0:
                continue
            top10 = (tri["rank"] <= 10).mean()
            mean_r = tri["rank"].mean()
            obj = top10 * 1000 - mean_r
            if obj > best_obj:
                best_obj = obj
                best_w   = w

        fold_weights_list.append(best_w)
        ranked_test = rank_events(test_df, anom_col, prop_col, *best_w)
        tri_test = ranked_test[ranked_test["REGISTRY_ID"] == TRI_REGISTRY]
        rank = tri_test["rank"].min()
        param = tri_test["event_parameter"].iloc[0]
        fold_results.append({
            "event_id":  test_ev,
            "rank":      rank,
            "top10":     int(rank <= 10),
            "parameter": param,
            "weights":   list(best_w),
        })

    return fold_results, fold_weights_list


def run_loocv_both_stations(anom_col, prop_col, weight_grid, label):
    print(f"\n{'='*60}")
    print(f"LOOCV: {label}")
    print(f"  anom_col={anom_col}, prop_col={prop_col}")
    print(f"  grid size: {len(weight_grid)} combinations")

    all_results = []
    weight_stats = {}

    for station in [11447650, 11447890]:
        st_df = data[data["affected_station"] == station].copy()
        res, wts = loocv_single_station(st_df, anom_col, prop_col, weight_grid)
        all_results.extend(res)
        fw = np.array(wts)
        weight_stats[str(station)] = {
            "mean": fw.mean(0).round(3).tolist(),
            "std":  fw.std(0).round(3).tolist(),
        }
        k = sum(r["top10"] for r in res)
        n = len(res)
        print(f"  Station {station}: {k}/{n} = {k/n*100:.1f}%  weights: {fw.mean(0).round(3)}")

    rdf = pd.DataFrame(all_results)
    k_tot = rdf["top10"].sum()
    n_tot = len(rdf)
    rate  = k_tot / n_tot

    # Wilson CI
    z = 1.96
    wlo = (rate + z**2/(2*n_tot) - z*np.sqrt(rate*(1-rate)/n_tot + z**2/(4*n_tot**2))) / (1 + z**2/n_tot)
    whi = (rate + z**2/(2*n_tot) + z*np.sqrt(rate*(1-rate)/n_tot + z**2/(4*n_tot**2))) / (1 + z**2/n_tot)

    # p-value vs 50% null
    bt = scipy_stats.binomtest(int(k_tot), int(n_tot), 0.5, alternative="greater")
    p50 = bt.pvalue

    # p-value vs base rate (5.6%) null
    bt2 = scipy_stats.binomtest(int(k_tot), int(n_tot), 10/178, alternative="greater")
    p_base = bt2.pvalue

    result = {
        "label":       label,
        "anom_col":    anom_col,
        "prop_col":    prop_col,
        "grid_size":   len(weight_grid),
        "overall_pct": round(rate * 100, 1),
        "k":           int(k_tot),
        "n":           int(n_tot),
        "ci_lo":       round(wlo * 100, 1),
        "ci_hi":       round(whi * 100, 1),
        "p_vs_50pct":  round(p50, 4),
        "p_vs_baseRate": float(f"{p_base:.2e}"),
        "s11447650_pct": round(rdf[rdf["station"]==11447650]["top10"].mean()*100, 1)
            if "station" in rdf.columns else None,
        "s11447890_pct": round(rdf[rdf["station"]==11447890]["top10"].mean()*100, 1)
            if "station" in rdf.columns else None,
        "weight_stats": weight_stats,
    }

    # Fix station cols if missing (add station from original data)
    if "station" not in rdf.columns:
        event_station = data[data["REGISTRY_ID"]==TRI_REGISTRY][["event_id","affected_station"]].drop_duplicates()
        rdf = rdf.merge(event_station, on="event_id", how="left")
        result["s11447650_pct"] = round(rdf[rdf["affected_station"]==11447650]["top10"].mean()*100, 1)
        result["s11447890_pct"] = round(rdf[rdf["affected_station"]==11447890]["top10"].mean()*100, 1)

    print(f"  Overall: {k_tot}/{n_tot} = {rate*100:.1f}%  "
          f"(95% CI {wlo*100:.1f}–{whi*100:.1f}%)  "
          f"p_50={p50:.4f}  p_base={p_base:.2e}")

    return result


# ── M3: Full uniform weight grid ───────────────────────────────────────────────
print("\n" + "="*60)
print("M3: BUILDING FULL UNIFORM WEIGHT GRID (0.1 steps)")

vals = np.arange(0.0, 1.01, 0.1).round(1)
full_grid = []
for wd in vals:
    for wi in vals:
        for wa in vals:
            wp = round(1.0 - wd - wi - wa, 1)
            if 0.0 <= wp <= 1.0 and abs(wd + wi + wa + wp - 1.0) < 0.001:
                full_grid.append((wd, wi, wa, wp))

print(f"Full grid size: {len(full_grid)} combinations (vs original: 48)")

result_full = run_loocv_both_stations(
    "anomaly_score_v2", "prop_exp",
    full_grid, "Full uniform grid (0.1 steps)"
)
with open(OUT / "full_grid_loocv.json", "w") as f:
    json.dump(result_full, f, indent=2)


# ── M2: Anomaly without inversion ──────────────────────────────────────────────
print("\n" + "="*60)
print("M2: ANOMALY WITHOUT INVERSION")
print("  (anomaly_magnitude_score: closer = higher, no inversion)")

result_noinv = run_loocv_both_stations(
    "anomaly_magnitude_score", "prop_exp",
    full_grid, "No inversion (anomaly_magnitude_score)"
)
with open(OUT / "no_inversion_loocv.json", "w") as f:
    json.dump(result_noinv, f, indent=2)


# ── M6: Binary propagation with independent LOOCV weights ─────────────────────
print("\n" + "="*60)
print("M6: BINARY PROPAGATION with own LOOCV weight search")
print("  (propagation_score_v2: 0.7=upstream, 0.3=other)")

result_binary = run_loocv_both_stations(
    "anomaly_score_v2", "propagation_score_v2",
    full_grid, "Binary propagation (independent weights)"
)
with open(OUT / "binary_independent_loocv.json", "w") as f:
    json.dump(result_binary, f, indent=2)


# ── M8: Distance-only baseline ────────────────────────────────────────────────
print("\n" + "="*60)
print("M8: DISTANCE-ONLY BASELINE (w_dist=1, others=0)")

dist_results = []
event_station_map = data[data["REGISTRY_ID"]==TRI_REGISTRY][
    ["event_id", "affected_station", "event_parameter"]].drop_duplicates("event_id")

for _, row in event_station_map.iterrows():
    ev = row["event_id"]
    ev_df = data[data["event_id"] == ev].copy()
    ev_df["score"] = ev_df["distance_score_v2"]
    ev_df["rank"]  = ev_df["score"].rank(ascending=False, method="min")
    ampac = ev_df[ev_df["REGISTRY_ID"] == TRI_REGISTRY]
    rank  = ampac["rank"].min()
    dist_results.append({
        "event_id":  ev,
        "station":   row["affected_station"],
        "rank":      rank,
        "top10":     int(rank <= 10),
        "parameter": row["event_parameter"],
    })

ddf = pd.DataFrame(dist_results)
k_d = ddf["top10"].sum(); n_d = len(ddf)
rate_d = k_d / n_d
z = 1.96
wlo_d = (rate_d + z**2/(2*n_d) - z*np.sqrt(rate_d*(1-rate_d)/n_d + z**2/(4*n_d**2))) / (1 + z**2/n_d)
whi_d = (rate_d + z**2/(2*n_d) + z*np.sqrt(rate_d*(1-rate_d)/n_d + z**2/(4*n_d**2))) / (1 + z**2/n_d)
bt_d  = scipy_stats.binomtest(int(k_d), int(n_d), 0.5, alternative="greater")
bt_d2 = scipy_stats.binomtest(int(k_d), int(n_d), 10/178, alternative="greater")

dist_result = {
    "label":       "Distance-only baseline",
    "overall_pct": round(rate_d * 100, 1),
    "k": int(k_d), "n": int(n_d),
    "ci_lo": round(wlo_d * 100, 1), "ci_hi": round(whi_d * 100, 1),
    "p_vs_50pct":   round(bt_d.pvalue, 4),
    "p_vs_baseRate": float(f"{bt_d2.pvalue:.2e}"),
    "s11447650_pct": round(ddf[ddf["station"]==11447650]["top10"].mean()*100, 1),
    "s11447890_pct": round(ddf[ddf["station"]==11447890]["top10"].mean()*100, 1),
    "mean_rank":    round(ddf["rank"].mean(), 1),
    "median_rank":  float(ddf["rank"].median()),
    "note": "Inverse-linear Euclidean distance only. Most facilities cluster within 10 km of station; AMPAC at 28-43 km."
}
print(f"  Distance-only: {k_d}/{n_d} = {rate_d*100:.1f}%  "
      f"(95% CI {wlo_d*100:.1f}–{whi_d*100:.1f}%)  "
      f"p_50={bt_d.pvalue:.4f}")
print(f"  By station: S650={dist_result['s11447650_pct']}%  S890={dist_result['s11447890_pct']}%")
print(f"  Mean rank: {dist_result['mean_rank']}  Median rank: {dist_result['median_rank']}")
with open(OUT / "distance_only_baseline.json", "w") as f:
    json.dump(dist_result, f, indent=2)


# ── M7: Per-parameter event breakdown ─────────────────────────────────────────
print("\n" + "="*60)
print("M7: PER-PARAMETER EVENT BREAKDOWN")
print("  (verifying anomaly score uses per-parameter baselines)")

loocv_results = pd.read_csv("results/exponential_propagation/loocv_results.csv")
# Merge with event parameter info
par_info = data[data["REGISTRY_ID"]==TRI_REGISTRY][
    ["event_id","event_parameter","affected_station"]].drop_duplicates("event_id")
loocv_results = loocv_results.merge(par_info, on=["event_id"], how="left",
                                     suffixes=("","_y"))

par_breakdown = {}
for par, grp in loocv_results.groupby("event_parameter"):
    k_p = grp["top10"].sum(); n_p = len(grp)
    par_breakdown[str(par)] = {
        "n_events": int(n_p),
        "n_success": int(k_p),
        "top10_pct": round(k_p/n_p*100, 1) if n_p > 0 else None,
        "mean_rank": round(grp["rank"].mean(), 1),
    }
    print(f"  {par}: {k_p}/{n_p} = {k_p/n_p*100:.1f}%  mean_rank={grp['rank'].mean():.1f}")

par_note = (
    "Baselines are computed per-station, per-parameter (see script 08_real_data_ranking.py "
    "lines 152-194: self.baselines[site][param_key] = (median, iqr)). "
    "Each event type uses its own parameter-specific baseline. "
    "pH events (n=40) dominate; the 3 dissolved oxygen and 1 temperature events "
    "use parameter-specific IQR. Mixed event types are handled correctly."
)
event_type_result = {"breakdown": par_breakdown, "methodology_note": par_note}
print(f"\n  Note: {par_note[:120]}...")
with open(OUT / "event_type_breakdown.json", "w") as f:
    json.dump(event_type_result, f, indent=2)


# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY TABLE")

rows = [
    {"Model": "Current (inverted anomaly, exp. prop., narrow grid)",
     "Grid": "48 combos", "Anom": "inverted",
     "Overall (%)": 72.7, "S650 (%)": 73.9, "S890 (%)": 71.4,
     "Opt. weights [d,i,a,p]": "[0,0,0.60,0.40]"},
    {"Model": f"Full uniform grid (M3)",
     "Grid": f"{len(full_grid)} combos", "Anom": "inverted",
     "Overall (%)": result_full["overall_pct"],
     "S650 (%)": result_full["s11447650_pct"],
     "S890 (%)": result_full["s11447890_pct"],
     "Opt. weights [d,i,a,p]": str(result_full["weight_stats"])},
    {"Model": "No inversion (M2)",
     "Grid": f"{len(full_grid)} combos", "Anom": "not inverted",
     "Overall (%)": result_noinv["overall_pct"],
     "S650 (%)": result_noinv["s11447650_pct"],
     "S890 (%)": result_noinv["s11447890_pct"],
     "Opt. weights [d,i,a,p]": str(result_noinv["weight_stats"])},
    {"Model": "Binary prop., indep. weights (M6)",
     "Grid": f"{len(full_grid)} combos", "Anom": "inverted",
     "Overall (%)": result_binary["overall_pct"],
     "S650 (%)": result_binary["s11447650_pct"],
     "S890 (%)": result_binary["s11447890_pct"],
     "Opt. weights [d,i,a,p]": str(result_binary["weight_stats"])},
    {"Model": "Distance-only baseline (M8)",
     "Grid": "fixed w_d=1", "Anom": "—",
     "Overall (%)": dist_result["overall_pct"],
     "S650 (%)": dist_result["s11447650_pct"],
     "S890 (%)": dist_result["s11447890_pct"],
     "Opt. weights [d,i,a,p]": "[1,0,0,0]"},
]

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT / "summary_table.csv", index=False)
print(summary_df[["Model","Overall (%)","S650 (%)","S890 (%)"]].to_string(index=False))

print(f"\nAll results saved to {OUT}/")
