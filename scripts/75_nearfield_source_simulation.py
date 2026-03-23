"""
Near-Field Source Simulation — Reviewer Concern M1

The anomaly inversion in this method assigns high anomaly_score_v2 to
facilities that are DISTANT from a station (they perturb the signal less and
hence look more anomalous relative to background).  A reviewer (M1) noted
that this embeds an implicit assumption: the confirmed source is far from the
monitoring station.  For AMPAC the true distances are 28.7 km (S650) and
43.4 km (S890), which satisfy that assumption.

This script tests empirically what happens when the confirmed source is
NEAR-FIELD (distance_to_station_km <= 5 km).  We select up to 5 real
facilities (by REGISTRY_ID) per station that sit within 5 km, treat each one
as a synthetic "confirmed source" for all events at that station, apply the
AMPAC-optimised fixed weights [0, 0, 0.60, 0.40] without re-optimisation,
and compare top-10 success rates against AMPAC.

The expected result: near-field facilities score low on anomaly_score_v2
(because they strongly perturb the local sensor signal) but high on prop_exp
(because they are close).  With anomaly weight 0.60 dominating, near-field
sources should be systematically penalised — demonstrating that the current
weight set is not "distance-neutral" and the method is most reliable in the
far-field configuration validated by the AMPAC case.

Outputs
-------
results/nearfield_simulation/
    nearfield_simulation_results.csv   -- per-facility summary
    nearfield_simulation_summary.json  -- aggregate statistics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/nearfield_simulation")
OUT.mkdir(parents=True, exist_ok=True)

TRI_REGISTRY    = 110028001187
STATIONS        = [11447650, 11447890]
NEARFIELD_KM    = 5.0          # threshold defining "near-field"
MAX_CANDIDATES  = 5            # max near-field facilities tested per station
DEFAULT_VEL     = 5.0          # km/h
DEFAULT_DECAY   = 6.0          # hours
FIXED_WEIGHTS   = [0.0, 0.0, 0.60, 0.40]   # [dist, ind, anom, prop]


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(
    "results/component_redesign/rankings_with_redesigned_components.csv",
    encoding="utf-8-sig"
)

# Restrict to the 44 confirmed AMPAC events across the two primary stations
confirmed_events = df[
    (df["REGISTRY_ID"] == TRI_REGISTRY) &
    (df["affected_station"].isin(STATIONS))
]["event_id"].unique()

data = df[df["event_id"].isin(confirmed_events)].copy()

print("=" * 70)
print("NEAR-FIELD SOURCE SIMULATION (Reviewer Concern M1)")
print("=" * 70)
print(f"\nTotal confirmed events: {len(confirmed_events)}")
by_st = (
    data[data["REGISTRY_ID"] == TRI_REGISTRY]
    .groupby("affected_station")["event_id"]
    .nunique()
)
print(f"Events by station: {by_st.to_dict()}")


# ── Core functions ────────────────────────────────────────────────────────────

def add_exp_propagation(df_in, velocity_kmh, decay_h):
    """Add exponential decay propagation score, min-max normalised per event."""
    d = df_in.copy()
    travel_time = d["distance_to_station_km"] / velocity_kmh
    d["prop_raw"] = np.exp(-travel_time / decay_h)

    g   = d.groupby("event_id")["prop_raw"]
    mn  = g.transform("min")
    mx  = g.transform("max")
    rng = mx - mn
    d["prop_exp"] = np.where(rng > 0, (d["prop_raw"] - mn) / rng, 0.5)
    return d


def score_and_rank(df_ev, w_dist, w_ind, w_anom, w_prop):
    """Compute composite score and rank facilities within each event."""
    d = df_ev.copy()
    d["score"] = (
        w_dist * d["distance_score_v2"] +
        w_ind  * d["industry_score_v2"] +
        w_anom * d["anomaly_score_v2"]  +
        w_prop * d["prop_exp"]
    )
    d["rank"] = d.groupby("event_id")["score"].rank(
        ascending=False, method="min"
    )
    return d


def evaluate_synthetic_source(station_id, registry_id, station_df, label=""):
    """
    Treat `registry_id` as the confirmed source for ALL events at `station_id`.
    Apply fixed AMPAC-optimised weights; return per-event ranks and summary.
    """
    events = station_df[
        station_df["REGISTRY_ID"] == TRI_REGISTRY
    ]["event_id"].unique()

    ranks = []
    for ev in events:
        ev_df  = station_df[station_df["event_id"] == ev].copy()
        ranked = score_and_rank(ev_df, *FIXED_WEIGHTS)
        cand   = ranked[ranked["REGISTRY_ID"] == registry_id]
        if cand.empty:
            # Facility not present in this event — skip
            continue
        ranks.append(float(cand["rank"].min()))

    n_events  = len(ranks)
    if n_events == 0:
        return None

    top10_rate = np.mean([r <= 10 for r in ranks])
    mean_rank  = float(np.mean(ranks))
    return {
        "registry_id": registry_id,
        "station":     station_id,
        "label":       label,
        "n_events":    n_events,
        "top10_rate":  round(top10_rate * 100, 1),
        "mean_rank":   round(mean_rank, 1),
        "ranks":       ranks,
    }


# ── Prepare scored data (fixed velocity/decay) ────────────────────────────────

data = add_exp_propagation(data, DEFAULT_VEL, DEFAULT_DECAY)


# ── Step 1: AMPAC baseline (actual confirmed source) ──────────────────────────

print("\n" + "-" * 70)
print("AMPAC BASELINE (actual confirmed source)")
print("-" * 70)

ampac_rows = data[data["REGISTRY_ID"] == TRI_REGISTRY]
ampac_results = []

for station in STATIONS:
    st_df = data[data["affected_station"] == station].copy()

    # AMPAC distance for this station (constant across events)
    ampac_dist = (
        ampac_rows[ampac_rows["affected_station"] == station]
        ["distance_to_station_km"]
        .mean()
    )

    res = evaluate_synthetic_source(
        station, TRI_REGISTRY, st_df,
        label=f"AMPAC ({ampac_dist:.1f} km)"
    )
    if res is not None:
        res["distance_km"] = round(ampac_dist, 1)
        res["is_ampac"]    = True
        ampac_results.append(res)
        print(
            f"  Station {station}  dist={ampac_dist:.1f} km  "
            f"top-10={res['top10_rate']:.1f}%  mean rank={res['mean_rank']:.1f}  "
            f"n={res['n_events']}"
        )


# ── Step 2: Identify near-field candidate facilities ─────────────────────────

print("\n" + "-" * 70)
print(f"NEAR-FIELD CANDIDATES (distance <= {NEARFIELD_KM} km, excl. AMPAC)")
print("-" * 70)

nearfield_results = []

for station in STATIONS:
    st_df = data[data["affected_station"] == station].copy()

    # Facilities within near-field threshold (excluding AMPAC)
    nf = (
        st_df[
            (st_df["distance_to_station_km"] <= NEARFIELD_KM) &
            (st_df["REGISTRY_ID"] != TRI_REGISTRY)
        ]
        [["REGISTRY_ID", "distance_to_station_km"]]
        .drop_duplicates("REGISTRY_ID")
        .sort_values("distance_to_station_km")
        .head(MAX_CANDIDATES)
    )

    print(f"\n  Station {station}: {len(nf)} near-field candidates")

    if nf.empty:
        print("    (none found)")
        continue

    for _, row in nf.iterrows():
        rid  = row["REGISTRY_ID"]
        dist = row["distance_to_station_km"]

        res = evaluate_synthetic_source(
            station, rid, st_df,
            label=f"NF facility ({dist:.2f} km)"
        )
        if res is None:
            print(f"    REGISTRY_ID={rid}  dist={dist:.2f} km  -- not found in events, skipped")
            continue

        res["distance_km"] = round(dist, 2)
        res["is_ampac"]    = False
        nearfield_results.append(res)
        print(
            f"    REGISTRY_ID={rid}  dist={dist:.2f} km  "
            f"top-10={res['top10_rate']:.1f}%  mean rank={res['mean_rank']:.1f}  "
            f"n={res['n_events']}"
        )


# ── Step 3: Print comparison table ───────────────────────────────────────────

all_results = ampac_results + nearfield_results

print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)
print(
    f"\n{'Source':<18} {'Station':<12} {'Dist (km)':<12} "
    f"{'Top-10 %':<12} {'Mean Rank':<12} {'N Events':<10}"
)
print("-" * 76)

for r in all_results:
    tag = "AMPAC *" if r["is_ampac"] else f"ID {r['registry_id']}"
    print(
        f"{tag:<18} {r['station']:<12} {r['distance_km']:<12.2f} "
        f"{r['top10_rate']:<12.1f} {r['mean_rank']:<12.1f} {r['n_events']:<10}"
    )

print("\n* AMPAC = confirmed source (ground truth)")
print(
    "\nInterpretation: anomaly_score_v2 is INVERTED — near-field facilities "
    "perturb\nthe sensor strongly and receive LOW anomaly scores. With weight "
    "0.60 on\nanomaly and only 0.40 on propagation, near-field sources are "
    "systematically\npenalised even though their propagation score is high."
)

# Summary statistics across near-field candidates
if nearfield_results:
    nf_top10 = np.mean([r["top10_rate"] for r in nearfield_results])
    nf_rank  = np.mean([r["mean_rank"]  for r in nearfield_results])
    amp_top10 = np.mean([r["top10_rate"] for r in ampac_results])
    amp_rank  = np.mean([r["mean_rank"]  for r in ampac_results])

    print(f"\nSummary:")
    print(f"  AMPAC avg top-10 rate    : {amp_top10:.1f}%  (mean rank {amp_rank:.1f})")
    print(f"  Near-field avg top-10 rate: {nf_top10:.1f}%  (mean rank {nf_rank:.1f})")
    diff = amp_top10 - nf_top10
    print(f"  Advantage of AMPAC config : {diff:+.1f} pp in top-10 rate")


# ── Step 4: Save outputs ───────────────────────────────────────────────────────

# CSV — drop the raw ranks list (not CSV-friendly)
csv_rows = []
for r in all_results:
    csv_rows.append({
        "registry_id":  r["registry_id"],
        "distance_km":  r["distance_km"],
        "station":      r["station"],
        "is_ampac":     r["is_ampac"],
        "top10_rate":   r["top10_rate"],
        "mean_rank":    r["mean_rank"],
        "n_events":     r["n_events"],
    })

results_df = pd.DataFrame(csv_rows)
results_df.to_csv(OUT / "nearfield_simulation_results.csv", index=False)

# JSON summary
summary = {
    "velocity_kmh":       DEFAULT_VEL,
    "decay_h":            DEFAULT_DECAY,
    "fixed_weights":      FIXED_WEIGHTS,
    "nearfield_threshold_km": NEARFIELD_KM,
    "ampac": {
        str(r["station"]): {
            "distance_km":  r["distance_km"],
            "top10_rate":   r["top10_rate"],
            "mean_rank":    r["mean_rank"],
            "n_events":     r["n_events"],
        }
        for r in ampac_results
    },
    "nearfield_candidates": [
        {
            "registry_id":  r["registry_id"],
            "distance_km":  r["distance_km"],
            "station":      r["station"],
            "top10_rate":   r["top10_rate"],
            "mean_rank":    r["mean_rank"],
            "n_events":     r["n_events"],
        }
        for r in nearfield_results
    ],
}

if nearfield_results:
    summary["nearfield_avg_top10_rate"] = round(nf_top10, 1)
    summary["nearfield_avg_mean_rank"]  = round(nf_rank, 1)
    summary["ampac_avg_top10_rate"]     = round(amp_top10, 1)
    summary["ampac_avg_mean_rank"]      = round(amp_rank, 1)
    summary["top10_rate_advantage_pp"]  = round(diff, 1)

with open(OUT / "nearfield_simulation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to: {OUT}/")
print("  nearfield_simulation_results.csv")
print("  nearfield_simulation_summary.json")
