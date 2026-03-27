"""
EchoWater second-case feasibility analysis.

Goal:
- Use existing Sacramento/Delta USGS stations already present in the repo
- Check whether EchoWater temperature-related DMR violation windows align with
  observed temperature anomalies in downstream stations
- Produce a small results table and summary for manuscript use
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/echowater_feasibility")
OUT.mkdir(parents=True, exist_ok=True)

REGISTRY_ID = "110000517432"
STATIONS = [11447650, 11447890]
WINDOW_DAYS = 7


def load_dmr_windows():
    base = Path("dataset/new data")
    viol = pd.read_csv(base / "echo_effluent_violations" / "extracted" / "CA_NPDES_EFF_VIOLATIONS.csv", dtype=str)
    links = pd.read_csv(base / "frs" / "extracted" / "FRS_PROGRAM_LINKS.csv", dtype=str)

    npdes_ids = links[
        (links["REGISTRY_ID"] == REGISTRY_ID)
        & (links["PGM_SYS_ACRNM"].str.contains("NPDES", case=False, na=False))
    ]["PGM_SYS_ID"].dropna().unique().tolist()

    sub = viol[viol["NPDES_ID"].isin(npdes_ids)].copy()
    sub["MONITORING_PERIOD_END_DATE"] = pd.to_datetime(sub["MONITORING_PERIOD_END_DATE"], errors="coerce")
    sub = sub[sub["MONITORING_PERIOD_END_DATE"].dt.year.between(2019, 2025, inclusive="both")]
    sub = sub[sub["PARAMETER_DESC"].fillna("").str.contains("Temperature", case=False, regex=True)].copy()

    windows = (
        sub[["NPDES_ID", "PARAMETER_DESC", "VIOLATION_DESC", "MONITORING_PERIOD_END_DATE", "VALUE_RECEIVED_DATE"]]
        .drop_duplicates()
        .sort_values("MONITORING_PERIOD_END_DATE")
        .reset_index(drop=True)
    )
    return windows


def load_usgs_series():
    df = pd.read_csv(
        "dataset/processed/qc/usgs_realtime_qc.csv",
        usecols=["site_no", "datetime", "00010"],
        parse_dates=["datetime"],
    )
    # Normalize timezone-aware timestamps to naive UTC-like timestamps for comparisons
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df[df["site_no"].isin(STATIONS)].copy()
    return df


def nearest_anomaly(series_df, target_date):
    if series_df.empty:
        return None

    target = pd.Timestamp(target_date)
    win = series_df[
        (series_df["datetime"] >= target - pd.Timedelta(days=WINDOW_DAYS))
        & (series_df["datetime"] <= target + pd.Timedelta(days=WINDOW_DAYS))
    ].copy()
    if win.empty:
        return None

    # Simple anomaly proxy: deviation from rolling 7-day median on 15-min temp series
    # 7 days * 24 h * 4 obs/h = 672 observations
    win = win.sort_values("datetime").copy()
    win["rolling_med"] = win["00010"].rolling(window=672, center=True, min_periods=96).median()
    win["abs_dev"] = (win["00010"] - win["rolling_med"]).abs()

    # Find strongest deviation in the +/- 7 day window
    idx = win["abs_dev"].idxmax()
    row = win.loc[idx]
    return {
        "peak_datetime": str(row["datetime"]),
        "peak_temp": None if pd.isna(row["00010"]) else float(row["00010"]),
        "rolling_median": None if pd.isna(row["rolling_med"]) else float(row["rolling_med"]),
        "abs_deviation": None if pd.isna(row["abs_dev"]) else float(row["abs_dev"]),
        "days_offset": round((pd.Timestamp(row["datetime"]) - target).total_seconds() / 86400, 2),
        "n_points_in_window": int(len(win)),
    }


def main():
    print("=" * 70)
    print("ECHOWATER FEASIBILITY ANALYSIS")
    print("=" * 70)

    windows = load_dmr_windows()
    usgs = load_usgs_series()

    print("\nEchoWater temperature-related DMR windows:")
    print(windows.to_string(index=False))

    results = []
    for _, w in windows.iterrows():
        target_date = pd.Timestamp(w["MONITORING_PERIOD_END_DATE"])
        for station in STATIONS:
            s = usgs[usgs["site_no"] == station].copy()
            hit = nearest_anomaly(s, target_date)
            results.append({
                "dmr_date": str(target_date.date()),
                "station": int(station),
                "parameter": w["PARAMETER_DESC"],
                "violation_desc": w["VIOLATION_DESC"],
                **({} if hit is None else hit),
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT / "echowater_temperature_alignment.csv", index=False)

    print("\nAlignment summary:")
    print(rdf.to_string(index=False))

    # Success heuristic: anomaly peak within 3 days and deviation >= 1.0 deg F
    rdf["usable_alignment"] = (
        rdf["days_offset"].abs().fillna(999) <= 3
    ) & (rdf["abs_deviation"].fillna(0) >= 1.0)

    summary = {
        "n_dmr_windows": int(len(windows)),
        "stations_checked": STATIONS,
        "window_days": WINDOW_DAYS,
        "n_usable_alignments": int(rdf["usable_alignment"].sum()),
        "candidate_case_feasible": bool(rdf["usable_alignment"].any()),
        "best_alignment": None,
    }

    if rdf["abs_deviation"].notna().any():
        best = rdf.sort_values(["usable_alignment", "abs_deviation"], ascending=[False, False]).iloc[0].to_dict()
        summary["best_alignment"] = best

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
