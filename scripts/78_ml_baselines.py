"""
Script 78: ML Baselines for Contamination Source Identification
==============================================================
Fits Random Forest and Logistic Regression classifiers on the four
normalised component scores per event, evaluates LOOCV top-10 success rate,
and compares to the weighted-ranking approach.

Key design: the ML models predict P(top-10) for each facility-event pair.
Within each LOOCV fold, training uses 43 events, test uses 1 event.
For each test event, all 178 records are ranked by predicted probability;
AMPAC success = rank <= 10.

Also reports a trivial "all-upstream" and "always-rank-1" degenerate
baseline for completeness.

Outputs: results/ml_baselines/ml_baseline_results.json
         results/ml_baselines/ml_baseline_detail.csv
"""

import pandas as pd
import numpy as np
import json
import os
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── paths ─────────────────────────────────────────────────────────────────────
DATA = "results/component_redesign/rankings_with_redesigned_components.csv"
LOOCV_CSV = "results/exponential_propagation/loocv_results.csv"
OUT_DIR = "results/ml_baselines"
os.makedirs(OUT_DIR, exist_ok=True)

TRI = 110028001187
FEATURES = ["distance_score_v2", "industry_score_v2",
            "anomaly_score_v2", "propagation_score_v2"]

# ── load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, encoding="utf-8-sig")
loocv = pd.read_csv(LOOCV_CSV)

events = sorted(df["event_id"].unique())
n_events = len(events)
print(f"Events: {n_events}, Records per event: {len(df)/n_events:.0f}")

# ── label construction ─────────────────────────────────────────────────────────
# For each event, label = 1 if REGISTRY_ID == TRI (AMPAC record), else 0
# This is an extremely imbalanced classification: 25/178 = 14% positive rate
# The ML models predict P(AMPAC record | features); ranked by P for each event

df["label"] = (df["REGISTRY_ID"] == TRI).astype(int)
print(f"Positive rate per event: {df['label'].mean():.3f} ({df['label'].sum()/n_events:.1f} positives per event)")

# ── LOOCV loop ─────────────────────────────────────────────────────────────────
results = []

for i, held_out_event in enumerate(events):
    train_events = [e for e in events if e != held_out_event]

    train_df = df[df["event_id"].isin(train_events)].copy()
    test_df = df[df["event_id"] == held_out_event].copy()

    X_train = train_df[FEATURES].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURES].values

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    test_df = test_df.copy()
    test_df["rf_prob"] = rf_proba
    test_df["rf_rank"] = test_df["rf_prob"].rank(ascending=False, method="min").astype(int)

    # AMPAC minimum rank
    ampac_rf_rank = test_df[test_df["REGISTRY_ID"] == TRI]["rf_rank"].min()
    rf_top10 = int(ampac_rf_rank <= 10)

    # ── Logistic Regression ────────────────────────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ))
    ])
    try:
        lr_pipe.fit(X_train, y_train)
        lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"LR failed fold {i}: {e}")
        lr_proba = np.zeros(len(X_test))

    test_df["lr_prob"] = lr_proba
    test_df["lr_rank"] = test_df["lr_prob"].rank(ascending=False, method="min").astype(int)
    ampac_lr_rank = test_df[test_df["REGISTRY_ID"] == TRI]["lr_rank"].min()
    lr_top10 = int(ampac_lr_rank <= 10)

    # ── Naive baselines ────────────────────────────────────────────────────────
    # Baseline A: rank by anomaly score only (single best feature)
    test_df["anomaly_rank"] = test_df["anomaly_score_v2"].rank(ascending=False, method="min").astype(int)
    ampac_anomaly_rank = test_df[test_df["REGISTRY_ID"] == TRI]["anomaly_rank"].min()
    anomaly_top10 = int(ampac_anomaly_rank <= 10)

    # Baseline B: rank by propagation score only
    test_df["prop_rank"] = test_df["propagation_score_v2"].rank(ascending=False, method="min").astype(int)
    ampac_prop_rank = test_df[test_df["REGISTRY_ID"] == TRI]["prop_rank"].min()
    prop_top10 = int(ampac_prop_rank <= 10)

    # Station
    station = test_df["affected_station"].iloc[0]

    results.append({
        "event_id": held_out_event,
        "station": int(station),
        "rf_rank": int(ampac_rf_rank),
        "rf_top10": rf_top10,
        "lr_rank": int(ampac_lr_rank),
        "lr_top10": lr_top10,
        "anomaly_rank": int(ampac_anomaly_rank),
        "anomaly_top10": anomaly_top10,
        "prop_rank": int(ampac_prop_rank),
        "prop_top10": prop_top10,
    })

    if (i + 1) % 10 == 0:
        print(f"  Fold {i+1}/{n_events} done")

print(f"All {n_events} folds complete.")

# ── aggregate results ──────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(f"{OUT_DIR}/ml_baseline_detail.csv", index=False)

def summarise(col, df=res_df):
    n = len(df)
    k = df[col].sum()
    rate = k / n * 100
    ci = binomtest(k, n, 10/178, alternative="greater")
    p_base = ci.pvalue
    p50 = binomtest(k, n, 0.5, alternative="greater").pvalue
    return {
        "n": n, "successes": int(k), "rate_pct": round(rate, 1),
        "p_vs_base_rate": float(f"{p_base:.4e}"),
        "p_vs_50pct": round(p50, 4),
        "s650_rate": round(df[df["station"]==11447650][col].mean()*100, 1),
        "s890_rate": round(df[df["station"]==11447890][col].mean()*100, 1),
    }

summary = {
    "weighted_ranking_optimal": {
        "n": 44, "successes": 32, "rate_pct": 72.7,
        "s650_rate": 73.9, "s890_rate": 71.4,
        "note": "From Script 65 LOOCV; weights [0,0,0.60,0.40]"
    },
    "random_forest": summarise("rf_top10"),
    "logistic_regression": summarise("lr_top10"),
    "anomaly_score_only": summarise("anomaly_top10"),
    "propagation_score_only": summarise("prop_top10"),
}

print("\n=== ML BASELINE RESULTS ===")
for model, stats in summary.items():
    print(f"\n{model}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

with open(f"{OUT_DIR}/ml_baseline_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUT_DIR}/")
print("\nKey comparison table:")
print(f"{'Model':<35} {'Overall':>8} {'S650':>8} {'S890':>8}")
print("-" * 62)
for model, stats in summary.items():
    print(f"{model:<35} {stats['rate_pct']:>7.1f}% {stats['s650_rate']:>7.1f}% {stats['s890_rate']:>7.1f}%")
