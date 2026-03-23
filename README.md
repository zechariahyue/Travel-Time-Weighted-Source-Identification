# Reproducibility Package

**Paper:** "Travel-Time Weighted Contamination Source Identification in the Sacramento River Watershed: A Single-Source Proof-of-Concept Study"
**Journal:** Water Research (under review)
**Repository:** https://github.com/zechariahyue/Travel-Time-Weighted-Source-Identification

---

## Contents

```
reproducibility/
  scripts/                        ← core analysis scripts (Python 3.11)
    65_loocv_exponential_propagation.py     ← primary LOOCV + sensitivity
    66_circular_validation_and_correlation.py
    67_figure1_cv_performance_updated.py    ← Figure 1
    68_figure2_rank_distribution.py         ← Figure 2
    69_figure3_propagation_comparison.py    ← Figure 3
    74_revision_analyses.py                 ← distance-only baseline, worked example
    75_nearfield_source_simulation.py       ← near-field source simulation
    76_fair_binary_vs_exponential.py        ← fair binary vs exponential comparison
    77_sensitivity_misattribution_topk.py   ← sigma_d sweep, mis-attribution, top-k
  sample_data/
    rankings_ampac_events.csv     ← 7,832 rows: all 44 AMPAC events × 178 records
  README.md                       ← this file
```

---

## Data

### Input data (included)
`sample_data/rankings_ampac_events.csv` contains all facility-permit records for the 44 confirmed AMPAC events used in the manuscript. Key columns:

| Column | Description |
|--------|-------------|
| `REGISTRY_ID` | EPA Facility Registry ID (AMPAC = 110028001187) |
| `event_id` | Unique event identifier |
| `affected_station` | USGS station ID (11447650 or 11447890) |
| `event_parameter` | Sensor parameter (pH, dissolved_oxygen, temperature) |
| `distance_to_station_km` | Euclidean distance from facility to station |
| `distance_score_v2` | Normalised inverse-linear distance score |
| `industry_score_v2` | Binary NAICS-match score |
| `anomaly_score_v2` | Inverted proximity-weighted anomaly score (σ_d = 20 km) |
| `anomaly_mean_value` | Mean sensor reading during event window |
| `event_start` | Event onset timestamp |

### Primary data sources
All source data are publicly accessible:
- **USGS NWIS** (pH time series): https://waterdata.usgs.gov — stations 11447650, 11447890
- **EPA FRS** (facility geocodes): https://www.epa.gov/frs
- **EPA ECHO** (NPDES permits): https://echo.epa.gov/tools/data-downloads
- **EPA TRI** (industry classification): https://www.epa.gov/toxics-release-inventory-tri-program

---

## Dependencies

```
Python 3.11
pandas >= 2.1
numpy >= 1.26
scipy >= 1.12
matplotlib >= 3.8
```

Install: `pip install pandas numpy scipy matplotlib`

---

## How to Reproduce Each Manuscript Result

All scripts expect to be run from the **project root** (the directory containing `results/`). The sample data must be placed at:
```
results/component_redesign/rankings_with_redesigned_components.csv
```
(copy `sample_data/rankings_ampac_events.csv` there, or use the full dataset from the repository).

### Table 2 & Figure 1 — Primary LOOCV results (72.7% top-10)
```bash
python scripts/65_loocv_exponential_propagation.py
# outputs: results/exponential_propagation/loocv_results.csv
#          results/exponential_propagation/loocv_summary.json
#          results/exponential_propagation/sensitivity_grid.csv

python scripts/67_figure1_cv_performance_updated.py
# outputs: manuscript/Figure1_CV_Performance.png
```

### Figure 2 — Bimodal rank distribution
```bash
python scripts/68_figure2_rank_distribution.py
# outputs: manuscript/Figure2_Rank_Distribution.png
```

### Table 4 & Figure 3 — Propagation model comparison
```bash
python scripts/74_revision_analyses.py   # distance-only baseline
python scripts/76_fair_binary_vs_exponential.py  # fair comparison
python scripts/69_figure3_propagation_comparison.py
# outputs: manuscript/Figure3_Propagation_Comparison.png
#          results/fair_binary_comparison/
```

### Figure (Decision Boundary) — Score-space bimodal mechanism
```bash
python scripts/66_circular_validation_and_correlation.py
```

### Near-field source simulation (Section 3.6)
```bash
python scripts/75_nearfield_source_simulation.py
# outputs: results/nearfield_simulation/
```

### Sensitivity analyses — σ_d sweep, mis-attribution, top-k (Section 3.5)
```bash
python scripts/77_sensitivity_misattribution_topk.py
# outputs: results/sensitivity_misattribution/
```

### Figure S5 — Sensitivity heatmap (Supplementary)
```bash
python scripts/65_loocv_exponential_propagation.py   # also generates sensitivity_grid.csv
# heatmap generated in: scripts/71_supplementary_figures_s1_s2.py
```

---

## Ground Truth

AMPAC Fine Chemicals (EPA TRI Registry ID: 110028001187; NAICS 325199) is the sole confirmed contamination source. Attribution is based on:
- NPDES Discharge Monitoring Reports (DMRs)
- EPA ECHO enforcement actions
- Direct communication with the California Regional Water Quality Control Board, Central Valley Region

Per-event evidentiary detail was not available for all 44 events; see manuscript Section 4.4 for the mis-attribution sensitivity analysis.

---

## Confirmed Source Location

| Parameter | Value |
|-----------|-------|
| Facility | AMPAC Fine Chemicals |
| Coordinates | 38.614°N, 121.216°W |
| Distance to Station 11447650 | 28.7 km (Euclidean) |
| Distance to Station 11447890 | 43.4 km (Euclidean) |
| NAICS code | 325199 (All Other Basic Organic Chemical Manufacturing) |

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| LOOCV top-10 success rate (overall) | 72.7% (32/44) |
| 95% CI (Wilson) | 58.2–83.7% |
| p-value vs 5.6% base-rate null | < 0.0001 |
| p-value vs 50% null | 0.0018 |
| Optimal weights [d, i, a, p] | [0.00, 0.00, 0.60, 0.40] |
| Weight variance across 44 folds | 0 (all folds identical) |
| Station 11447650 top-10 rate | 73.9% (17/23) |
| Station 11447890 top-10 rate | 71.4% (15/21) |
| Distance-only baseline top-10 rate | 0.0% (mean rank 144.5) |
