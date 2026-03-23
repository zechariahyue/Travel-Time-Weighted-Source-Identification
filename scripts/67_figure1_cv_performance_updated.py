"""
Figure 1 (Updated): Cross-Validation Performance by Station
Exponential propagation results from script 65.

Reads loocv_summary.json for primary numbers.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

# ── Load data ─────────────────────────────────────────────────────────────────
with open("results/exponential_propagation/loocv_summary.json") as f:
    s = json.load(f)

# Wilson 95% CI helper
def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return centre - margin, centre + margin

# Per-station counts
k_650 = round(s["station_11447650_top10_pct"] / 100 * s["n_11447650"])
k_890 = round(s["station_11447890_top10_pct"] / 100 * s["n_11447890"])
n_650 = s["n_11447650"]
n_890 = s["n_11447890"]

rates = [
    s["station_11447650_top10_pct"] / 100,
    s["station_11447890_top10_pct"] / 100,
    s["overall_top10_pct"] / 100,
]
counts  = [k_650,          k_890,          s["overall_k"]]
totals  = [n_650,          n_890,          s["overall_n"]]

ci_lo_650, ci_hi_650 = wilson_ci(k_650, n_650)
ci_lo_890, ci_hi_890 = wilson_ci(k_890, n_890)

ci_lo = [ci_lo_650,             ci_lo_890,             s["ci_lo_pct"]/100]
ci_hi = [ci_hi_650,             ci_hi_890,             s["ci_hi_pct"]/100]

labels = ['Station\n11447650', 'Station\n11447890', 'Overall']

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['#2ecc71', '#3498db', '#9b59b6']
x = np.arange(len(labels))
bars = ax.bar(x, rates, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

# CI error bars
errs_lo = [rates[i] - ci_lo[i] for i in range(3)]
errs_hi = [ci_hi[i] - rates[i] for i in range(3)]
ax.errorbar(x, rates, yerr=[errs_lo, errs_hi], fmt='none',
            ecolor='black', capsize=8, capthick=2, linewidth=2)

# Null baseline
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2,
           label='Null baseline (50%)', zorder=0)

# Labels on bars
for i, (bar, rate, k, n) in enumerate(zip(bars, rates, counts, totals)):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{rate:.1%}\n({k}/{n})',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Statistical annotation
p_val = s["p_value"]
ci_lo_pct = s["ci_lo_pct"]
ci_hi_pct = s["ci_hi_pct"]
ax.text(0.02, 0.02,
        f'Overall: p={p_val:.4f} (vs 50% null)\n'
        f'95% CI: [{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%]',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35))

# Formatting
ax.set_ylabel('Top-10 Success Rate', fontweight='bold')
ax.set_title('Cross-Validated Source Identification Performance (LOOCV)\n'
             'Exponential Travel-Time Propagation  |  n = 178 candidate facilities',
             fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 1.18])
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
Path("results/publication_figures").mkdir(parents=True, exist_ok=True)
plt.savefig("results/publication_figures/Figure1_CV_Performance.png",
            dpi=300, bbox_inches='tight')
plt.savefig("results/publication_figures/Figure1_CV_Performance.pdf",
            bbox_inches='tight')
plt.close()

print("[OK] Figure 1 saved (updated with exponential propagation results):")
print(f"     Overall: {s['overall_top10_pct']:.1f}% ({s['overall_k']}/{s['overall_n']})")
print(f"     Station 11447650: {s['station_11447650_top10_pct']:.1f}%")
print(f"     Station 11447890: {s['station_11447890_top10_pct']:.1f}%")
print("     PNG: results/publication_figures/Figure1_CV_Performance.png")
print("     PDF: results/publication_figures/Figure1_CV_Performance.pdf")
