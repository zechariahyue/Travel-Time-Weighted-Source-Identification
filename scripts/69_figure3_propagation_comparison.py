"""
Figure 3 (Updated): Binary vs Exponential Propagation LOOCV Comparison

Compares LOOCV performance under two propagation parameterisations:
  - Binary (original, proximity-threshold model)
  - Exponential (travel-time decay model, from script 65)

Binary results are from the original LOOCV analysis (script 45/47):
  S650=100%, S890=28.6%, Overall=65.9% (29/44)

Exponential results are read from loocv_summary.json (script 65).

Also annotates that circular validation == LOOCV under exponential propagation,
confirming global weight convergence with no overfitting.
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
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300

# ── Binary propagation results (confirmed from original LOOCV, script 45) ────
binary_rates = [1.000, 0.286, 0.659]
binary_n     = [23, 21, 44]
binary_k     = [23, 6, 29]

# ── Exponential propagation results (from script 65 loocv_summary.json) ──────
with open("results/exponential_propagation/loocv_summary.json") as f:
    loocv = json.load(f)

exp_rates = [
    loocv["station_11447650_top10_pct"] / 100,
    loocv["station_11447890_top10_pct"] / 100,
    loocv["overall_top10_pct"] / 100,
]
exp_n = [loocv["n_11447650"], loocv["n_11447890"], loocv["overall_n"]]
exp_k = [round(r * n) for r, n in zip(exp_rates, exp_n)]

# ── Groups ────────────────────────────────────────────────────────────────────
groups = ['Station\n11447650', 'Station\n11447890', 'Overall']

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

x     = np.arange(len(groups))
width = 0.35

bars_b = ax.bar(x - width/2, binary_rates, width,
                label='Binary Propagation (LOOCV)',
                color='#e67e22', alpha=0.85, edgecolor='black', linewidth=1.2)
bars_e = ax.bar(x + width/2, exp_rates, width,
                label='Exponential Propagation (LOOCV)',
                color='#3498db', alpha=0.85, edgecolor='black', linewidth=1.2)

# Labels on bars
for bar, rate, k, n in zip(bars_b, binary_rates, binary_k, binary_n):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{rate:.1%}\n({k}/{n})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, rate, k, n in zip(bars_e, exp_rates, exp_k, exp_n):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{rate:.1%}\n({k}/{n})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Delta annotations (exponential - binary)
for i, (rb, re) in enumerate(zip(binary_rates, exp_rates)):
    delta = re - rb
    if abs(delta) > 0.01:
        color_d = '#27ae60' if delta > 0 else '#c0392b'
        sign    = '+' if delta > 0 else ''
        ax.annotate('',
                    xy=(i + width/2, re + 0.03),
                    xytext=(i - width/2, rb + 0.03),
                    arrowprops=dict(arrowstyle='->', color=color_d, lw=1.8))
        ax.text(i, max(rb, re) + 0.11,
                f'{sign}{delta*100:.1f}pp',
                ha='center', va='bottom', fontsize=9,
                color=color_d, fontweight='bold')

# Null baseline
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.8,
           label='Null baseline (50%)', zorder=0)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Top-10 Success Rate', fontweight='bold')
ax.set_title('Propagation Model Comparison: Binary vs Exponential\n'
             'Leave-One-Out Cross-Validation  |  n = 178 candidate facilities',
             fontweight='bold', pad=12)
ax.set_ylim([0, 1.28])
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Weight note
ax.text(0.98, 0.04,
        'Binary weights (LOOCV): [0.00, 0.00, 0.55, 0.45]\n'
        'Exponential weights (LOOCV): [0.00, 0.00, 0.60, 0.40]\n'
        'Circular = LOOCV under exponential propagation\n'
        '(global weight convergence; no overfitting)',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────────────
Path("results/publication_figures").mkdir(parents=True, exist_ok=True)
plt.savefig("results/publication_figures/Figure3_Propagation_Comparison.png",
            dpi=300, bbox_inches='tight')
plt.savefig("results/publication_figures/Figure3_Propagation_Comparison.pdf",
            bbox_inches='tight')
plt.close()

print("[OK] Figure 3 saved: Binary vs Exponential Propagation Comparison")
print(f"     Binary:      S650={binary_rates[0]:.1%}  S890={binary_rates[1]:.1%}  Overall={binary_rates[2]:.1%}")
print(f"     Exponential: S650={exp_rates[0]:.1%}  "
      f"S890={exp_rates[1]:.1%}  Overall={exp_rates[2]:.1%}")
delta_s650 = (exp_rates[0] - binary_rates[0]) * 100
delta_s890 = (exp_rates[1] - binary_rates[1]) * 100
delta_all  = (exp_rates[2] - binary_rates[2]) * 100
print(f"     Delta:       S650={delta_s650:+.1f}pp  S890={delta_s890:+.1f}pp  Overall={delta_all:+.1f}pp")
print("     PNG: results/publication_figures/Figure3_Propagation_Comparison.png")
print("     PDF: results/publication_figures/Figure3_Propagation_Comparison.pdf")
