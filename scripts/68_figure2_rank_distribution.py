"""
Figure 2 (New): Bimodal Rank Distribution and Model Structure

Two-panel figure:
  Panel A: AMPAC rank for each LOOCV event at each station.
           Shows the bimodal structure: rank 4-5 (success) vs rank 144-145
           (failure), with no intermediate ranks.
  Panel B: Weight bar chart showing the optimal weight allocation
           and labels clarifying what each component actually contributes.

Reads: results/exponential_propagation/loocv_results.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
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

# ── Load LOOCV results ────────────────────────────────────────────────────────
df = pd.read_csv("results/exponential_propagation/loocv_results.csv")

# Assign jitter to y-axis so dots don't overlap
np.random.seed(42)
df = df.sort_values(["station", "event_id"]).reset_index(drop=True)

# ── Create figure ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 6))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ── Panel A: Rank strip plot ──────────────────────────────────────────────────
station_labels = {11447650: 'Station 11447650\n(23 events)',
                  11447890: 'Station 11447890\n(21 events)'}
station_y      = {11447650: 1, 11447890: 0}
colors_outcome = {1: '#2ecc71', 0: '#e74c3c'}  # green=success, red=failure

for _, row in df.iterrows():
    y_base = station_y[row["station"]]
    jitter  = np.random.uniform(-0.12, 0.12)
    color   = colors_outcome[row["top10"]]
    ax_a.scatter(row["rank"], y_base + jitter,
                 color=color, s=60, alpha=0.85, edgecolors='white', linewidths=0.5,
                 zorder=3)

# Top-10 boundary line
ax_a.axvline(x=10, color='navy', linestyle='--', linewidth=1.5,
             label='Top-10 threshold', zorder=2)

# Shaded region for top-10
ax_a.axvspan(0, 10, alpha=0.07, color='green', zorder=1)

# Annotations: rank clusters
for st, ypos in station_y.items():
    st_df = df[df["station"] == st]
    succ  = st_df[st_df["top10"] == 1]
    fail  = st_df[st_df["top10"] == 0]
    if len(succ) > 0:
        mean_rank = int(succ["rank"].median())
        ax_a.annotate(f'Rank ~{mean_rank}',
                      xy=(mean_rank, ypos + 0.20),
                      xytext=(mean_rank + 15, ypos + 0.30),
                      fontsize=8.5, color='#27ae60', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2))
    if len(fail) > 0:
        mean_rank_f = int(fail["rank"].median())
        ax_a.annotate(f'Rank ~{mean_rank_f}',
                      xy=(mean_rank_f, ypos - 0.20),
                      xytext=(mean_rank_f - 40, ypos - 0.35),
                      fontsize=8.5, color='#c0392b', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))

# Formatting Panel A
ax_a.set_yticks([0, 1])
ax_a.set_yticklabels([station_labels[11447890], station_labels[11447650]])
ax_a.set_xlabel('AMPAC Rank among 178 Candidate Facilities', fontweight='bold')
ax_a.set_title('(A) LOOCV Rank Distribution: Bimodal Structure',
               fontweight='bold', pad=10)
ax_a.set_xlim([0, 180])
ax_a.set_ylim([-0.5, 1.5])
ax_a.grid(axis='x', alpha=0.3, linestyle='--')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Legend Panel A
succ_patch = mpatches.Patch(color='#2ecc71', label='Top-10 (success)')
fail_patch  = mpatches.Patch(color='#e74c3c', label='Rank >10 (failure)')
ax_a.legend(handles=[succ_patch, fail_patch],
            loc='upper center', frameon=True, fontsize=9)

# ── Panel B: Weight allocation bar chart ─────────────────────────────────────
components = ['Distance\n(d)', 'Industry\n(i)', 'Anomaly\n(a)', 'Propagation\n(p)']
weights    = [0.00, 0.00, 0.60, 0.40]
bar_colors = ['#95a5a6', '#95a5a6', '#f39c12', '#3498db']
roles      = ['Not used\n(parameterisation\nmismatch)',
              'Not used\n(single facility\nin class)',
              'Event-level\nweight\n(severity)',
              'Within-event\ndiscriminator\n(travel time)']

bars_b = ax_b.bar(range(len(components)), weights,
                  color=bar_colors, alpha=0.85, edgecolor='black', linewidth=1.2)

for i, (bar, w, role) in enumerate(zip(bars_b, weights, roles)):
    if w > 0:
        ax_b.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                  f'{w:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax_b.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2.,
                  role, ha='center', va='center', fontsize=7.5, color='white',
                  fontweight='bold', wrap=True)
    else:
        ax_b.text(bar.get_x() + bar.get_width()/2., 0.02,
                  role, ha='center', va='bottom', fontsize=7.5, color='#555',
                  style='italic')

ax_b.set_xticks(range(len(components)))
ax_b.set_xticklabels(components)
ax_b.set_ylabel('Optimal Weight', fontweight='bold')
ax_b.set_title('(B) Optimal Weight Allocation\nand Component Role',
               fontweight='bold', pad=10)
ax_b.set_ylim([0, 0.75])
ax_b.grid(axis='y', alpha=0.3, linestyle='--')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Add note below Panel B
fig.text(0.73, 0.01,
         'Within any single event, anomaly is\nconstant across all 178 facilities.\nRanking determined solely by propagation.',
         ha='center', fontsize=7.5, style='italic', color='#555',
         va='bottom')

# ── Save ──────────────────────────────────────────────────────────────────────
Path("results/publication_figures").mkdir(parents=True, exist_ok=True)
plt.savefig("results/publication_figures/Figure2_Rank_Distribution.png",
            dpi=300, bbox_inches='tight')
plt.savefig("results/publication_figures/Figure2_Rank_Distribution.pdf",
            bbox_inches='tight')
plt.close()

print("[OK] Figure 2 saved: Bimodal Rank Distribution")
print("     PNG: results/publication_figures/Figure2_Rank_Distribution.png")
print("     PDF: results/publication_figures/Figure2_Rank_Distribution.pdf")

# Print rank summary for verification
print("\nRank summary from LOOCV results:")
for st in [11447650, 11447890]:
    st_df = df[df["station"] == st]
    print(f"  Station {st}: "
          f"unique ranks = {sorted(st_df['rank'].unique())}, "
          f"success = {st_df['top10'].sum()}/{len(st_df)}")
