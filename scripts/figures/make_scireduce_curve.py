"""Generate the SCI-ReDuce held-out coverage curve (Fig. 2 and Table 4 in the paper).

Reads runs/sci_reduce_coverage_curve.json (mean coverage and standard
deviation per training size over four seeds) and writes
figures/scireduce_curve.pdf and .png. The orange line matches the
higher-band colour of the scale-curve figure; the filled band is one
standard deviation.
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import json
import os

# Same colour as the higher band in make_scale_curve.py.
COLOR_TIER3 = "#E07B39"

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
d = json.load(open(os.path.join(ROOT, "runs", "sci_reduce_coverage_curve.json")))

xs = [r["train_size"] for r in d["results"]]
ys = [r["mean_coverage_pct"] / 100.0 for r in d["results"]]
yerr = [r["stdev"] / 100.0 for r in d["results"]]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(6, 4))

# Error band first so it sits behind the line.
ax.fill_between(xs,
                [y - e for y, e in zip(ys, yerr)],
                [y + e for y, e in zip(ys, yerr)],
                alpha=0.20, color=COLOR_TIER3, linewidth=0,
                label='±1 stdev (4 seeds)')

ax.errorbar(xs, ys, yerr=yerr, fmt='o-', color=COLOR_TIER3,
            markerfacecolor=COLOR_TIER3, markeredgecolor=COLOR_TIER3,
            capsize=3, markersize=8, linewidth=1.8,
            label='SCI-ReDuce coverage')

ax.set_xlabel("Training trajectories", fontsize=11)
ax.set_ylabel("Held-out plan coverage", fontsize=11)
ax.set_title("SCI-ReDuce empirical learning curve\n"
             "(Mystery Blocksworld, 4 seeds)",
             fontsize=12, fontweight='bold')
ax.grid(True, which='major', axis='y', alpha=0.25, linewidth=0.7)
ax.tick_params(axis='both', which='major', length=4)
ax.set_ylim(0, 0.5)
ax.set_xlim(8, 43)
ax.set_xticks(xs)

# Slope annotation (extrapolated regime).
ax.annotate(r"slope $\approx$ 1 pp / trajectory",
            xy=(34, 0.30), xytext=(13, 0.38),
            fontsize=10, color='#444',
            arrowprops=dict(arrowstyle='->', color='#444',
                            alpha=0.7, lw=1.0))

ax.legend(loc='upper left', fontsize=9, frameon=True,
          framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout()
out_path = os.path.join(ROOT, "figures", "scireduce_curve.pdf")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, bbox_inches='tight')
plt.savefig(out_path.replace('.pdf', '.png'),
            bbox_inches='tight', dpi=200)
print(f"wrote: {out_path} (+ .png)")
