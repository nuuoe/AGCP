"""Generate the ALFWorld scale-curve figure (Fig. 3 in the paper).

Accept counts are read from the Table 2 run files: n_accept (n_llm_accept
for the Qwen-1.5B fair-eval files) over n_total rows, from runs/ for the
IID split and runs/ood/ for the OOD split. Systems are coloured by the
three bands of Sec. 5.2 (band boundaries visible in the pairwise McNemar
matrices of App. I): grey for regex and Qwen-1.5B, blue for GPT-3.5,
Qwen-7B, Llama-70B and 4o-mini, orange for Haiku, GPT-4o and Sonnet, with
light background shading per band. IID points are filled circles, OOD
points open triangles, both with Wilson 95% intervals; each IID point is
labelled with a short model name. The palette is colourblind-safe (Paul
Tol).
"""
import json
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import os


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


# Band palette (Paul Tol), distinguishable in greyscale and under
# deuteranopia and protanopia.
COLOR_BAND1 = "#777777"   # neutral grey
COLOR_BAND2 = "#5B8AC8"   # muted blue (Tol "muted" cb-safe)
COLOR_BAND3 = "#E07B39"   # rich orange (Tol "muted" cb-safe)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def accept_count(rel_path):
    """(k, n) from a Table 2 run file: n_accept (n_llm_accept for the
    fair-eval files) over n_total, the fields scripts/verify_tables.py checks."""
    with open(os.path.join(ROOT, rel_path)) as f:
        d = json.load(f)
    k = d["n_llm_accept"] if "n_llm_accept" in d else d["n_accept"]
    return k, d["n_total"]


# Bands 1 (~30%), 2 (~40-49%) and 3 (~50-65%) follow the IID accept rates
# and the grouping of Sec. 5.2: the four mid-range LLMs (3.5/7B/70B/4o-mini)
# form band 2 and Haiku joins GPT-4o + Sonnet in band 3 (the McNemar matrix
# separates Haiku from the band-2 cluster and from GPT-4o).
SYSTEM_FILES = [
    # (full label, short label, IID run file, OOD run file, x-pos, band)
    ("Regex",               "Regex",       "alfworld_regex_v2_full.json",          "alfworld_regex_v2_unseen.json",          0.5, 1),
    ("Qwen-1.5B",           "Qwen-1.5B",   "alfworld_fair_eval_full.json",         "alfworld_fair_eval_unseen.json",         1.5, 1),
    ("GPT-3.5-turbo",       "GPT-3.5",     "alfworld_cloud_gpt35.json",            "alfworld_cloud_gpt35.json",              6.0, 2),
    ("Qwen-7B-Turbo",       "Qwen-7B",     "alfworld_cloud_together_qwen7b.json",  "alfworld_cloud_together_qwen7b.json",    7.0, 2),
    ("Llama-3.3-70B-Turbo", "Llama-70B",   "alfworld_cloud_together_llama70b.json","alfworld_cloud_together_llama70b.json",  70.0, 2),
    ("GPT-4o-mini",         "GPT-4o-mini", "alfworld_cloud_openai_mini.json",      "alfworld_cloud_openai_mini.json",        25.0, 2),
    ("Claude Haiku 4.5",    "Haiku",       "alfworld_cloud_claude_haiku.json",     "alfworld_cloud_claude_haiku.json",       60.0, 3),
    ("GPT-4o",              "GPT-4o",      "alfworld_cloud_gpt4o.json",            "alfworld_cloud_gpt4o.json",              800.0, 3),
    ("Claude Sonnet 4.6",   "Sonnet",      "alfworld_cloud_claude_sonnet.json",    "alfworld_cloud_claude_sonnet.json",      1200.0, 3),
]
SYSTEMS = [
    # (full label, short label, IID (k,n), OOD (k,n), x-pos, band)
    (full, short, accept_count(os.path.join("runs", iid)),
     accept_count(os.path.join("runs", "ood", ood)), x, band)
    for full, short, iid, ood, x, band in SYSTEM_FILES
]

BAND_COLOR = {1: COLOR_BAND1, 2: COLOR_BAND2, 3: COLOR_BAND3}

# Band y-bands for background shading.
BANDS = [
    (0.275, 0.355, COLOR_BAND1, "Lower band"),
    (0.39,  0.50,  COLOR_BAND2, "Middle band"),
    (0.50,  0.72,  COLOR_BAND3, "Frontier group"),
]

# Figure.
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

# Background band bands (very light, no legend entry).
for y_lo, y_hi, color, _ in BANDS:
    ax.axhspan(y_lo, y_hi, facecolor=color, alpha=0.09, zorder=0)

# One line per band; IID filled circles, OOD open triangles.
for band in (1, 2, 3):
    color = BAND_COLOR[band]
    xs_i, ys_i, lo_i, hi_i = [], [], [], []
    xs_o, ys_o, lo_o, hi_o = [], [], [], []
    for _, _, iid, ood, x, t in SYSTEMS:
        if t != band:
            continue
        if iid is not None:
            k, n = iid
            p, lo, hi = wilson(k, n)
            xs_i.append(x); ys_i.append(p)
            lo_i.append(p - lo); hi_i.append(hi - p)
        if ood is not None:
            k, n = ood
            p, lo, hi = wilson(k, n)
            xs_o.append(x); ys_o.append(p)
            lo_o.append(p - lo); hi_o.append(hi - p)

    # Sort by x so the line runs left to right.
    order_i = np.argsort(xs_i)
    xs_i = np.array(xs_i)[order_i]; ys_i = np.array(ys_i)[order_i]
    lo_i = np.array(lo_i)[order_i]; hi_i = np.array(hi_i)[order_i]
    order_o = np.argsort(xs_o)
    xs_o = np.array(xs_o)[order_o]; ys_o = np.array(ys_o)[order_o]
    lo_o = np.array(lo_o)[order_o]; hi_o = np.array(hi_o)[order_o]

    ax.errorbar(xs_i, ys_i, yerr=[lo_i, hi_i],
                fmt='o-', color=color, markerfacecolor=color,
                markeredgecolor=color, markersize=8, linewidth=1.4,
                capsize=3, alpha=0.95, zorder=3)
    ax.errorbar(xs_o, ys_o, yerr=[lo_o, hi_o],
                fmt='^--', color=color, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=1.5,
                markersize=8, linewidth=1.0, capsize=3,
                alpha=0.95, zorder=3)

# Label offsets chosen so the labels clear the error bars.
ANNOTATE_OFFSETS = {
    "Regex":       (6, -12),
    "Qwen-1.5B":   (6, -14),
    "GPT-3.5":     (6, -14),
    "Qwen-7B":     (-30, 6),
    "Llama-70B":   (-50, -16),
    "GPT-4o-mini": (-25, 9),
    "Haiku":       (6, -12),
    "GPT-4o":      (6, -12),
    "Sonnet":      (-12, 9),
}
for _, short, iid, _, x, band in SYSTEMS:
    k, n = iid
    p, _, _ = wilson(k, n)
    dx, dy = ANNOTATE_OFFSETS.get(short, (6, -12))
    ax.annotate(short, (x, p), xytext=(dx, dy),
                textcoords='offset points', fontsize=8.5,
                color=BAND_COLOR[band], fontweight='medium')

# Regex baseline.
regex_p, _, _ = wilson(*SYSTEMS[0][2])
ax.axhline(y=regex_p, color=COLOR_BAND1, linestyle=':',
           alpha=0.55, linewidth=1.0, zorder=1)

# Axes.
ax.set_xscale('log')
ax.set_xlabel("Model scale (B parameters; regex at 0.5 for layout)",
              fontsize=11)
ax.set_ylabel(r"Accept rate ($F_1 \geq 0.9$)", fontsize=11)
ax.set_title("ALFWorld NL-parse scale curve  ($N{=}251$, Wilson 95% CIs)",
             fontsize=12, fontweight='bold')

ax.set_ylim(0.22, 0.72)
ax.set_xlim(0.35, 2200)
ax.grid(True, which='major', axis='y', alpha=0.25, linewidth=0.7)
ax.grid(True, which='minor', axis='y', alpha=0.0)
ax.tick_params(axis='both', which='major', length=4)

# Two legends: bands by colour, splits by marker.
from matplotlib.lines import Line2D
band_handles = [
    Line2D([0], [0], color=COLOR_BAND1, marker='o', linestyle='-',
           markersize=7, label='Lower band (regex + 1.5B)'),
    Line2D([0], [0], color=COLOR_BAND2, marker='o', linestyle='-',
           markersize=7, label='Middle band (3.5 / 7B / 70B / 4o-mini)'),
    Line2D([0], [0], color=COLOR_BAND3, marker='o', linestyle='-',
           markersize=7, label='Frontier group (Haiku / 4o / Sonnet)'),
]
split_handles = [
    Line2D([0], [0], color='#333', marker='o', linestyle='-',
           markersize=7, label='In-distribution (seen)'),
    Line2D([0], [0], color='#333', marker='^', linestyle='--',
           markerfacecolor='white', markeredgewidth=1.5,
           markersize=7, label='Out-of-distribution (unseen)'),
]
leg1 = ax.legend(handles=band_handles, loc='upper left',
                 fontsize=8, frameon=True, framealpha=0.9,
                 edgecolor='#cccccc')
ax.add_artist(leg1)
ax.legend(handles=split_handles, loc='lower right',
          fontsize=8, frameon=True, framealpha=0.9,
          edgecolor='#cccccc')

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "alfworld_scale_curve.pdf")
plt.savefig(out_path, bbox_inches='tight')
print(f"wrote: {out_path}")
plt.savefig(out_path.replace('.pdf', '.png'),
            bbox_inches='tight', dpi=200)
print(f"wrote: {out_path.replace('.pdf', '.png')}")
