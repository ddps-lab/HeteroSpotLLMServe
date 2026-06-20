#!/usr/bin/env python3
"""
Compare optimizer cost on the 15-type vs 46-type cluster, swept over the
node-multiplier M. Two panels:
  (a) first-pipeline (round-1) search time vs nodes
  (b) full extraction time vs nodes

Data is parsed directly from the collected sweeps:
  uswest2_46type/M*/{json,logs}       — 46-type, one dir per M
  large_hetero_15type/{json,logs}     — 15-type, one combined run

Run: python3 plot_comparison.py   →   cluster_type_comparison.png
"""

import os
import re
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

# ── Font / style config (tweak these and re-run) ──────────────────────
FONT_TITLE   = 15
FONT_LABEL   = 24
FONT_TICK    = 20
FONT_LEGEND  = 18
LINE_WIDTH   = 2.0
MARKER_SIZE  = 8
FIG_W, FIG_H = 11, 4.6          # combined two-panel figure
FIG_W_SINGLE, FIG_H_SINGLE = 5.5, 4.1   # each standalone (paper) figure
DPI          = 300
PANEL_A_YMAX = 4200   # raise (a)'s y-limit to flatten the curves / clear the legend
LEGEND_LOC   = "upper left"
LEGEND_NCOL  = 2      # standalone legend is exported separately, this many columns
# Fixed margins for the standalone panels so both have an IDENTICAL plot-box
# width regardless of y-tick label digit count (4000 vs 40). left must clear
# the y-label + widest tick label.
PANEL_MARGINS = dict(left=0.235, right=0.92, bottom=0.205, top=0.965)
# Inset (figure-in-figure) zooming the first few points of (b), where the
# curve is superlinear (knee) before it straightens out at large N.
INSET_ON        = True
# RIGHT box — the inset itself: position & size ON THE FIGURE (axes fraction).
INSET_RECT      = (0.62, 0.13, 0.30, 0.34)   # x0, y0, w, h
# ZOOM VIEW — the data range the inset magnifies (x = # instances, y = hours).
INSET_XMIN      = -8     # so the 1st point isn't cut off at the inset's left edge
INSET_YMIN      = -0.3   # so the 1st point isn't cut off at the inset's bottom
INSET_XMAX      = 85     # first 4 points (N=15,30,45,60) with margin
INSET_YMAX      = 4.0
# LEFT box — the gray rectangle drawn on the PARENT plot, in data coords
# (x0, x1, y0, y1). This is the knob for the left box's size & position,
# INDEPENDENT of the zoom view above. None → match the zoom view exactly.
INSET_BOX       = 0, 75, -0.9, 3.5   # e.g. (-5, 95, -1.0, 5.0)
INSET_FONT_TICK = 13

COLOR_15 = "tab:blue"
COLOR_46 = "tab:red"

NODE_MAX = 1000   # only plot points with fewer than this many nodes

_HERE = os.path.dirname(os.path.abspath(__file__))


# ── Data parsing ──────────────────────────────────────────────────────
def round1_time_sec(worker_log: str) -> float:
    """Total optimizer time of round 1 (first pipeline) = last layer-80 total."""
    last = 0.0
    for line in open(worker_log):
        if "══ Round 2" in line:
            break
        m = re.search(r"Layer\s+\d+/80 done .*?total=\s*([\d.]+)s", line)
        if m:
            last = float(m.group(1))
    return last


def worker_log_for(run_dir: str, M: int) -> str:
    suffix = "" if M == 1 else f"_x{M}"
    for p in glob.glob(os.path.join(run_dir, "logs", "workers_*", "m=*.log")):
        if p.endswith(f"full_cluster{suffix}_k=1.log"):
            return p
    raise FileNotFoundError(f"worker log for M={M} under {run_dir}")


def p1_rows(json_glob, worker_subdir):
    """First-pipeline-only runs (firstpipe_p1/): extend the 1st-pipeline curve
    to large M. They have NO full-extraction time → full column is NaN, so they
    appear only in panel (a). round-1 time is parsed from the p1 worker log,
    identical in definition to the full-run points."""
    jf = glob.glob(os.path.join(_HERE, "firstpipe_p1", "json", json_glob))[0]
    rows = []
    for r in [x for x in json.load(open(jf))["results"] if "error" not in x]:
        M = r["node_multiplier"]
        wl = glob.glob(os.path.join(_HERE, "firstpipe_p1", "logs", worker_subdir,
                                    f"*_x{M}_p1_k=1.log"))[0]
        rows.append((r["total_nodes"], round1_time_sec(wl), float("nan")))
    return rows


def _capped(rows):
    return sorted(r for r in rows if r[0] < NODE_MAX)


def load_46type():
    """46-type: per-M full runs (M=1..6) + first-pipeline-only runs (M=8..)."""
    rows = []
    for M in (1, 2, 3, 4, 6):
        base = os.path.join(_HERE, "uswest2_46type", f"M{M}")
        jf = glob.glob(os.path.join(base, "json", "*.json"))[0]
        r = [x for x in json.load(open(jf))["results"] if "error" not in x][0]
        rows.append((r["total_nodes"], round1_time_sec(worker_log_for(base, M)),
                     r["wall_time_sec"]))
    rows += p1_rows("results_*uswest2_full_cluster*p1.json", "workers_46type_M8-24")
    return _capped(rows)


def load_15type():
    """15-type: combined full run (M=1..32) + first-pipeline-only runs (M=40..)."""
    run_dir = os.path.join(_HERE, "large_hetero_15type")
    jf = glob.glob(os.path.join(run_dir, "json", "*.json"))[0]
    rows = []
    for r in [x for x in json.load(open(jf))["results"] if "error" not in x]:
        M = r["node_multiplier"]
        rows.append((r["total_nodes"], round1_time_sec(worker_log_for(run_dir, M)),
                     r["wall_time_sec"]))
    rows += p1_rows("results_*large_hetero_cluster*p1.json", "workers_15type_M40-80")
    return _capped(rows)


# ── Plot ──────────────────────────────────────────────────────────────
# Panel specs: (key, y-column, y-scale, title, ylabel, ymax-or-None)
PANELS = {
    "find_1st": (1, 1.0,    "(a) Time to find the 1st pipeline", "Time (s)", "PANEL_A_YMAX"),
    "full":     (2, 1/3600, "(b) Full extraction time",         "Time (h)", None),
}


def add_inset(ax, series):
    """Figure-in-figure zoom of the first points (style follows
    modelplacement_top_k_beam_figure.ipynb): inset_axes + a rectangle on the
    parent (indicate_inset) + two corner connectors (ConnectionPatch)."""
    axins = ax.inset_axes(INSET_RECT)
    for data, color, mk, lbl in series:
        # plot the FULL series; the axes clips it to the zoom window, so the
        # line continues past the 4th point and exits the edge (not floating).
        axins.plot(data[:, 0], data[:, 2] / 3600, marker=mk, color=color,
                   lw=LINE_WIDTH, ms=MARKER_SIZE - 2)
    axins.set_xlim(INSET_XMIN, INSET_XMAX)
    axins.set_ylim(INSET_YMIN, INSET_YMAX)
    axins.tick_params(labelsize=INSET_FONT_TICK)
    axins.grid(alpha=0.3)

    # Left box (gray rectangle on the parent): INSET_BOX if given, else the
    # zoom view. x0,x1,y0,y1 are parent data coords.
    if INSET_BOX is None:
        x0, x1, y0, y1 = INSET_XMIN, INSET_XMAX, INSET_YMIN, INSET_YMAX
    else:
        x0, x1, y0, y1 = INSET_BOX
    ax.indicate_inset((x0, y0, x1 - x0, y1 - y0), edgecolor="dimgray",
                      alpha=1.0, linewidth=1.2)
    for corner_y, inset_xy in [(y1, (0, 1)), (y0, (0, 0))]:
        con = ConnectionPatch(xyA=(x1, corner_y), coordsA="data", axesA=ax,
                              xyB=inset_xy, coordsB="axes fraction", axesB=axins,
                              color="dimgray", lw=1.0)
        ax.figure.add_artist(con)


def draw_panel(ax, panel_key, series, standalone=False):
    ycol, yscale, title, ylabel, ymax_var = PANELS[panel_key]
    for data, color, mk, lbl in series:
        # drop rows with no value for this panel (p1 rows have NaN full time,
        # so they appear only in panel (a), not (b))
        valid = data[~np.isnan(data[:, ycol])]
        ax.plot(valid[:, 0], valid[:, ycol] * yscale, marker=mk, color=color,
                label=lbl, lw=LINE_WIDTH, ms=MARKER_SIZE)
    # standalone (paper) figures: no title; x-axis labelled "# instances";
    # legend is exported separately, so it is omitted from the panel itself.
    ax.set_xlabel("# instances" if standalone else "nodes", fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    if not standalone:
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.legend(fontsize=FONT_LEGEND, loc=LEGEND_LOC)
    if ymax_var:
        ax.set_ylim(0, globals()[ymax_var])
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(alpha=0.3)
    if panel_key == "full" and INSET_ON:
        add_inset(ax, series)


def save_legend(series, stem="optimizer_scalability_legend"):
    """Export the shared legend on its own, as a single horizontal row."""
    handles = [Line2D([0], [0], color=color, marker=mk, lw=LINE_WIDTH,
                      ms=MARKER_SIZE, label=lbl) for _, color, mk, lbl in series]
    fig = plt.figure(figsize=(FIG_W, 0.5))
    fig.legend(handles=handles, ncol=LEGEND_NCOL, loc="center",
               fontsize=FONT_LEGEND, frameon=True)
    save(fig, stem)
    plt.close(fig)


def save(fig, stem, tight=True):
    """Save a figure as both PNG and PDF under figures/.

    tight=True crops to content (combined fig, legend); tight=False honors the
    exact figsize + margins (standalone panels, so plot boxes match in width).
    """
    figdir = os.path.join(_HERE, "figures")
    os.makedirs(figdir, exist_ok=True)
    bbox = "tight" if tight else None
    for ext in ("png", "pdf"):
        path = os.path.join(figdir, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches=bbox)
        print(f"saved {path}")


def main():
    r15 = np.array(load_15type(), float)   # columns: nodes, round1_s, full_s
    r46 = np.array(load_46type(), float)
    series = [(r15, COLOR_15, "o", "15-type cluster"),
              (r46, COLOR_46, "s", "46-type cluster")]

    # Combined two-panel version (kept).
    fig, ax = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    draw_panel(ax[0], "find_1st", series)
    draw_panel(ax[1], "full", series)
    fig.tight_layout()
    save(fig, "cluster_type_comparison")
    plt.close(fig)

    # Standalone per-panel versions (paper-ready, png + pdf each).
    for key, stem in [("find_1st", "time_find_1st_pipeline"),
                      ("full", "time_full_extraction")]:
        fig, ax = plt.subplots(1, 1, figsize=(FIG_W_SINGLE, FIG_H_SINGLE))
        draw_panel(ax, key, series, standalone=True)
        # Fixed identical margins (no tight crop) → equal plot-box width.
        fig.subplots_adjust(**PANEL_MARGINS)
        save(fig, stem, tight=False)
        plt.close(fig)

    # Shared legend, exported on its own (2 columns, side by side).
    save_legend(series)


if __name__ == "__main__":
    main()
