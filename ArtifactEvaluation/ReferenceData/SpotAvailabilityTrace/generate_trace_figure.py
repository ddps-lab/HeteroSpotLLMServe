#!/usr/bin/env python3
"""
Generate a spot availability trace figure for a given time window.
Edit WINDOW_START / WINDOW_END below, then run:
    python generate_trace_figure.py
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

# ── Instance config ──────────────────────────────────────────────
HETERO_CONFIG = {
    'g6.12xlarge': 3,
    'g5.12xlarge': 2,
    'g6e.xlarge': 4,
}

INSTANCE_ORDER = ['g5.12xlarge', 'g6.12xlarge', 'g6e.xlarge']
LINESTYLES = ['-', '--', '-.']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']

# ── Window to plot (edit here) ──────────────────────────────────
WINDOW_START = '2026-03-17 03:55'
WINDOW_END   = '2026-03-17 04:45'

# ── Font sizes (edit here) ──────────────────────────────────────
FONT_XLABEL = 16
FONT_YLABEL = 16
FONT_TICK = 16
FONT_LEGEND = 16

# ── Figure sizes ────────────────────────────────────────────────
FIGSIZE_TRACE = (4, 3)
FIGSIZE_LEGEND = (4, 0.4)

# ── Output directory ────────────────────────────────────────────
OUTPUT_DIR = 'figures'


def load_data(window_start, window_end):
    """Load CSV files covering the window and resample to 5-min availability."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_csvs = sorted(glob.glob(os.path.join(script_dir, '2026-03-*.csv')))

    start_date = window_start.normalize()
    end_date = window_end.normalize()

    csv_files = [
        f for f in all_csvs
        if start_date <= pd.Timestamp(os.path.basename(f).replace('.csv', '')) <= end_date
    ]
    if not csv_files:
        sys.exit(f'No CSV files found for {start_date.date()} ~ {end_date.date()}')

    instance_types = list(HETERO_CONFIG.keys())
    df_raw = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df_raw = df_raw[df_raw['InstanceType'].isin(instance_types)].copy()
    df_raw['RequestDateTime'] = pd.to_datetime(df_raw['RequestDateTime'])
    df_raw = df_raw.sort_values('RequestDateTime').reset_index(drop=True)

    availability_series = {}
    for itype in instance_types:
        needed = HETERO_CONFIG[itype]
        subset = df_raw[df_raw['InstanceType'] == itype][['RequestDateTime', 'Success']].copy()
        subset = subset.set_index('RequestDateTime').sort_index()
        subset = subset[~subset.index.duplicated(keep='last')]
        subset.index = subset.index.floor('5min')
        subset_resampled = subset.resample('5min').ffill()
        subset_resampled['availability'] = subset_resampled['Success'].clip(upper=needed)
        availability_series[itype] = subset_resampled['availability']

    avail_df = pd.DataFrame(availability_series).dropna()
    return avail_df


def plot_trace(avail_df, window_start, window_end, output_path):
    """Generate and save trace figure + separate legend."""
    window_data = avail_df.loc[window_start:window_end].iloc[:-1]
    # Extend last value flat to window_end (avoid steps-post vertical artifact)
    last_row = window_data.iloc[[-1]].copy()
    last_row.index = [window_end]
    window_data = pd.concat([window_data, last_row])
    if window_data.empty:
        sys.exit(f'No data in window {window_start} ~ {window_end}')

    # ── Trace figure ──
    fig, ax = plt.subplots(figsize=FIGSIZE_TRACE)

    # Convert to elapsed minutes from window_start
    elapsed_min = (window_data.index - window_start).total_seconds() / 60

    for j, itype in enumerate(INSTANCE_ORDER):
        ax.plot(
            elapsed_min, window_data[itype],
            label=itype, color=COLORS[j],
            linestyle=LINESTYLES[j], linewidth=2,
            drawstyle='steps-post', marker='o', markersize=3,
        )

    total_min = (window_end - window_start).total_seconds() / 60
    ax.set_xlim(0, total_min)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylim(-0.2, max(HETERO_CONFIG.values()) + 0.5)
    ax.set_yticks(range(0, max(HETERO_CONFIG.values()) + 1))
    ax.set_xlabel('Elapsed Time (min)', fontsize=FONT_XLABEL)
    ax.set_ylabel('Spot Availability', fontsize=FONT_YLABEL)
    ax.tick_params(labelsize=FONT_TICK)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=600)
    print(f'Saved: {output_path}')

    # ── Separate legend figure ──
    handles, labels = ax.get_legend_handles_labels()
    plt.close(fig)

    fig_leg, ax_leg = plt.subplots(figsize=FIGSIZE_LEGEND)
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, loc='center', ncol=len(INSTANCE_ORDER),
                  fontsize=FONT_LEGEND, frameon=True)
    fig_leg.tight_layout()

    figures_dir = os.path.dirname(output_path)
    ext = os.path.splitext(output_path)[1]
    legend_path = os.path.join(figures_dir, f'spot_trace_legend{ext}')
    fig_leg.savefig(legend_path, bbox_inches='tight', dpi=600)
    print(f'Saved: {legend_path}')
    plt.close(fig_leg)


def main():
    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)

    if window_end <= window_start:
        sys.exit('End time must be after start time.')

    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(figures_dir, exist_ok=True)

    s = window_start.strftime('%m%d_%H%M')
    e = window_end.strftime('%m%d_%H%M')
    output_path = os.path.join(figures_dir, f'spot_trace_{s}_{e}.pdf')

    avail_df = load_data(window_start, window_end)
    plot_trace(avail_df, window_start, window_end, output_path)


if __name__ == '__main__':
    main()
