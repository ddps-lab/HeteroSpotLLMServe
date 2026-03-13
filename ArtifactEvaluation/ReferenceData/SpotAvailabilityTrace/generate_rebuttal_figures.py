"""
Generate Rebuttal Appendix Figures for Spot Trace Analysis

This script generates publication-ready figures and tables to address the reviewer's
question: "Why did you only use one trace for experiments?"

Outputs:
- Individual trace figures (top-10, ranked by fluctuation)
- Summary table of all qualified traces
- CSV files with trace data
- LaTeX equation snippet

Usage:
    python generate_rebuttal_figures.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import cluster_analysis
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cluster_analysis as ca

# ============================================================================
# Configuration
# ============================================================================

# Data files (relative to parent directory)
CSV_FILES = [
    '../2025-09-29.csv',
    '../2025-09-30.csv',
    '../2025-10-01.csv'
]

# Heterogeneous configuration
HETERO_CONFIG = {
    'g6.12xlarge': 3,
    'g5.12xlarge': 2,
    'g6e.xlarge': 4
}

# Homogeneous configurations to compare
HOMO_CONFIGS = [
    {'g6.xlarge': 7},
    {'g6.12xlarge': 2},
    {'g6.xlarge': 8},
    {'g6e.xlarge': 4},
    {'g5.12xlarge': 2}
]

# Model size (70B model = 141GB)
MODEL_SIZE = 141000  # MB

# Output directory
OUTPUT_DIR = 'appendix_A_figures'

# Figure style (from ddd_scenario_figure.ipynb)
FONTSIZE = 24  # Increased by 4 for better readability in 4x5 layout
LINESTYLES = ['-', '--', '-.', ':']
LINEWIDTH = 4
MARKERSIZE = 3
DPI = 300
FIGSIZE = (6.5, 5)  # Adjusted width to fit 4 per row on letter size

# ============================================================================
# Helper Functions
# ============================================================================

def filter_full_initial_allocation(optimal_windows, df, hetero_config):
    """
    Filter windows where heterogeneous config has full initial allocation.

    Args:
        optimal_windows: List of optimal time windows
        df: DataFrame with availability data
        hetero_config: Heterogeneous configuration dict

    Returns:
        Filtered list of windows with full initial allocation
    """
    filtered_windows = []

    for window in optimal_windows:
        time_start = window['time_start']

        # Get data at first timestamp
        first_timestamp_data = df[df['RequestDateTime'] == time_start]

        # Check if all required instances are available
        can_allocate = True
        for instance_type, needed_count in hetero_config.items():
            instance_data = first_timestamp_data[
                first_timestamp_data['InstanceType'] == instance_type
            ]

            if instance_data.empty:
                can_allocate = False
                break

            available = instance_data.iloc[0]['Success']
            if available < needed_count:
                can_allocate = False
                break

        if can_allocate:
            filtered_windows.append(window)

    # Sort by fluctuation (highest first)
    filtered_windows.sort(key=lambda x: x['avg_fluctuation'], reverse=True)

    return filtered_windows


def generate_trace_figure(window, df, hetero_config, rank, output_dir):
    """
    Generate a single trace figure showing instance availability over time.

    Style matches ddd_scenario_figure.ipynb
    """
    time_start = window['time_start']
    time_end = window['time_end']
    fluctuation = window['avg_fluctuation']

    # Filter data for this window with 25-minute offset
    # Start from 25 minutes after window start, show 50 minutes
    offset_start = time_start + pd.Timedelta(minutes=25)
    offset_end = offset_start + pd.Timedelta(minutes=50)

    # Get a wider range to interpolate from (5 minutes before)
    data_start = offset_start - pd.Timedelta(minutes=5)
    data_end = offset_end

    window_data = df[
        (df['RequestDateTime'] >= data_start) &
        (df['RequestDateTime'] <= data_end)
    ]

    # Get instance types in specific order: g5.12xlarge, g6.12xlarge, g6e.xlarge
    instance_types = ['g5.12xlarge', 'g6.12xlarge', 'g6e.xlarge']

    # Create figure
    plt.figure(figsize=FIGSIZE)

    for idx, instance_type in enumerate(instance_types):
        subset = window_data[window_data['InstanceType'] == instance_type]
        subset = subset.sort_values('RequestDateTime')

        if not subset.empty:
            # Cap availability at the needed count
            needed = hetero_config[instance_type]

            # Create 5-minute interval data by forward-filling
            # Generate all 5-minute timestamps we need
            time_range = pd.date_range(start=offset_start, end=offset_end, freq='5min')

            # Create a series with the original data
            original_times = subset['RequestDateTime']
            original_avails = subset['Success'].apply(lambda x: min(x, needed))

            # Interpolate to 5-minute intervals (forward fill)
            interpolated_avails = []
            for target_time in time_range:
                # Find the most recent data point before or at target_time
                valid_data = subset[subset['RequestDateTime'] <= target_time]
                if not valid_data.empty:
                    last_value = valid_data.iloc[-1]['Success']
                    interpolated_avails.append(min(needed, last_value))
                else:
                    # If no data before target_time, use the first available value
                    interpolated_avails.append(min(needed, subset.iloc[0]['Success']))

            # Convert to minutes elapsed from offset start (0-50 range)
            times = np.array([(t - offset_start).total_seconds() / 60 for t in time_range])
            avails = np.array(interpolated_avails)

            plt.plot(times, avails,
                    label=instance_type,
                    marker='o',
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH,
                    linestyle=LINESTYLES[idx % len(LINESTYLES)],
                    drawstyle='steps-post')

    # Set x-axis to show 50 minutes (0-50)
    plt.xlim(0, 50)
    plt.xlabel('Elapsed Time (minute)', fontsize=FONTSIZE + 3)
    plt.ylabel('Instance Availability', fontsize=FONTSIZE + 3)

    # Set y-axis to integer ticks
    max_needed = max(hetero_config.values())
    plt.yticks(range(0, max_needed + 1), fontsize=FONTSIZE + 3)
    plt.xticks(fontsize=FONTSIZE + 3)
    plt.ylim(bottom=0)

    # No legend on individual plots
    # Legend will be generated separately

    # No grid
    plt.grid(False)

    plt.tight_layout()

    # Save figure (no fluctuation value in filename)
    filename = f'trace_rank_{rank}.pdf'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  Generated: {filename}")


def generate_legend_only(output_dir):
    """
    Generate a separate legend file for all traces.
    """
    # Create a dummy figure just to extract the legend
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')

    # Create dummy lines with the same styles as in traces
    instance_types = ['g5.12xlarge', 'g6.12xlarge', 'g6e.xlarge']
    linestyles = ['-', '--', '-.', ':']

    for idx, instance_type in enumerate(instance_types):
        ax.plot([], [],
                label=instance_type,
                linewidth=LINEWIDTH,
                linestyle=linestyles[idx % len(linestyles)],
                marker='o',
                markersize=MARKERSIZE)

    # Create legend
    legend = ax.legend(loc='center', ncol=3, fontsize=FONTSIZE,
                      edgecolor='black', frameon=True)

    # Save just the legend
    filepath = os.path.join(output_dir, 'legend.pdf')
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  Generated: legend.pdf")


def generate_summary_table(filtered_windows, output_dir):
    """
    Generate a visual table (as PDF) showing all filtered traces.
    """
    # Prepare data
    data = []
    for idx, window in enumerate(filtered_windows, 1):
        data.append({
            'Rank': idx,
            'Start Time': window['time_start'].strftime('%m-%d %H:%M'),
            'End Time': window['time_end'].strftime('%m-%d %H:%M'),
            'Fluctuation': f"{window['avg_fluctuation']:.3f}",
            'Het Rate': f"{window['hetero_success_rate']:.2%}",
            'Avg Adv': f"{window['avg_advantage']:.2%}"
        })

    df_table = pd.DataFrame(data)

    # Create figure for table
    fig, ax = plt.subplots(figsize=(14, max(8, len(data) * 0.3)))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df_table.values,
                    colLabels=df_table.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.18, 0.18, 0.15, 0.15, 0.15])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(len(df_table.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Highlight rank 1 (used in paper)
    for i in range(len(df_table.columns)):
        cell = table[(1, i)]  # Row 1 in table (rank 1)
        cell.set_facecolor('#FFD966')
        cell.set_text_props(weight='bold')

    # Alternate row colors for readability
    for i in range(1, len(data) + 1):
        if i > 1:  # Skip rank 1 (already highlighted)
            color = '#F2F2F2' if i % 2 == 0 else 'white'
            for j in range(len(df_table.columns)):
                table[(i, j)].set_facecolor(color)

    plt.title('All Qualified Traces (Ranked by Fluctuation)\n' +
             'Rank #1 (highlighted) was used in main paper evaluation',
             fontsize=14, fontweight='bold', pad=20)

    # Save
    filepath = os.path.join(output_dir, 'table_all_traces.pdf')
    plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"  Generated: table_all_traces.pdf")


def save_csv_files(filtered_windows, output_dir):
    """
    Save trace data as CSV files.
    """
    # Prepare data
    data = []
    for idx, window in enumerate(filtered_windows, 1):
        # Apply same offset as in figures: 25-minute offset, 50-minute duration
        original_start = window['time_start']
        offset_start = original_start + pd.Timedelta(minutes=25)
        offset_end = offset_start + pd.Timedelta(minutes=50)

        data.append({
            'Rank': idx,
            'Date': offset_start.strftime('%Y-%m-%d'),
            'Start_Time': offset_start.strftime('%H:%M'),
            'End_Time': offset_end.strftime('%H:%M'),
            'Start_DateTime_ISO': offset_start.isoformat(),
            'End_DateTime_ISO': offset_end.isoformat(),
            'Fluctuation': window['avg_fluctuation'],
            'Het_Success_Rate': window['hetero_success_rate'],
            'Avg_Advantage': window['avg_advantage'],
            'Best_Advantage': window['best_advantage'],
            'Independence_Score': window['independence_score'],
            'Max_Homo_Rate': window['max_homo_rate'],
            'Max_Homo_Advantage': window['max_homo_advantage']
        })

    df = pd.DataFrame(data)

    # Save all traces
    filepath_all = os.path.join(output_dir, 'all_traces.csv')
    df.to_csv(filepath_all, index=False)
    print(f"  Generated: all_traces.csv ({len(df)} traces)")

    # Save top-10
    filepath_top10 = os.path.join(output_dir, 'top10_traces.csv')
    df.head(10).to_csv(filepath_top10, index=False)
    print(f"  Generated: top10_traces.csv")

    # Save bottom-10
    filepath_bottom10 = os.path.join(output_dir, 'bottom10_traces.csv')
    df.tail(10).to_csv(filepath_bottom10, index=False)
    print(f"  Generated: bottom10_traces.csv")


def save_latex_equation(output_dir):
    """
    Save fluctuation metric equation as LaTeX snippet.
    """
    latex_code = r"""% Fluctuation Metric Definition
\begin{equation}
\text{Fluctuation}(W) = \frac{1}{|T|} \sum_{t \in T} \sigma_{1h}(t)
\end{equation}

where $\sigma_{1h}(t)$ is the standard deviation of success rates within a 1-hour rolling window at time $t$, and $T$ is the set of all timestamps in window $W$.
"""

    filepath = os.path.join(output_dir, 'fluctuation_equation.tex')
    with open(filepath, 'w') as f:
        f.write(latex_code)

    print(f"  Generated: fluctuation_equation.tex")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("Generating Rebuttal Appendix Figures")
    print("="*80)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = ca.load_csv_data(CSV_FILES)

    # Filter out g6e.12xlarge (as in original analysis)
    df = df[df['InstanceType'] != 'g6e.12xlarge']

    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df['RequestDateTime'].min()} to {df['RequestDateTime'].max()}")

    # Step 2: Calculate success rates
    print("\n[2/6] Calculating success rates...")
    df = ca.calculate_success_rate(df)

    # Step 3: Detect fluctuations
    print("\n[3/6] Detecting fluctuation periods...")
    df = ca.detect_fluctuation_periods(df, window='1h', threshold=0.15)

    # Step 4: Find optimal windows
    print("\n[4/6] Finding optimal time windows...")
    optimal_windows = ca.find_optimal_time_windows(
        df,
        hetero_config=HETERO_CONFIG,
        homo_configs=HOMO_CONFIGS,
        window_size='1h',
        window_step='10min',
        min_fluctuation=0.15,
        max_fluctuation=0.6,
        require_recovery=True
    )

    print(f"  Found {len(optimal_windows)} optimal windows")

    # Step 5: Filter for full initial allocation
    print("\n[5/6] Filtering for full initial allocation...")
    filtered_windows = filter_full_initial_allocation(optimal_windows, df, HETERO_CONFIG)

    print(f"  Filtered to {len(filtered_windows)} windows with full initial allocation")
    print(f"  Sorted by fluctuation (highest first)")

    # Step 6: Generate outputs
    print("\n[6/6] Generating output files...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate individual trace figures (top-10 high fluctuation)
    print("\n  Generating trace figures (top-10 high fluctuation)...")
    for idx, window in enumerate(filtered_windows[:10], 1):
        generate_trace_figure(window, df, HETERO_CONFIG, idx, OUTPUT_DIR)

    # Generate individual trace figures (bottom-10 low fluctuation)
    print("\n  Generating trace figures (bottom-10 low fluctuation)...")
    num_windows = len(filtered_windows)
    for idx in range(max(0, num_windows - 10), num_windows):
        rank = idx + 1
        window = filtered_windows[idx]
        generate_trace_figure(window, df, HETERO_CONFIG, rank, OUTPUT_DIR)

    # Generate separate legend
    print("\n  Generating legend...")
    generate_legend_only(OUTPUT_DIR)

    # Generate summary table
    print("\n  Generating summary table...")
    generate_summary_table(filtered_windows, OUTPUT_DIR)

    # Save CSV files
    print("\n  Generating CSV files...")
    save_csv_files(filtered_windows, OUTPUT_DIR)

    # Save LaTeX equation
    print("\n  Generating LaTeX equation...")
    save_latex_equation(OUTPUT_DIR)

    # Summary
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"  - {min(10, len(filtered_windows))} trace figures (top-10 high fluctuation)")
    print(f"  - {min(10, len(filtered_windows))} trace figures (bottom-10 low fluctuation)")
    print(f"  - 1 legend file (PDF)")
    print(f"  - 1 summary table (PDF)")
    print(f"  - 3 CSV files (all traces, top-10, bottom-10)")
    print(f"  - 1 LaTeX equation file")
    print(f"\nTotal qualified traces: {len(filtered_windows)}")
    if filtered_windows:
        print(f"Top-1 trace (used in paper):")
        print(f"  Time: {filtered_windows[0]['time_start']} to {filtered_windows[0]['time_end']}")
        print(f"  Fluctuation: {filtered_windows[0]['avg_fluctuation']:.3f} (highest)")
        print(f"  Het Success Rate: {filtered_windows[0]['hetero_success_rate']:.2%}")
        print(f"  Avg Advantage: {filtered_windows[0]['avg_advantage']:.2%}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
