"""
Visualization utilities for Azure LLM Inference Conversation Dataset.
Creates figures showing request arrival patterns.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


def plot_request_distribution(
    csv_path: str,
    output_path: str = None,
    max_requests: int = None,
    figsize: tuple = (12, 5),
    title_fontsize: int = 18,
    subtitle_fontsize: int = 16,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    tick_fontsize: int = 14
):
    """
    Create a figure with two subplots:
    (a) Length distribution - histogram showing input/output token lengths
    (b) Arrival rate - arrival rate over time

    Args:
        csv_path: Path to the Azure trace CSV file
        output_path: Path to save the figure (default: same directory as CSV)
        max_requests: Maximum number of requests to load (None for all)
        figsize: Figure size (width, height)
        title_fontsize: Font size for main title
        subtitle_fontsize: Font size for subplot titles
        label_fontsize: Font size for axis labels
        legend_fontsize: Font size for legend
        tick_fontsize: Font size for axis tick labels

    Returns:
        str: Path to the saved figure
    """
    # Load full trace data
    trace_data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_requests is not None and i >= max_requests:
                break
            timestamp_str = row['TIMESTAMP']
            # Trace timestamps carry 7-digit fractional seconds (.NET DateTime
            # ticks); strip to 6 digits so datetime.fromisoformat accepts them
            # on Python < 3.11.
            if '.' in timestamp_str:
                head, _, frac = timestamp_str.partition('.')
                timestamp_str = f"{head}.{frac[:6]}"
            timestamp = datetime.fromisoformat(timestamp_str)
            context_tokens = int(row['ContextTokens'])
            generated_tokens = int(row['GeneratedTokens'])
            trace_data.append((timestamp, context_tokens, generated_tokens))

    if not trace_data:
        raise ValueError("No data loaded from CSV")

    # Extract data
    timestamps = [t[0] for t in trace_data]
    context_tokens = [t[1] for t in trace_data]
    generated_tokens = [t[2] for t in trace_data]

    first_timestamp = timestamps[0]
    arrival_times = [(ts - first_timestamp).total_seconds() for ts in timestamps]

    # Calculate arrival rates
    max_time = int(max(arrival_times)) + 1
    request_counts = [0] * max_time
    for t in arrival_times:
        second = int(t)
        if second < max_time:
            request_counts[second] += 1

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # (a) Length distribution - histogram with input and output lengths
    bins = 100
    ax1.hist(context_tokens, bins=bins, alpha=0.7, label='Input length', color='#1f77b4')
    ax1.hist(generated_tokens, bins=bins, alpha=0.7, label='Output length', color='#ff7f0e')
    ax1.set_xlabel('Length', fontsize=label_fontsize)
    ax1.set_ylabel('Num of Requests', fontsize=label_fontsize)
    ax1.set_title('Request Length distribution.', fontsize=subtitle_fontsize, loc='center', fontweight='bold')
    ax1.legend(fontsize=legend_fontsize, loc='upper right')
    ax1.tick_params(labelsize=tick_fontsize)

    # (b) Arrival rate over time
    time_axis = np.arange(len(request_counts))
    ax2.plot(time_axis, request_counts, linewidth=0.5, color='#1f77b4')
    ax2.set_xlabel('Time (s)', fontsize=label_fontsize)
    ax2.set_ylabel('Arrival Rate (Req./s)', fontsize=label_fontsize)
    ax2.set_title('Request Arrival Rate.', fontsize=subtitle_fontsize, loc='center', fontweight='bold')
    ax2.tick_params(labelsize=tick_fontsize)
    ax2.set_xlim(0, max_time)

    # Main title
    fig.suptitle(f'Statistics of {Path(csv_path).stem} dataset.', fontsize=title_fontsize, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_distribution.png"

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Distribution figure saved to: {output_path}\n")

    return str(output_path)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to the dataset at ArtifactEvaluation/Datasets/<trace>.csv
        current_dir = Path(__file__).parent
        csv_path = current_dir.parents[2] / "Datasets" / "AzureLLMInferenceConvTrace_pruned_2048.csv"

    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    print(f"Analyzing: {csv_path}\n")

    # Save next to this script regardless of where the CSV lives.
    output_path = Path(__file__).parent / f"{Path(csv_path).stem}_distribution.png"
    plot_request_distribution(csv_path, output_path=str(output_path))
