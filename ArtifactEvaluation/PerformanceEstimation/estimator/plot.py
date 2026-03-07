"""
Plot predicted vs measured throughput for performance estimation accuracy.
Generates normalized throughput comparison figure.

Usage:
    python plot.py
    python plot.py --output estimation_accuracy.pdf
"""
import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12


def load_results(results_dir="results"):
    """Load all predicted and measured result files."""
    predicted = {}
    measured = {}

    for f in glob.glob(os.path.join(results_dir, "predicted_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        key = f"{data['model'].split('/')[-1]}_{data['instance']}"
        predicted[key] = data

    for f in glob.glob(os.path.join(results_dir, "measured_*.json")):
        with open(f) as fh:
            data = json.load(fh)
        key = data["config"]
        measured[key] = data

    return predicted, measured


def normalize_throughput(results, key="throughput_rps"):
    """Normalize throughput relative to batch_size=1."""
    if not results:
        return [], []

    base = None
    for r in results:
        val = r.get(key)
        if val and r["batch_size"] == 1:
            base = val
            break

    if base is None or base == 0:
        # Fallback: use first valid value
        for r in results:
            val = r.get(key)
            if val and val > 0:
                base = val
                break

    if base is None or base == 0:
        return [], []

    batch_sizes = []
    normalized = []
    for r in results:
        val = r.get(key)
        if val and val > 0:
            batch_sizes.append(r["batch_size"])
            normalized.append(val / base)

    return batch_sizes, normalized


def plot_comparison(predicted, measured, output_path="estimation_accuracy.pdf"):
    """
    Create a figure with subplots for each configuration.
    Each subplot shows normalized predicted vs measured throughput.
    """
    # Match configs
    # predicted keys: "Llama-3.1-70B-Instruct_g6e.12xlarge"
    # measured keys: "70B_L40S", "8B_L4"
    config_mapping = {
        "70B_L40S": {
            "pred_key_pattern": "70B",
            "title": "(a) Llama-3.1-70B on L40S×4",
        },
        "8B_L4": {
            "pred_key_pattern": "8B",
            "title": "(b) Llama-3.1-8B on L4×1",
        },
    }

    # Find matched pairs
    pairs = []
    for meas_key, meas_data in measured.items():
        mapping = config_mapping.get(meas_key)
        if not mapping:
            continue

        # Find matching predicted
        pred_data = None
        for pk, pv in predicted.items():
            if mapping["pred_key_pattern"] in pk:
                pred_data = pv
                break

        if pred_data:
            pairs.append((mapping["title"], pred_data, meas_data))

    if not pairs:
        print("No matched predicted/measured pairs found.")
        print(f"Predicted keys: {list(predicted.keys())}")
        print(f"Measured keys: {list(measured.keys())}")
        return

    num_plots = len(pairs)
    fig, axes = plt.subplots(1, num_plots, figsize=(5.5 * num_plots, 4.5))
    if num_plots == 1:
        axes = [axes]

    for ax, (title, pred_data, meas_data) in zip(axes, pairs):
        # Normalize
        pred_bs, pred_norm = normalize_throughput(pred_data["results"])
        meas_bs, meas_norm = normalize_throughput(meas_data["results"])

        ax.plot(pred_bs, pred_norm, 'o-', color='#2196F3', linewidth=2,
                markersize=7, label='Predicted', zorder=3)
        ax.plot(meas_bs, meas_norm, 's--', color='#FF5722', linewidth=2,
                markersize=7, label='Measured', zorder=3)

        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Normalized Throughput')
        ax.set_title(title, fontsize=13, pad=10)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

        # Calculate normalized MAPE for overlapping batch sizes
        common_bs = set(pred_bs) & set(meas_bs)
        if common_bs:
            pred_dict = dict(zip(pred_bs, pred_norm))
            meas_dict = dict(zip(meas_bs, meas_norm))
            errors = []
            for bs in sorted(common_bs):
                if meas_dict[bs] > 0:
                    err = abs(pred_dict[bs] - meas_dict[bs]) / meas_dict[bs] * 100
                    errors.append(err)
            if errors:
                mape = np.mean(errors)
                ax.text(0.97, 0.05, f'Norm. MAPE: {mape:.1f}%',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                               facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()


def plot_predicted_only(predicted, output_path="estimation_predicted_only.pdf"):
    """
    If measured data is not yet available, plot predicted values only.
    Useful for previewing before running actual experiments.
    """
    fig, axes = plt.subplots(1, len(predicted), figsize=(5.5 * len(predicted), 4.5))
    if len(predicted) == 1:
        axes = [axes]

    for ax, (key, data) in zip(axes, predicted.items()):
        batch_sizes = [r["batch_size"] for r in data["results"]]
        throughputs = [r["throughput_rps"] for r in data["results"]]

        ax.plot(batch_sizes, throughputs, 'o-', color='#2196F3', linewidth=2, markersize=7)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Predicted Throughput (req/s)')
        ax.set_title(f'{data["model"].split("/")[-1]}\non {data["instance"]}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Predicted-only figure saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="estimation_accuracy.pdf")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    predicted, measured = load_results(args.results_dir)

    print(f"Found {len(predicted)} predicted result(s)")
    print(f"Found {len(measured)} measured result(s)")

    if predicted and measured:
        plot_comparison(predicted, measured, args.output)
    elif predicted:
        print("No measured data found. Plotting predicted values only.")
        plot_predicted_only(predicted)
    else:
        print("No results found. Run predict.py and measure.py first.")


if __name__ == "__main__":
    main()
