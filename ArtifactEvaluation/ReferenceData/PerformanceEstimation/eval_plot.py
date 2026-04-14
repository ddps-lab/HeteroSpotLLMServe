"""
Generate evaluation plots for ShuntServe performance estimation.
Compares roofline estimator predictions against TRT-LLM gptManagerBenchmark.
"""
import json, glob, os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13

BASE = os.path.dirname(os.path.abspath(__file__))
WORKLOAD = 'in763-out232'
PRED_DIR = os.path.join(BASE, 'trtllm', 'llama3-70b', WORKLOAD, 'predicted')
MEAS_ROOT = os.path.join(BASE, 'trtllm', 'llama3-70b', WORKLOAD, 'measured')
OUT_DIR = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

INSTANCE_LABELS = {
    'g5-48xlarge': 'g5 (A10G)',
    'g6-48xlarge': 'g6 (L4)',
    'g6e-48xlarge': 'g6e (L40S)',
    'p4d-24xlarge': 'p4d (A100)',
}

# ── Data Loading ──────────────────────────────────────────────────

def parse_gptbench_log(filepath):
    metrics = {}
    with open(filepath) as f:
        for line in f:
            if not line.strip().startswith('[BENCHMARK]'): continue
            parts = line.replace('[BENCHMARK] ', '').strip()
            tokens = parts.rsplit(' ', 1)
            if len(tokens) != 2: continue
            key = re.sub(r'\(.*?\)', '', tokens[0]).strip()
            try: metrics[key] = float(tokens[1])
            except: pass
    return metrics

def load_all():
    pred_rows = []
    for f in sorted(glob.glob(os.path.join(PRED_DIR, 'est_*.json'))):
        d = json.load(open(f))
        if not d.get('feasible'): continue
        inst = d['instance_type'].replace('.', '-')
        for e in d.get('batch_sweep', []):
            pred_rows.append({
                'instance': inst, 'tp': d['tp_size'], 'pp': d['pp_size'],
                'batch': e['batch_size'],
                'e2e_est': e['batch_latency_ms'] / 1000.0,
                'ttft_est': e.get('ttft_ms', 0) / 1000.0,
                'rps_est': e['throughput_rps'],
            })
    pred_df = pd.DataFrame(pred_rows)

    meas_rows = []
    for inst_dir in sorted(os.listdir(MEAS_ROOT)):
        log_dir = os.path.join(MEAS_ROOT, inst_dir, 'gptBench', 'log')
        if not os.path.isdir(log_dir): continue
        for fp in sorted(glob.glob(os.path.join(log_dir, 'trtllm_*.log'))):
            m = re.match(r'trtllm_tp(\d+)_pp(\d+)_bs(\d+)\.log', os.path.basename(fp))
            if not m: continue
            metrics = parse_gptbench_log(fp)
            if not metrics or metrics.get('num_samples', 0) == 0: continue
            ns = int(metrics.get('num_samples', 0))
            total_lat = metrics.get('total_latency', 0)
            meas_rows.append({
                'instance': inst_dir, 'tp': int(m.group(1)), 'pp': int(m.group(2)),
                'batch': int(m.group(3)),
                'e2e_meas': metrics.get('avg_sequence_latency', 0) / 1000.0,
                'ttft_meas': metrics.get('avg_time_to_first_token', 0) / 1000.0,
                'rps_meas': ns / (total_lat / 1000) if total_lat > 0 else 0,
            })
    meas_df = pd.DataFrame(meas_rows)

    merged = pd.merge(pred_df, meas_df, on=['instance', 'tp', 'pp', 'batch'])
    merged['config'] = merged.apply(lambda r: f"TP{r['tp']}PP{r['pp']}", axis=1)
    merged['e2e_mape'] = ((merged['e2e_est'] - merged['e2e_meas']).abs() / merged['e2e_meas'].clip(lower=1e-9) * 100)
    merged['rps_mape'] = ((merged['rps_est'] - merged['rps_meas']).abs() / merged['rps_meas'].clip(lower=1e-9) * 100)
    merged['ttft_mape'] = ((merged['ttft_est'] - merged['ttft_meas']).abs() / merged['ttft_meas'].clip(lower=1e-9) * 100)
    return merged

# ── Plot 1: Per-instance, per-config E2E scatter (Est vs Meas) ────

def plot_scatter_by_instance(df):
    instances = sorted(df['instance'].unique())
    fig, axes = plt.subplots(1, len(instances), figsize=(5 * len(instances), 4.5), sharey=False)
    if len(instances) == 1: axes = [axes]
    
    configs = sorted(df['config'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    config_color = dict(zip(configs, colors))
    markers = {'TP1PP8': 'o', 'TP2PP4': 's', 'TP4PP2': '^', 'TP8PP1': 'D'}

    for ax, inst in zip(axes, instances):
        sub = df[df['instance'] == inst]
        max_val = max(sub['e2e_est'].max(), sub['e2e_meas'].max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, linewidth=1, label='y=x')
        
        for cfg in configs:
            csub = sub[sub['config'] == cfg]
            if csub.empty: continue
            ax.scatter(csub['e2e_meas'], csub['e2e_est'],
                       c=[config_color[cfg]], marker=markers.get(cfg, 'o'),
                       s=50, alpha=0.8, edgecolors='k', linewidths=0.5,
                       label=cfg)
        
        mape = sub['e2e_mape'].mean()
        ax.set_title(f"{INSTANCE_LABELS.get(inst, inst)}\nMAPE: {mape:.1f}%")
        ax.set_xlabel('Measured E2E (s)')
        ax.set_ylabel('Estimated E2E (s)')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Estimated vs Measured E2E Latency (TRT-LLM, Llama 3.1 70B)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'scatter_e2e_by_instance.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT_DIR, 'scatter_e2e_by_instance.png'), bbox_inches='tight', dpi=150)
    print(f"Saved scatter_e2e_by_instance")

# ── Plot 2: Per-config bar chart (Est vs Meas across batch sizes) ──

def plot_bars_by_config(df):
    instances = sorted(df['instance'].unique())
    configs = sorted(df['config'].unique())
    
    fig, axes = plt.subplots(len(instances), len(configs), 
                              figsize=(4.5 * len(configs), 3.5 * len(instances)),
                              squeeze=False)
    
    for i, inst in enumerate(instances):
        for j, cfg in enumerate(configs):
            ax = axes[i][j]
            sub = df[(df['instance'] == inst) & (df['config'] == cfg)].sort_values('batch')
            if sub.empty:
                ax.set_visible(False)
                continue
            
            x = np.arange(len(sub))
            w = 0.35
            bars_meas = ax.bar(x - w/2, sub['e2e_meas'], w, label='Measured', color='#4C72B0', alpha=0.85)
            bars_est = ax.bar(x + w/2, sub['e2e_est'], w, label='Estimated', color='#DD8452', alpha=0.85)
            
            ax.set_xticks(x)
            ax.set_xticklabels(sub['batch'].values, fontsize=8)
            ax.set_xlabel('Batch Size')
            if j == 0:
                ax.set_ylabel(f'{INSTANCE_LABELS.get(inst, inst)}\nE2E Latency (s)')
            if i == 0:
                ax.set_title(cfg)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.2, axis='y')
            
            # Add MAPE annotation
            mape = sub['e2e_mape'].mean()
            ax.annotate(f'MAPE: {mape:.1f}%', xy=(0.98, 0.95), xycoords='axes fraction',
                       ha='right', va='top', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    fig.suptitle('E2E Latency: Estimated vs Measured (TRT-LLM, Llama 3.1 70B)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'bars_e2e_by_config.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT_DIR, 'bars_e2e_by_config.png'), bbox_inches='tight', dpi=150)
    print(f"Saved bars_e2e_by_config")

# ── Plot 3: MAPE summary bar chart ──────────────────────────────

def plot_mape_summary(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # By Instance
    ax = axes[0]
    inst_mape = df.groupby('instance')['e2e_mape'].mean().sort_values()
    labels = [INSTANCE_LABELS.get(k, k) for k in inst_mape.index]
    bars = ax.barh(labels, inst_mape.values, color='#4C72B0', alpha=0.85)
    ax.set_xlabel('E2E MAPE (%)')
    ax.set_title('By Instance')
    for bar, v in zip(bars, inst_mape.values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=9)
    ax.set_xlim(0, inst_mape.max() * 1.3)
    ax.grid(True, alpha=0.3, axis='x')
    
    # By Parallelism
    ax = axes[1]
    cfg_mape = df.groupby('config')['e2e_mape'].mean().sort_values()
    bars = ax.barh(cfg_mape.index, cfg_mape.values, color='#DD8452', alpha=0.85)
    ax.set_xlabel('E2E MAPE (%)')
    ax.set_title('By Parallelism Strategy')
    for bar, v in zip(bars, cfg_mape.values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=9)
    ax.set_xlim(0, cfg_mape.max() * 1.3)
    ax.grid(True, alpha=0.3, axis='x')
    
    # By Batch Size
    ax = axes[2]
    bs_mape = df.groupby('batch')['e2e_mape'].mean()
    ax.bar(range(len(bs_mape)), bs_mape.values, color='#55A868', alpha=0.85)
    ax.set_xticks(range(len(bs_mape)))
    ax.set_xticklabels(bs_mape.index, rotation=45, fontsize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('E2E MAPE (%)')
    ax.set_title('By Batch Size')
    ax.grid(True, alpha=0.3, axis='y')
    
    overall_mape = df['e2e_mape'].mean()
    fig.suptitle(f'E2E Estimation Accuracy — Overall MAPE: {overall_mape:.1f}% (n={len(df)})', fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'mape_summary.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT_DIR, 'mape_summary.png'), bbox_inches='tight', dpi=150)
    print(f"Saved mape_summary")

# ── Plot 4: Line chart — Est vs Meas E2E across batch sizes per instance ──

def plot_lines_e2e(df):
    instances = sorted(df['instance'].unique())
    fig, axes = plt.subplots(1, len(instances), figsize=(5 * len(instances), 4), sharey=False)
    if len(instances) == 1: axes = [axes]
    
    configs = sorted(df['config'].unique())
    colors = {'TP1PP8': '#1f77b4', 'TP2PP4': '#ff7f0e', 'TP4PP2': '#2ca02c', 'TP8PP1': '#d62728'}
    
    for ax, inst in zip(axes, instances):
        sub = df[df['instance'] == inst]
        for cfg in configs:
            csub = sub[sub['config'] == cfg].sort_values('batch')
            if csub.empty: continue
            color = colors.get(cfg, '#333')
            ax.plot(csub['batch'], csub['e2e_meas'], 'o-', color=color, label=f'{cfg} meas', linewidth=1.5, markersize=5)
            ax.plot(csub['batch'], csub['e2e_est'], 's--', color=color, label=f'{cfg} est', linewidth=1.5, markersize=4, alpha=0.7)
        
        mape = sub['e2e_mape'].mean()
        ax.set_title(f"{INSTANCE_LABELS.get(inst, inst)} (MAPE: {mape:.1f}%)")
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('E2E Latency (s)')
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=6, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('E2E Latency Scaling: Estimated vs Measured', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lines_e2e.pdf'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUT_DIR, 'lines_e2e.png'), bbox_inches='tight', dpi=150)
    print(f"Saved lines_e2e")

# ── Plot 5: Table ─────────────────────────────────────────────────

def print_table(df):
    """Print a LaTeX-ready table of MAPE by instance × config."""
    configs = sorted(df['config'].unique())
    instances = sorted(df['instance'].unique())
    
    print("\n=== MAPE Table (Instance × Parallelism) ===")
    header = f"{'Instance':<20}" + "".join(f"{c:>10}" for c in configs) + f"{'Overall':>10}"
    print(header)
    print("-" * len(header))
    for inst in instances:
        row = f"{INSTANCE_LABELS.get(inst, inst):<20}"
        for cfg in configs:
            sub = df[(df['instance'] == inst) & (df['config'] == cfg)]
            if sub.empty:
                row += f"{'—':>10}"
            else:
                row += f"{sub['e2e_mape'].mean():>9.1f}%"
        isub = df[df['instance'] == inst]
        row += f"{isub['e2e_mape'].mean():>9.1f}%"
        print(row)
    
    row = f"{'Overall':<20}"
    for cfg in configs:
        sub = df[df['config'] == cfg]
        row += f"{sub['e2e_mape'].mean():>9.1f}%"
    row += f"{df['e2e_mape'].mean():>9.1f}%"
    print(row)

# ── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_all()
    print(f"Total data points: {len(df)}")
    print(f"Instances: {sorted(df['instance'].unique())}")
    print(f"Configs: {sorted(df['config'].unique())}")
    print(f"Batch sizes: {sorted(df['batch'].unique())}")
    print()
    
    print_table(df)
    print()
    
    plot_scatter_by_instance(df)
    plot_bars_by_config(df)
    plot_mape_summary(df)
    plot_lines_e2e(df)
    
    print(f"\nAll figures saved to {OUT_DIR}/")
