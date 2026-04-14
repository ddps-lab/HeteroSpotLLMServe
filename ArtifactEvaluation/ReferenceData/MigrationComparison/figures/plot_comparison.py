#!/usr/bin/env python3
"""
KV Cache Migration vs Re-computing Latency Comparison
Generates comparison plots for Llama 3 models (3B, 8B, 70B)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up the style for academic paper (sans-serif to match other figures)
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (3.2, 2.6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'lines.markersize': 6
})

# Get script directory for relative paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent   # raw CSVs live at ../
FIGURES_DIR = SCRIPT_DIR        # save figures next to this script

def load_data():
    """Load CSV files"""
    recomputing_df = pd.read_csv(DATA_DIR / 'recomputing.csv')
    kv_migration_df = pd.read_csv(DATA_DIR / 'kv_cache_migration.csv')
    recomputing_df = recomputing_df[recomputing_df['Context Length'] > 256]
    kv_migration_df = kv_migration_df[kv_migration_df['Context Length'] > 256]
    return recomputing_df, kv_migration_df

def get_recomputing_data(df, model_size, instance_type, scale_factor=1.0):
    """Extract re-computing data for a specific model and instance"""
    filtered = df[(df['Model Size'] == model_size) & (df['Instance Type'] == instance_type)]
    filtered = filtered.dropna(subset=['Re-Computing Latency (ms)'])
    return filtered['Context Length'].values, filtered['Re-Computing Latency (ms)'].values * scale_factor

def get_kv_migration_data(df, model_size, source_instance, dest_instance, scale_factor=1.0):
    """Extract KV migration data for a specific model and instance pair"""
    filtered = df[
        (df['Model Size'] == model_size) & 
        (df['Source Instance'] == source_instance) & 
        (df['Destination Instance'] == dest_instance)
    ]
    return filtered['Context Length'].values, filtered['KV Cache Transfer Time (ms)'].values * scale_factor

def plot_comparison(recomputing_df, kv_migration_df, model_config, output_filename, title, show_ylabel=True):
    """
    Plot comparison for a specific model configuration
    
    model_config: dict with keys:
        - recompute_l4: (model_size, instance_type, scale_factor) for L4
        - recompute_l40s: (model_size, instance_type, scale_factor) for L40S
        - kv_l4: (model_size, source, dest, scale_factor) for L4
        - kv_l40s: (model_size, source, dest, scale_factor) for L40S
    """
    fig, ax = plt.subplots()
    
    # Get re-computing data (unpack with scale_factor)
    rc_l4 = model_config['recompute_l4']
    rc_l40s = model_config['recompute_l40s']
    ctx_recomp_l4, lat_recomp_l4 = get_recomputing_data(
        recomputing_df, rc_l4[0], rc_l4[1], rc_l4[2] if len(rc_l4) > 2 else 1.0
    )
    ctx_recomp_l40s, lat_recomp_l40s = get_recomputing_data(
        recomputing_df, rc_l40s[0], rc_l40s[1], rc_l40s[2] if len(rc_l40s) > 2 else 1.0
    )
    
    # Get KV migration data (unpack with scale_factor)
    kv_l4 = model_config['kv_l4']
    kv_l40s = model_config['kv_l40s']
    ctx_kv_l4, lat_kv_l4 = get_kv_migration_data(
        kv_migration_df, kv_l4[0], kv_l4[1], kv_l4[2], kv_l4[3] if len(kv_l4) > 3 else 1.0
    )
    ctx_kv_l40s, lat_kv_l40s = get_kv_migration_data(
        kv_migration_df, kv_l40s[0], kv_l40s[1], kv_l40s[2], kv_l40s[3] if len(kv_l40s) > 3 else 1.0
    )
    
    # Plot lines (no labels here - legend will be separate)
    ax.plot(ctx_recomp_l4, lat_recomp_l4, 'o-', color='#e74c3c', linewidth=1.8, markersize=6, 
            markerfacecolor='white', markeredgewidth=1.4)
    ax.plot(ctx_recomp_l40s, lat_recomp_l40s, 's-', color='#c0392b', linewidth=1.8, markersize=6, 
            markerfacecolor='white', markeredgewidth=1.4)
    ax.plot(ctx_kv_l4, lat_kv_l4, 'o--', color='#3498db', linewidth=1.8, markersize=6, 
            markerfacecolor='white', markeredgewidth=1.4)
    ax.plot(ctx_kv_l40s, lat_kv_l40s, 's--', color='#2980b9', linewidth=1.8, markersize=6, 
            markerfacecolor='white', markeredgewidth=1.4)
    
    # Set scales and labels (no title)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Context Length')
    if show_ylabel:
        ax.set_ylabel('Latency (ms)')
    
    # Set x-axis ticks (fewer labels for compact view)
    context_lengths = [512, 2048, 8192, 32768]
    ax.set_xticks(context_lengths)
    ax.set_xticklabels(['512', '2K', '8K', '32K'])
    
    plt.tight_layout()
    
    # Save PNG and PDF
    png_path = FIGURES_DIR / f"{output_filename}.png"
    pdf_path = FIGURES_DIR / f"{output_filename}.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def create_ylabel(filename):
    """Create a separate ylabel figure"""
    fig, ax = plt.subplots(figsize=(0.4, 2.8))
    ax.text(0.5, 0.5, 'Latency (ms)', rotation=90, 
            ha='center', va='center', fontsize=16,
            transform=ax.transAxes)
    ax.axis('off')
    
    plt.tight_layout()
    
    png_path = FIGURES_DIR / f'{filename}.png'
    pdf_path = FIGURES_DIR / f'{filename}.pdf'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def create_legend(filename):
    """Create a separate legend figure"""
    fig, ax = plt.subplots(figsize=(7, 0.9))
    
    # Create dummy plots for legend
    ax.plot([], [], 'o-', color='#e74c3c', linewidth=1.8, markersize=6, 
            label='Recomputation (g6.xlarge)', markerfacecolor='white', markeredgewidth=1.4)
    ax.plot([], [], 's-', color='#c0392b', linewidth=1.8, markersize=6, 
            label='Recomputation (g6e.xlarge)', markerfacecolor='white', markeredgewidth=1.4)
    ax.plot([], [], 'o--', color='#3498db', linewidth=1.8, markersize=6, 
            label='KV Cache Transfer (g6.xlarge)', markerfacecolor='white', markeredgewidth=1.4)
    ax.plot([], [], 's--', color='#2980b9', linewidth=1.8, markersize=6, 
            label='KV Cache Transfer (g6e.xlarge)', markerfacecolor='white', markeredgewidth=1.4)
    
    ax.axis('off')
    legend = ax.legend(loc='center', ncol=2, frameon=True, framealpha=0.9, fontsize=12)
    
    plt.tight_layout()
    
    png_path = FIGURES_DIR / f'{filename}.png'
    pdf_path = FIGURES_DIR / f'{filename}.pdf'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

def main():
    # Create figures directory
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Load data
    recomputing_df, kv_migration_df = load_data()
    
    # Model configurations
    configs = {
        '3B': {
            'recompute_l4': ('3B', 'L4 (g6.xlarge)'),
            'recompute_l40s': ('3B', 'L40S (g6e.xlarge)'),
            'kv_l4': ('3B', 'g6.xlarge', 'g6.xlarge'),
            'kv_l40s': ('3B', 'g6e.xlarge', 'g6e.xlarge'),
        },
        '8B': {
            'recompute_l4': ('8B', 'L4 (g6.xlarge)'),
            'recompute_l40s': ('8B', 'L40S (g6e.xlarge)'),
            'kv_l4': ('8B', 'g6.xlarge', 'g6.xlarge'),
            'kv_l40s': ('8B', 'g6e.xlarge', 'g6e.xlarge'),
        },
        '70B': {
            # Per-layer normalization: L4 has 2 layers, L40S has 8 layers
            # Scale factor = 1/num_layers to get per-layer latency
            'recompute_l4': ('70B(2layer)', 'L4 (g6.xlarge)', 1/2),
            'recompute_l40s': ('70B(8layer)', 'L40S (g6e.xlarge)', 1/8),
            'kv_l4': ('70B', 'g6.xlarge', 'g6.xlarge', 1/2),
            'kv_l40s': ('70B', 'g6e.xlarge', 'g6e.xlarge', 1/8),
        }
    }
    
    # Generate plots - no ylabel on any (will add separately in LaTeX)
    for model_name, config in configs.items():
        suffix = '_per_layer' if model_name == '70B' else ''
        plot_comparison(
            recomputing_df,
            kv_migration_df,
            config,
            f'migration_comparison_{model_name.lower()}{suffix}',
            None,  # No title
            show_ylabel=False
        )
    
    # Create separate legend and ylabel
    create_legend('migration_comparison_legend')
    create_ylabel('migration_comparison_ylabel')
    
    print("\nAll plots generated!")

if __name__ == '__main__':
    main()
