#!/usr/bin/env python3
"""
Visualization script for comparing accuracy across different system prompts and benchmarks.
This script reads JSON result files and creates visualizations comparing performance.
"""

import json
import os
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Set style for better-looking plots
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def calculate_accuracy(json_file):
    """
    Calculate accuracy from a JSON result file.

    Args:
        json_file (str): Path to the JSON file

    Returns:
        tuple: (accuracy, total_samples, correct_samples)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        responses = data.get('responses', [])
        if not responses:
            return None, 0, 0

        total_samples = len(responses)
        correct_samples = sum(1 for r in responses if r.get('is_correct') == 1 or r.get('is_correct') == True)

        accuracy = (correct_samples / total_samples * 100) if total_samples > 0 else 0

        return accuracy, total_samples, correct_samples
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None, 0, 0


def parse_result_path(filepath, results_dir):
    """
    Parse the result file path to extract metadata.

    Args:
        filepath (str): Full path to the result file
        results_dir (str): Base results directory

    Returns:
        dict: Metadata containing benchmark, model, max_tokens, and system_prompt
    """
    # Get relative path from results directory
    rel_path = os.path.relpath(filepath, results_dir)
    parts = rel_path.split(os.sep)

    if len(parts) >= 4:
        benchmark = parts[0]
        model = parts[1]
        max_tokens = parts[2]
        system_prompt = os.path.splitext(parts[3])[0]  # Remove .json extension

        return {
            'benchmark': benchmark,
            'model': model,
            'max_tokens': max_tokens,
            'system_prompt': system_prompt,
            'filepath': filepath
        }
    return None


def collect_all_results(results_dir):
    """
    Collect all result files and compute their accuracies.

    Args:
        results_dir (str): Path to results directory

    Returns:
        pd.DataFrame: DataFrame with all results
    """
    json_files = glob.glob(os.path.join(results_dir, '**/*.json'), recursive=True)

    results = []
    for json_file in json_files:
        metadata = parse_result_path(json_file, results_dir)
        if metadata:
            accuracy, total, correct = calculate_accuracy(json_file)
            if accuracy is not None:
                results.append({
                    'benchmark': metadata['benchmark'],
                    'model': metadata['model'],
                    'max_tokens': metadata['max_tokens'],
                    'system_prompt': metadata['system_prompt'],
                    'accuracy': accuracy,
                    'total_samples': total,
                    'correct_samples': correct,
                    'filepath': metadata['filepath']
                })

    return pd.DataFrame(results)


def plot_system_prompt_comparison(df, output_dir='visualizations'):
    """
    Create bar plots comparing accuracy across different system prompts.

    Args:
        df (pd.DataFrame): Results dataframe
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: System prompts comparison across all benchmarks
    fig, ax = plt.subplots(figsize=(14, 8))

    # Group by benchmark and system_prompt
    grouped = df.groupby(['benchmark', 'system_prompt'])['accuracy'].mean().reset_index()

    # Create grouped bar chart
    benchmarks = grouped['benchmark'].unique()
    system_prompts = grouped['system_prompt'].unique()

    x = np.arange(len(benchmarks))
    width = 0.25

    for i, prompt in enumerate(system_prompts):
        prompt_data = grouped[grouped['system_prompt'] == prompt]
        accuracies = [prompt_data[prompt_data['benchmark'] == b]['accuracy'].values[0]
                     if len(prompt_data[prompt_data['benchmark'] == b]) > 0 else 0
                     for b in benchmarks]
        ax.bar(x + i * width, accuracies, width, label=prompt)

    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison Across System Prompts by Benchmark', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(benchmarks, rotation=15, ha='right')
    ax.legend(title='System Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_prompt_comparison_by_benchmark.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {os.path.join(output_dir, 'system_prompt_comparison_by_benchmark.pdf')}")


def plot_heatmap(df, output_dir='visualizations'):
    """
    Create heatmap showing accuracy for each combination of benchmark and system prompt.

    Args:
        df (pd.DataFrame): Results dataframe
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create pivot table
    pivot_data = df.groupby(['benchmark', 'system_prompt'])['accuracy'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='system_prompt', columns='benchmark', values='accuracy')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        import seaborn as sns
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
    except ImportError:
        # Fallback to matplotlib imshow if seaborn not available
        im = ax.imshow(pivot_table.values, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns, rotation=15, ha='right')
        ax.set_yticklabels(pivot_table.index)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                           color="black" if value > 50 else "white", fontweight='bold')

    ax.set_title('Accuracy Heatmap: System Prompts vs Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('System Prompt', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {os.path.join(output_dir, 'accuracy_heatmap.pdf')}")


def plot_model_comparison(df, output_dir='visualizations'):
    """
    Create plots comparing different models across system prompts.

    Args:
        df (pd.DataFrame): Results dataframe
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # For each benchmark, create a plot
    for benchmark in df['benchmark'].unique():
        benchmark_df = df[df['benchmark'] == benchmark]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group by model and system_prompt
        models = benchmark_df['model'].unique()
        system_prompts = benchmark_df['system_prompt'].unique()

        x = np.arange(len(system_prompts))
        width = 0.8 / len(models)

        for i, model in enumerate(models):
            model_data = benchmark_df[benchmark_df['model'] == model]
            accuracies = [model_data[model_data['system_prompt'] == sp]['accuracy'].values[0]
                         if len(model_data[model_data['system_prompt'] == sp]) > 0 else 0
                         for sp in system_prompts]
            ax.bar(x + i * width, accuracies, width, label=model)

        ax.set_xlabel('System Prompt', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison on {benchmark}', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(system_prompts, rotation=15, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f'model_comparison_{benchmark.replace(" ", "_")}.pdf'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {os.path.join(output_dir, filename)}")


def plot_max_tokens_effect(df, output_dir='visualizations'):
    """
    Create plots showing the effect of max_tokens on accuracy.

    Args:
        df (pd.DataFrame): Results dataframe
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if there are multiple max_tokens values
    if len(df['max_tokens'].unique()) <= 1:
        print("Skipping max_tokens plot - only one max_tokens value found")
        return

    for benchmark in df['benchmark'].unique():
        benchmark_df = df[df['benchmark'] == benchmark]

        fig, ax = plt.subplots(figsize=(12, 6))

        for prompt in benchmark_df['system_prompt'].unique():
            prompt_data = benchmark_df[benchmark_df['system_prompt'] == prompt]
            grouped = prompt_data.groupby('max_tokens')['accuracy'].mean().reset_index()
            grouped = grouped.sort_values('max_tokens')
            ax.plot(grouped['max_tokens'], grouped['accuracy'], marker='o', label=prompt, linewidth=2)

        ax.set_xlabel('Max Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Effect of Max Tokens on Accuracy - {benchmark}', fontsize=14, fontweight='bold')
        ax.legend(title='System Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        filename = f'max_tokens_effect_{benchmark.replace(" ", "_")}.pdf'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {os.path.join(output_dir, filename)}")


def print_summary_table(df):
    """
    Print a summary table of all results.

    Args:
        df (pd.DataFrame): Results dataframe
    """
    print("\n" + "="*100)
    print("SUMMARY TABLE: Accuracy by Benchmark and System Prompt")
    print("="*100)

    # Group by benchmark and system_prompt
    summary = df.groupby(['benchmark', 'system_prompt']).agg({
        'accuracy': 'mean',
        'total_samples': 'first',
        'correct_samples': 'sum'
    }).reset_index()

    for benchmark in summary['benchmark'].unique():
        print(f"\n{benchmark}")
        print("-" * 100)
        benchmark_data = summary[summary['benchmark'] == benchmark]
        print(f"{'System Prompt':<20} {'Accuracy (%)':<15} {'Correct Samples':<20} {'Total Samples':<15}")
        print("-" * 100)

        for _, row in benchmark_data.iterrows():
            print(f"{row['system_prompt']:<20} {row['accuracy']:>12.2f}    "
                  f"{int(row['correct_samples']):>15}    {int(row['total_samples']):>12}")

    print("\n" + "="*100)
    print("OVERALL SUMMARY: Average Accuracy by System Prompt (across all benchmarks)")
    print("="*100)

    overall = df.groupby('system_prompt')['accuracy'].mean().reset_index()
    overall = overall.sort_values('accuracy', ascending=False)

    print(f"{'System Prompt':<20} {'Average Accuracy (%)':<25}")
    print("-" * 100)
    for _, row in overall.iterrows():
        print(f"{row['system_prompt']:<20} {row['accuracy']:>20.2f}")

    print("="*100 + "\n")


def main():
    """Main function to run all visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize accuracy results across different system prompts and benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help="Filter results by model name (e.g., 'qwen3_8B' or 'qwen3_8B_base'). If not specified, includes all models."
    )

    parser.add_argument(
        '--benchmark', '-b',
        type=str,
        default=None,
        help="Filter results by benchmark name (e.g., 'aime_2025_default', 'math_500_test', 'arc_challenge_test'). If not specified, includes all benchmarks."
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='visualizations',
        help="Output directory for visualizations (default: visualizations)"
    )

    args = parser.parse_args()

    # Set results directory
    results_dir = 'results'

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found!")
        return

    print("Collecting results from all JSON files...")
    df = collect_all_results(results_dir)

    if df.empty:
        print("No results found!")
        return

    # Apply filters
    if args.model:
        original_len = len(df)
        df = df[df['model'] == args.model]
        print(f"Filtered by model '{args.model}': {original_len} -> {len(df)} results")
        if df.empty:
            print(f"No results found for model '{args.model}'")
            print(f"Available models: {collect_all_results(results_dir)['model'].unique()}")
            return

    if args.benchmark:
        original_len = len(df)
        df = df[df['benchmark'] == args.benchmark]
        print(f"Filtered by benchmark '{args.benchmark}': {original_len} -> {len(df)} results")
        if df.empty:
            print(f"No results found for benchmark '{args.benchmark}'")
            print(f"Available benchmarks: {collect_all_results(results_dir)['benchmark'].unique()}")
            return

    print(f"\nFound {len(df)} result files")
    print(f"Benchmarks: {df['benchmark'].unique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"System Prompts: {df['system_prompt'].unique()}")
    print(f"Max Tokens: {df['max_tokens'].unique()}")

    # Print summary table
    print_summary_table(df)

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating visualizations...")

    # Generate all plots
    plot_system_prompt_comparison(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_model_comparison(df, output_dir)
    plot_max_tokens_effect(df, output_dir)

    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to: {csv_path}")

    print(f"\nAll visualizations saved to '{output_dir}/' directory")
    print("Done!")


if __name__ == "__main__":
    main()
