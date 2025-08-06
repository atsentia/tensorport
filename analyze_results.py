#!/usr/bin/env python3
"""
Analyze benchmark results from Modal runs.
Creates visualizations and reports comparing JAX vs PyTorch performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

def load_results(results_file: Path) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(results_file) as f:
        return json.load(f)

def extract_metrics(results: Dict[str, Any]) -> pd.DataFrame:
    """Extract key metrics into a DataFrame for analysis."""
    rows = []
    
    for key, data in results.items():
        if "stats" in data and "error" not in data["stats"]:
            stats = data["stats"]
            gpu_type, framework = key.rsplit("_", 1)
            
            row = {
                "gpu_type": gpu_type,
                "framework": framework,
                "mean_latency": stats["inference_times"]["mean"],
                "median_latency": stats["inference_times"]["median"],
                "p95_latency": stats["inference_times"]["p95"],
                "p99_latency": stats["inference_times"]["p99"],
                "std_latency": stats["inference_times"]["std"],
                "tokens_per_second": stats["tokens_per_second"],
                "load_time": stats["load_time"],
                "warmup_time": stats["warmup_time"],
                "total_cost": stats["estimated_cost"],
                "cost_per_1k_tokens": stats["cost_per_1k_tokens"],
                "successful_inferences": stats["successful_inferences"],
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def calculate_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speedup of JAX vs PyTorch for each GPU."""
    speedup_data = []
    
    for gpu in df["gpu_type"].unique():
        gpu_df = df[df["gpu_type"] == gpu]
        
        jax_row = gpu_df[gpu_df["framework"] == "jax"]
        pytorch_row = gpu_df[gpu_df["framework"] == "pytorch"]
        
        if not jax_row.empty and not pytorch_row.empty:
            speedup = {
                "gpu_type": gpu,
                "latency_speedup": pytorch_row["mean_latency"].values[0] / jax_row["mean_latency"].values[0],
                "throughput_speedup": jax_row["tokens_per_second"].values[0] / pytorch_row["tokens_per_second"].values[0],
                "cost_reduction": (pytorch_row["cost_per_1k_tokens"].values[0] - jax_row["cost_per_1k_tokens"].values[0]) / pytorch_row["cost_per_1k_tokens"].values[0] * 100,
                "jax_tokens_per_sec": jax_row["tokens_per_second"].values[0],
                "pytorch_tokens_per_sec": pytorch_row["tokens_per_second"].values[0],
            }
            speedup_data.append(speedup)
    
    return pd.DataFrame(speedup_data)

def generate_markdown_report(df: pd.DataFrame, speedup_df: pd.DataFrame, output_file: Path):
    """Generate a markdown report with results."""
    
    report = []
    report.append("# GPT-OSS-20B Benchmark Results: JAX vs PyTorch\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    
    best_throughput = df.loc[df["tokens_per_second"].idxmax()]
    best_cost = df.loc[df["cost_per_1k_tokens"].idxmin()]
    
    report.append(f"- **Best Throughput**: {best_throughput['framework'].upper()} on {best_throughput['gpu_type']} ")
    report.append(f"({best_throughput['tokens_per_second']:.1f} tokens/sec)\n")
    report.append(f"- **Most Cost-Effective**: {best_cost['framework'].upper()} on {best_cost['gpu_type']} ")
    report.append(f"(${best_cost['cost_per_1k_tokens']:.4f}/1k tokens)\n")
    
    if not speedup_df.empty:
        avg_speedup = speedup_df["throughput_speedup"].mean()
        report.append(f"- **Average JAX Speedup**: {avg_speedup:.2f}x over PyTorch\n")
    
    # Detailed Results Table
    report.append("\n## Detailed Performance Metrics\n")
    report.append("\n| GPU | Framework | Mean Latency (s) | Tokens/sec | $/1k tokens | Load Time (s) |\n")
    report.append("|-----|-----------|------------------|------------|-------------|---------------|\n")
    
    for _, row in df.sort_values(["gpu_type", "framework"]).iterrows():
        report.append(f"| {row['gpu_type']} | {row['framework']} | "
                     f"{row['mean_latency']:.4f} | "
                     f"{row['tokens_per_second']:.1f} | "
                     f"${row['cost_per_1k_tokens']:.4f} | "
                     f"{row['load_time']:.1f} |\n")
    
    # Speedup Analysis
    if not speedup_df.empty:
        report.append("\n## JAX vs PyTorch Speedup Analysis\n")
        report.append("\n| GPU | Latency Speedup | Throughput Speedup | Cost Reduction |\n")
        report.append("|-----|-----------------|-------------------|----------------|\n")
        
        for _, row in speedup_df.iterrows():
            report.append(f"| {row['gpu_type']} | "
                         f"{row['latency_speedup']:.2f}x | "
                         f"{row['throughput_speedup']:.2f}x | "
                         f"{row['cost_reduction']:.1f}% |\n")
    
    # Latency Distribution
    report.append("\n## Latency Distribution\n")
    report.append("\n| GPU | Framework | P50 (median) | P95 | P99 | Std Dev |\n")
    report.append("|-----|-----------|--------------|-----|-----|----------|\n")
    
    for _, row in df.sort_values(["gpu_type", "framework"]).iterrows():
        report.append(f"| {row['gpu_type']} | {row['framework']} | "
                     f"{row['median_latency']:.4f} | "
                     f"{row['p95_latency']:.4f} | "
                     f"{row['p99_latency']:.4f} | "
                     f"{row['std_latency']:.4f} |\n")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    
    # Best for different use cases
    report.append("\n### By Use Case\n")
    
    # High throughput
    high_throughput = df.nlargest(1, "tokens_per_second").iloc[0]
    report.append(f"- **High Throughput Applications**: Use {high_throughput['framework'].upper()} "
                 f"on {high_throughput['gpu_type']} for {high_throughput['tokens_per_second']:.1f} tokens/sec\n")
    
    # Cost optimization
    low_cost = df.nsmallest(1, "cost_per_1k_tokens").iloc[0]
    report.append(f"- **Cost-Optimized Deployment**: Use {low_cost['framework'].upper()} "
                 f"on {low_cost['gpu_type']} at ${low_cost['cost_per_1k_tokens']:.4f}/1k tokens\n")
    
    # Low latency
    low_latency = df.nsmallest(1, "mean_latency").iloc[0]
    report.append(f"- **Low Latency Requirements**: Use {low_latency['framework'].upper()} "
                 f"on {low_latency['gpu_type']} with {low_latency['mean_latency']:.4f}s mean latency\n")
    
    # Cost-Performance Analysis
    report.append("\n### Cost-Performance Analysis\n")
    
    # Calculate efficiency score (tokens per dollar)
    df["tokens_per_dollar"] = 1000 / df["cost_per_1k_tokens"]
    
    report.append("\n| Configuration | Tokens per Dollar | Relative Efficiency |\n")
    report.append("|---------------|-------------------|--------------------|\n")
    
    max_efficiency = df["tokens_per_dollar"].max()
    for _, row in df.sort_values("tokens_per_dollar", ascending=False).iterrows():
        relative_eff = (row["tokens_per_dollar"] / max_efficiency) * 100
        report.append(f"| {row['gpu_type']} + {row['framework']} | "
                     f"{row['tokens_per_dollar']:.0f} | "
                     f"{relative_eff:.1f}% |\n")
    
    # Key Insights
    report.append("\n## Key Insights\n")
    
    if not speedup_df.empty:
        if speedup_df["throughput_speedup"].mean() > 1.5:
            report.append("1. **JAX consistently outperforms PyTorch** across all tested GPU types\n")
        elif speedup_df["throughput_speedup"].mean() > 1.0:
            report.append("1. **JAX shows moderate performance gains** over PyTorch\n")
        else:
            report.append("1. **Performance parity** between JAX and PyTorch implementations\n")
    
    # Load time comparison
    jax_load_avg = df[df["framework"] == "jax"]["load_time"].mean()
    pytorch_load_avg = df[df["framework"] == "pytorch"]["load_time"].mean()
    
    if jax_load_avg < pytorch_load_avg * 0.5:
        report.append("2. **JAX model loading is significantly faster** than PyTorch\n")
    else:
        report.append("2. **Model loading times are comparable** between frameworks\n")
    
    # GPU scaling
    if len(df["gpu_type"].unique()) > 2:
        gpu_scaling = df.groupby("gpu_type")["tokens_per_second"].max()
        if gpu_scaling.max() / gpu_scaling.min() > 3:
            report.append("3. **Strong GPU scaling observed** - higher-tier GPUs provide proportional benefits\n")
        else:
            report.append("3. **Limited GPU scaling** - consider cost-optimized GPUs for this workload\n")
    
    # Write report
    with open(output_file, "w") as f:
        f.write("".join(report))
    
    print(f"Report saved to: {output_file}")

def generate_csv_export(df: pd.DataFrame, output_file: Path):
    """Export results to CSV for further analysis."""
    df.to_csv(output_file, index=False)
    print(f"CSV export saved to: {output_file}")

def print_summary(df: pd.DataFrame):
    """Print a summary to console."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print("\n📊 Performance Rankings (by throughput):")
    ranked = df.sort_values("tokens_per_second", ascending=False)
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"{i}. {row['gpu_type']:12} + {row['framework']:8} : "
              f"{row['tokens_per_second']:8.1f} tok/s | "
              f"${row['cost_per_1k_tokens']:.4f}/1k")
    
    print("\n💰 Cost Efficiency Rankings:")
    cost_ranked = df.sort_values("cost_per_1k_tokens")
    for i, (_, row) in enumerate(cost_ranked.iterrows(), 1):
        print(f"{i}. {row['gpu_type']:12} + {row['framework']:8} : "
              f"${row['cost_per_1k_tokens']:.4f}/1k tokens")
    
    print("\n⚡ Latency Rankings (lower is better):")
    latency_ranked = df.sort_values("mean_latency")
    for i, (_, row) in enumerate(latency_ranked.iterrows(), 1):
        print(f"{i}. {row['gpu_type']:12} + {row['framework']:8} : "
              f"{row['mean_latency']:.4f}s mean | "
              f"{row['p99_latency']:.4f}s p99")

def main():
    parser = argparse.ArgumentParser(description="Analyze Modal benchmark results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", default=".", help="Output directory for reports")
    parser.add_argument("--format", default="all", choices=["markdown", "csv", "all"],
                       help="Output format for report")
    
    args = parser.parse_args()
    
    # Load results
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    results = load_results(results_file)
    
    # Extract metrics
    df = extract_metrics(results)
    if df.empty:
        print("Error: No valid results found in file")
        return
    
    # Calculate speedup
    speedup_df = calculate_speedup(df)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.format in ["markdown", "all"]:
        markdown_file = output_dir / f"benchmark_report_{timestamp}.md"
        generate_markdown_report(df, speedup_df, markdown_file)
    
    if args.format in ["csv", "all"]:
        csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
        generate_csv_export(df, csv_file)
    
    # Print summary
    print_summary(df)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()