"""
AO-PQ Multi-Epoch Window Latency Scaling Benchmark (JIT-Warmed)
==============================================================
Evaluates the scan overhead of varying active epoch window sizes W in {1, 2, 4, 8, 16}
across 100,000 vectors to prove that multi-epoch lookups maintain sub-millisecond execution.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import AdaptiveOnlinePQ, _fast_unified_multi_epoch_adc


def run_window_scaling_benchmark(
    n_vectors: int = 100000,
    d: int = 128,
    m: int = 8,
    k: int = 256,
    n_queries: int = 1000
):
    print("=" * 75)
    print("      AO-PQ ACTIVE EPOCH WINDOW LATENCY SCALING ABLATION")
    print("=" * 75)
    np.random.seed(42)

    window_sizes = [1, 2, 4, 8, 16]
    results = []

    codes = np.random.randint(0, k, size=(n_vectors, m), dtype=np.uint8)
    queries = np.random.randn(n_queries, d).astype(np.float32)

    # 1. Warm up JIT kernel across all window sizes
    print("[*] Pre-warming JIT kernels across all window sizes...")
    for w in window_sizes:
        dummy_epochs = np.zeros(100, dtype=np.uint64)
        dummy_map = np.arange(w, dtype=np.uint64)
        dummy_lut = np.zeros((w, m, k), dtype=np.float32)
        _fast_unified_multi_epoch_adc(codes[:100], dummy_epochs, dummy_lut, dummy_map, m, k)

    # 2. Timed benchmarking
    for w in window_sizes:
        print(f"[*] Benchmarking Window Size W = {w:2d} ({n_vectors:,} vectors, {n_queries:,} queries)...")
        
        epochs = (np.arange(n_vectors, dtype=np.uint64) % w)
        epoch_id_map = np.arange(w, dtype=np.uint64)
        
        codebooks = [
            np.random.randn(m, k, d // m).astype(np.float32)
            for _ in range(w)
        ]

        latencies = []
        for q_idx, q in enumerate(queries):
            q_sub = q.reshape(m, d // m)
            
            epoch_lut_matrix = np.zeros((w, m, k), dtype=np.float32)
            for slot in range(w):
                cb = codebooks[slot]
                for sub_i in range(m):
                    epoch_lut_matrix[slot, sub_i] = np.sum((cb[sub_i] - q_sub[sub_i]) ** 2, axis=1)

            t0 = time.perf_counter()
            dists = _fast_unified_multi_epoch_adc(
                codes, epochs, epoch_lut_matrix, epoch_id_map, m, k
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            # Skip query 0 to discard residual cache misses
            if q_idx > 0:
                latencies.append(elapsed_ms)

        lat_arr = np.array(latencies)
        p50 = np.percentile(lat_arr, 50)
        p95 = np.percentile(lat_arr, 95)
        p99 = np.percentile(lat_arr, 99)
        mean_lat = np.mean(lat_arr)

        print(f"    -> Mean: {mean_lat:.3f} ms | P50: {p50:.3f} ms | P95: {p95:.3f} ms | P99: {p99:.3f} ms")

        for lat in lat_arr:
            results.append({
                "Window_Size": w,
                "Latency_ms": lat
            })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/epoch_window_scaling_metrics.csv", index=False)

    # 3. Clean publication figure
    sns.set_theme(style="ticks", font_scale=1.1)
    plt.figure(figsize=(7.5, 4.5))

    sns.boxplot(
        x="Window_Size",
        y="Latency_ms",
        hue="Window_Size",
        data=df,
        palette="Blues_d",
        legend=False,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )

    plt.title(f"Core ADC Scan Latency vs. Active Epoch Window Size ($N={n_vectors:,}$)", fontweight="bold", fontsize=11)
    plt.xlabel("Active Epoch Window Size ($W$)")
    plt.ylabel("Query Latency (ms)")
    plt.ylim(0.15, 0.75)
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig("results/fig6_epoch_window_scaling.png", dpi=300)
    plt.close()

    print("\n[SUCCESS] Generated clean results/fig6_epoch_window_scaling.png")


if __name__ == "__main__":
    run_window_scaling_benchmark()