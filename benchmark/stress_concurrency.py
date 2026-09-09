"""
AO-PQ v3.8 — Zero-Downtime Multi-Threaded Concurrency Benchmark
===============================================================
Validates:
1. Lock-free reader throughput under simultaneous streaming drift ingestion.
2. Reader P50, P95, and P99 latency stability during active codebook promotions.
3. Zero data tearing, NaN returns, or unhandled exceptions across concurrent workers.
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import AdaptiveOnlinePQ


def run_concurrency_stress_test(
    num_reader_threads: int = 8,
    initial_vectors: int = 10000,
    streaming_batches: int = 25,
    batch_size: int = 1000,
    d: int = 128,
    m: int = 8,
    k: int = 256
):
    print("=" * 75)
    print("      AO-PQ ZERO-DOWNTIME MULTI-THREADED CONCURRENCY BENCHMARK")
    print("=" * 75)
    np.random.seed(42)

    # 1. Initialize and bootstrap AO-PQ Engine
    print(f"[*] Bootstrapping AO-PQ on {initial_vectors} initial vectors (d={d}, m={m}, k={k})...")
    engine = AdaptiveOnlinePQ(
        d=d, m=m, k=k, lr=0.12, momentum=0.85, drift_threshold=0.015, max_active_window=4
    )
    X_init = np.random.randn(initial_vectors, d).astype(np.float32)
    engine.fit_initial(X_init)
    engine.ingest_stream_batch(X_init)

    # Pre-warm JIT kernels
    dummy_q = X_init[0].copy()
    for _ in range(5):
        engine.search_adc_only(dummy_q, top_k=10)
        engine.search(dummy_q, top_k=10)

    stop_event = threading.Event()
    query_records: List[Dict[str, float]] = []
    reader_errors: List[str] = []
    records_lock = threading.Lock()

    # 2. Worker definition for continuous reader threads
    def reader_worker(thread_id: int):
        local_records = []
        while not stop_event.is_set():
            q = np.random.randn(d).astype(np.float32)
            t0 = time.perf_counter()
            try:
                ids, dists = engine.search_adc_only(q, top_k=10)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                
                # Check for invariant failure (NaN or incomplete top-k)
                if len(ids) != 10 or np.isnan(dists).any():
                    reader_errors.append(f"Thread {thread_id}: Invalid distance / NaN observed.")
                else:
                    local_records.append({
                        "Thread_ID": thread_id,
                        "Timestamp": time.time(),
                        "Latency_ms": elapsed_ms
                    })
            except Exception as e:
                reader_errors.append(f"Thread {thread_id} Exception: {str(e)}")
            
            # Small sleep to simulate high-frequency query interval
            time.sleep(0.0005)

        with records_lock:
            query_records.extend(local_records)

    # 3. Spawn Reader Threads
    print(f"[*] Launching {num_reader_threads} concurrent search reader threads...")
    reader_threads = [
        threading.Thread(target=reader_worker, args=(i,), daemon=True)
        for i in range(num_reader_threads)
    ]
    for t in reader_threads:
        t.start()

    # 4. Ingest high-drift streaming batches on main thread
    print(f"[*] Streaming {streaming_batches} drifting batches ({batch_size} vectors/batch)...")
    ingest_timings = []
    t_start_total = time.time()

    for b in range(streaming_batches):
        # Induce shifting distribution
        drift_batch = np.random.randn(batch_size, d).astype(np.float32) + (b * 4.0)
        
        t_ing_0 = time.perf_counter()
        # Periodically force codebook swaps to vigorously stress generation swapping
        engine.ingest_stream_batch(drift_batch, force_swap=(b % 3 == 0))
        ingest_elapsed = (time.perf_counter() - t_ing_0) * 1000.0
        ingest_timings.append(ingest_elapsed)

        time.sleep(0.04)  # Small pacing to allow readers to hit active swap windows

    total_stream_time = time.time() - t_start_total

    # 5. Stop readers and drain worker queues
    print("[*] Ingestion complete. Stopping reader threads and draining migration queue...")
    stop_event.set()
    for t in reader_threads:
        t.join()

    engine.migration_queue.join()
    engine.close()

    # 6. Quantitative Analysis
    df_queries = pd.DataFrame(query_records)
    os.makedirs("results", exist_ok=True)
    df_queries.to_csv("results/concurrency_stress_metrics.csv", index=False)

    total_queries = len(df_queries)
    qps = total_queries / total_stream_time
    p50 = np.percentile(df_queries["Latency_ms"], 50)
    p95 = np.percentile(df_queries["Latency_ms"], 95)
    p99 = np.percentile(df_queries["Latency_ms"], 99)

    print("\n" + "=" * 75)
    print("                    STRESS BENCHMARK RESULTS")
    print("=" * 75)
    print(f" Total Codebook Promotions (Swaps)     : {engine.total_swaps}")
    print(f" Total Background Compactions Handled : {engine.total_compactions}")
    print(f" Total Concurrent Queries Served       : {total_queries:,}")
    print(f" Concurrent Query Throughput (QPS)     : {qps:.1f} queries/sec")
    print(f" Reader Execution Exceptions           : {len(reader_errors)}")
    print("-" * 75)
    print(f" P50 Latency (Median)                  : {p50:.3f} ms")
    print(f" P95 Latency                           : {p95:.3f} ms")
    print(f" P99 Latency (Tail Stability)          : {p99:.3f} ms")
    print("=" * 75)

    assert len(reader_errors) == 0, f"Invariant Violation: Observed reader errors: {reader_errors[:5]}"

    # 7. Generate Concurrency Timeline Plot
    sns.set_theme(style="ticks", font_scale=1.1)
    plt.figure(figsize=(9, 4.5))

    # Scatter of individual query latencies across time
    t_min = df_queries["Timestamp"].min()
    rel_time = df_queries["Timestamp"] - t_min

    plt.scatter(
        rel_time,
        df_queries["Latency_ms"],
        alpha=0.25,
        s=8,
        color="#0275d8",
        label=f"Concurrent Queries (N={total_queries:,})"
    )
    plt.axhline(p50, color="#5cb85c", linestyle="-", linewidth=1.8, label=f"P50 Median ({p50:.2f} ms)")
    plt.axhline(p99, color="#d9534f", linestyle="--", linewidth=1.8, label=f"P99 Tail ({p99:.2f} ms)")

    plt.title(f"AO-PQ Zero-Downtime Concurrency: {num_reader_threads} Readers During Continuous Ingestion", fontweight="bold", fontsize=11)
    plt.xlabel("Elapsed Time (Seconds)")
    plt.ylabel("Query Latency (ms)")
    plt.ylim(0, max(2.5, p99 * 1.6))
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/concurrency_stress_latency.png", dpi=300)
    plt.close()

    print("\n[SUCCESS] Generated results/concurrency_stress_latency.png")


if __name__ == "__main__":
    run_concurrency_stress_test()