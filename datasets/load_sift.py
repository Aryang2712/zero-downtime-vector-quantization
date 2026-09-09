"""
SIFT-1M 60K-Vector Streaming Benchmark under Controlled Distribution Drift (v3.7)
===============================================================================
Academic Rigor Standards:
- Matched Latency Comparisons:
    * Figure 3: Pure ADC Query Latency Distribution (Static PQ vs AO-PQ, N=1,500).
    * Figure 4: Pure ADC Search Latency Scaling (Segmented PQ with full buffer scan vs Unified AO-PQ).
- Fair 4-Way Recall Ablation:
    1. AO-PQ Two-Stage (ADC + Exact Rerank)
    2. Static PQ Two-Stage (ADC + Exact Rerank)
    3. AO-PQ Pure ADC (No Rerank)
    4. Static PQ Pure ADC (No Rerank)
- Ground truth excludes query vector with exact k+1 extraction.
"""

import os
import sys
import time
import urllib.request
import tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import AdaptiveOnlinePQ
from benchmark.simulate_stream import StaticPQBaseline, SegmentedPQBaseline


def read_fvecs(filename, max_vectors=None):
    with open(filename, 'rb') as f:
        dim_bytes = f.read(4)
        if not dim_bytes:
            return np.empty((0, 0), dtype=np.float32)
        d = np.frombuffer(dim_bytes, dtype=np.int32)[0]
        f.seek(0)
        
        record_size = 4 + d * 4
        f.seek(0, os.SEEK_END)
        total_records = f.tell() // record_size
        f.seek(0)
        
        n = total_records if max_vectors is None else min(max_vectors, total_records)
        raw_data = np.fromfile(f, dtype=np.float32, count=n * (d + 1))
        data = raw_data.reshape(n, d + 1)[:, 1:]
        return np.ascontiguousarray(data, dtype=np.float32)


def download_sift1m(data_dir="datasets/sift1m"):
    os.makedirs(data_dir, exist_ok=True)
    base_file = os.path.join(data_dir, "sift_base.fvecs")
    
    if not os.path.exists(base_file):
        tar_path = os.path.join(data_dir, "sift.tar.gz")
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
        print(f"[*] Downloading SIFT-1M dataset from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("[*] Extracting SIFT-1M...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        nested = os.path.join(data_dir, "sift")
        if os.path.exists(nested):
            for fname in os.listdir(nested):
                os.rename(os.path.join(nested, fname), os.path.join(data_dir, fname))
            os.rmdir(nested)
            
    print(f"[*] Loading SIFT-1M base vectors from {base_file}...")
    data = read_fvecs(base_file, max_vectors=100000)
    
    assert data.shape[1] == 128, f"Invalid SIFT vector dimension: {data.shape[1]}"
    assert data.shape[0] >= 60000, f"Insufficient SIFT vectors loaded: {data.shape[0]}"
    return data


def run_sift_benchmark():
    np.random.seed(42)
    d, m, k = 128, 8, 256
    
    raw_sift = download_sift1m()
    print(f"[*] Loaded dataset with verified shape: {raw_sift.shape} (dim={d})")

    print("[*] Clustering SIFT vectors to construct continuous streaming drift...")
    km = MiniBatchKMeans(n_clusters=30, batch_size=4096, random_state=42)
    cluster_labels = km.fit_predict(raw_sift)
    
    sort_order = np.argsort(cluster_labels)
    sift_stream = raw_sift[sort_order]

    num_initial = 10000
    batch_size = 1000
    num_batches = 50

    X_init = sift_stream[:num_initial]
    stream_batches = [
        sift_stream[num_initial + i * batch_size : num_initial + (i + 1) * batch_size]
        for i in range(num_batches)
    ]

    print("[1/3] Fitting bootstrap codebooks on initial 10,000 SIFT vectors...")
    adaptive_engine = AdaptiveOnlinePQ(
        d=d, m=m, k=k, lr=0.10, momentum=0.85, drift_threshold=0.015, max_active_window=4
    )
    static_engine = StaticPQBaseline(d=d, m=m, k=k)
    segmented_engine = SegmentedPQBaseline(d=d, m=m, k=k, segment_size=5000)

    adaptive_engine.fit_initial(X_init)
    adaptive_engine.ingest_stream_batch(X_init)

    static_engine.fit(X_init)
    static_engine.ingest(X_init)

    segmented_engine.ingest(X_init)

    # Pre-warm JIT search kernels
    dummy_q = X_init[0].copy()
    for _ in range(5):
        adaptive_engine.search(dummy_q, top_k=10, candidate_pool=80)
        adaptive_engine.search_adc_only(dummy_q, top_k=10)
        static_engine.search(dummy_q, top_k=10)
        segmented_engine.search(dummy_q, top_k=10)

    raw_accum = [X_init]
    batch_metrics = []
    individual_latencies = []

    print("[2/3] Streaming 50 drifting batches (60,000 indexed vectors)...")
    for b in tqdm(range(num_batches), desc="Streaming Ingestion"):
        batch = stream_batches[b]
        raw_accum.append(batch)
        full_raw = np.vstack(raw_accum)

        # Ingest
        t_ingest_start = time.perf_counter()
        adaptive_engine.ingest_stream_batch(batch)
        adp_ingest_ms = (time.perf_counter() - t_ingest_start) * 1000.0

        static_engine.ingest(batch)
        segmented_engine.ingest(batch)

        # Reconstruction MSE
        adp_snap = adaptive_engine.epoch_manager.registry[adaptive_engine.epoch_manager.active_epoch_id]
        adp_mse = adaptive_engine.compute_reconstruction_mse(batch, adp_snap.data)
        stc_mse = static_engine.compute_reconstruction_mse(batch)

        # 30 Queries with Rigorous Ground Truth (k+1 self-exclusion)
        sample_indices = np.random.choice(batch.shape[0], 30, replace=False)
        test_queries = batch[sample_indices]
        query_gids = (full_raw.shape[0] - batch.shape[0]) + sample_indices

        adp_rec10_rr, stc_rec10_rr = [], []
        adp_rec10_adc, stc_rec10_adc = [], []
        adp_adc_lats, stc_adc_lats, seg_adc_lats = [], [], []

        for q_i, q in enumerate(test_queries):
            self_gid = query_gids[q_i]

            # Ground Truth Top-10 (excluding self)
            exact_dists = np.sum((full_raw - q) ** 2, axis=1)
            exact_dists[self_gid] = np.inf
            true_top10 = np.argpartition(exact_dists, 10)[:10]
            true_top10 = true_top10[np.argsort(exact_dists[true_top10])]
            set10 = set(true_top10)

            # 1. AO-PQ (Two-Stage Rerank)
            pred_adp_rr, _ = adaptive_engine.search(q, top_k=11, candidate_pool=80)
            clean_adp_rr = [p for p in pred_adp_rr if p != self_gid][:10]
            adp_rec10_rr.append(len(set(clean_adp_rr).intersection(set10)) / 10.0)

            # 2. AO-PQ (Pure ADC Only - Matched Latency Measurement)
            t0 = time.perf_counter()
            pred_adp_adc, _ = adaptive_engine.search_adc_only(q, top_k=11)
            lat_adp_adc = (time.perf_counter() - t0) * 1000.0
            adp_adc_lats.append(lat_adp_adc)
            clean_adp_adc = [p for p in pred_adp_adc if p != self_gid][:10]
            adp_rec10_adc.append(len(set(clean_adp_adc).intersection(set10)) / 10.0)

            # 3. Static PQ (Pure ADC Only - Matched Latency Measurement)
            t1 = time.perf_counter()
            pred_stc_adc = static_engine.search(q, top_k=11)
            lat_stc_adc = (time.perf_counter() - t1) * 1000.0
            stc_adc_lats.append(lat_stc_adc)
            clean_stc_adc = [p for p in pred_stc_adc if p != self_gid][:10]
            stc_rec10_adc.append(len(set(clean_stc_adc).intersection(set10)) / 10.0)

            # 4. Static PQ (Two-Stage Rerank)
            pred_stc_rr, _ = static_engine.search_reranked(q, top_k=11, candidate_pool=80)
            clean_stc_rr = [p for p in pred_stc_rr if p != self_gid][:10]
            stc_rec10_rr.append(len(set(clean_stc_rr).intersection(set10)) / 10.0)

            # 5. Segmented PQ (Pure ADC with 100% Vector Coverage)
            t2 = time.perf_counter()
            pred_seg, _ = segmented_engine.search(q, top_k=10)
            lat_seg_adc = (time.perf_counter() - t2) * 1000.0
            seg_adc_lats.append(lat_seg_adc)

            individual_latencies.append({
                "System": "Static PQ (Pure ADC)",
                "Latency_ms": lat_stc_adc
            })
            individual_latencies.append({
                "System": "AO-PQ (Pure ADC Multi-Epoch)",
                "Latency_ms": lat_adp_adc
            })

        batch_metrics.append({
            "batch": b + 1,
            "total_vectors": full_raw.shape[0],
            "static_mse": stc_mse,
            "adaptive_mse": adp_mse,
            "adp_recall10_rerank": float(np.mean(adp_rec10_rr)),
            "stc_recall10_rerank": float(np.mean(stc_rec10_rr)),
            "adp_recall10_adc": float(np.mean(adp_rec10_adc)),
            "stc_recall10_adc": float(np.mean(stc_rec10_adc)),
            "adaptive_lat_ms": float(np.mean(adp_adc_lats)),
            "static_lat_ms": float(np.mean(stc_adc_lats)),
            "segmented_lat_ms": float(np.mean(seg_adc_lats)),
            "swaps": adaptive_engine.total_swaps,
            "compactions": adaptive_engine.total_compactions,
            "ingest_time_ms": adp_ingest_ms
        })

    adaptive_engine.migration_queue.join()
    adaptive_engine.close()

    df_batch = pd.DataFrame(batch_metrics)
    df_lat = pd.DataFrame(individual_latencies)

    df_batch.to_csv("results/sift1m_benchmark_metrics.csv", index=False)
    df_lat.to_csv("results/sift1m_individual_latencies.csv", index=False)

    print("[3/3] Generating publication-grade empirical figures...")
    sns.set_theme(style="ticks", font_scale=1.1)

    # --- Figure 1: Reconstruction MSE ---
    plt.figure(figsize=(8, 4.2))
    plt.plot(df_batch["batch"], df_batch["static_mse"], label="Static PQ Baseline", color="#d9534f", linestyle="--", linewidth=2.2)
    plt.plot(df_batch["batch"], df_batch["adaptive_mse"], label="AO-PQ Multi-Epoch (Ours)", color="#0275d8", linewidth=2.5)
    plt.title("SIFT-1M Subset Reconstruction MSE under Controlled Distribution Drift", fontweight="bold", fontsize=11)
    plt.xlabel("Streaming Batch Number (1,000 vectors/batch)")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig1_reconstruction_mse.png", dpi=300)
    plt.close()

    # --- Figure 2: Fair 4-Way Recall Comparison ---
    plt.figure(figsize=(8, 4.2))
    plt.plot(df_batch["batch"], df_batch["adp_recall10_rerank"] * 100.0, label="AO-PQ (ADC + Exact Rerank)", color="#0275d8", linewidth=2.4)
    plt.plot(df_batch["batch"], df_batch["stc_recall10_rerank"] * 100.0, label="Static PQ (ADC + Exact Rerank)", color="#f0ad4e", linestyle="-.", linewidth=2.0)
    plt.plot(df_batch["batch"], df_batch["adp_recall10_adc"] * 100.0, label="AO-PQ (Pure ADC Only)", color="#5cb85c", linestyle="--", linewidth=2.0)
    plt.plot(df_batch["batch"], df_batch["stc_recall10_adc"] * 100.0, label="Static PQ (Pure ADC Only)", color="#d9534f", linestyle=":", linewidth=2.0)
    plt.axhline(85.0, color="gray", linestyle="--", alpha=0.7, label="SLA Target (85%)")
    plt.title("SIFT-1M Subset Recall@10: Fair Algorithmic & Pipeline Ablation", fontweight="bold", fontsize=11)
    plt.xlabel("Streaming Batch Number (1,000 vectors/batch)")
    plt.ylabel("Empirical Recall@10 (%)")
    plt.ylim(30, 105)
    plt.xlim(1, 50)
    plt.legend(loc="lower left", fontsize=8.5, ncol=2)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig2_recall_retention.png", dpi=300)
    plt.close()

    # --- Figure 3: Matched Pure ADC Latency Distribution (N=1,500 Samples) ---
    plt.figure(figsize=(7, 4.2))
    sns.boxplot(
        x="System",
        y="Latency_ms",
        data=df_lat,
        palette=["#0275d8", "#5cb85c"],
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    plt.title("SIFT-1M Subset Pure ADC Latency Distribution (N=1,500 Queries)", fontweight="bold", fontsize=11)
    plt.ylabel("Query Latency (ms)")
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig("results/sift_fig3_latency_distribution.png", dpi=300)
    plt.close()

    # --- Figure 4: Matched Pure ADC Latency Scaling (Segmented vs Unified) ---
    plt.figure(figsize=(8, 4.2))
    plt.plot(df_batch["total_vectors"], df_batch["segmented_lat_ms"], label="Segmented PQ Store (Scatter-Gather)", color="#d9534f", linestyle="--", linewidth=2.2)
    plt.plot(df_batch["total_vectors"], df_batch["adaptive_lat_ms"], label="Unified AO-PQ (JIT Multi-Epoch)", color="#0275d8", linewidth=2.5)
    plt.title("SIFT-1M Subset Pure ADC Search Latency Scaling (Segmented vs Unified)", fontweight="bold", fontsize=11)
    plt.xlabel("Total Indexed Vectors in Memory (100% Coverage)")
    plt.ylabel("Query Latency (ms)")
    plt.legend(loc="upper left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig4_scalability.png", dpi=300)
    plt.close()

    print(f"\n[SUCCESS] SIFT-1M Subset Benchmark Completed.")
    print(f" -> Total Codebook Swaps: {adaptive_engine.total_swaps}")
    print(f" -> Total Background Compactions: {adaptive_engine.total_compactions}")


if __name__ == "__main__":
    run_sift_benchmark()