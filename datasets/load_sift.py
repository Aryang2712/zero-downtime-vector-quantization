"""
SIFT-1M Full 1,000,000 Vector Streaming Benchmark (AO-PQ v4.2 MVCC Edition)
===========================================================================
Evaluates continuous streaming drift across 1,000,000 SIFT vectors against:
- Static PQ Baseline
- Blue-Green Dual Index Rebuild Baseline (Industry Standard)
- Generational Staleness Tracking (Property 3 Verification)
"""

import os
import sys
import time
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Set
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.engine import AdaptiveOnlinePQ, _fast_unified_multi_epoch_adc, _fast_exact_rerank


def read_fvecs(filename: str, max_vectors: int = 1000000) -> np.ndarray:
    """Reads .fvecs format into float32 numpy array."""
    with open(filename, "rb") as f:
        dim_raw = f.read(4)
        if not dim_raw:
            raise ValueError("Empty fvecs file.")
        dim = np.frombuffer(dim_raw, dtype=np.int32)[0]
        f.seek(0)
        
        record_size = 4 + dim * 4
        f.seek(0, 2)
        total_records = min(f.tell() // record_size, max_vectors)
        f.seek(0)
        
        data = np.empty((total_records, dim), dtype=np.float32)
        for i in range(total_records):
            f.seek(4, 1)  # Skip 4-byte header
            data[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return data


def download_sift1m(data_dir: str = "datasets/sift1m") -> np.ndarray:
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
            
    print(f"[*] Loading complete 1,000,000 SIFT-1M base vectors from {base_file}...")
    data = read_fvecs(base_file, max_vectors=1000000)
    assert data.shape[0] == 1000000 and data.shape[1] == 128
    print(f"[*] Loaded dataset with verified shape: {data.shape} (dim={data.shape[1]})")
    return data


class StaticPQEngine:
    """Fixed-codebook static baseline."""
    def __init__(self, d: int = 128, m: int = 8, k: int = 256):
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.centroids = np.zeros((m, k, self.d_sub), dtype=np.float32)
        self.codes = []
        self.raw_vectors = []

    def fit(self, X_train: np.ndarray):
        X_sub = X_train.reshape(X_train.shape[0], self.m, self.d_sub)
        for i in range(self.m):
            km = MiniBatchKMeans(n_clusters=self.k, batch_size=min(2048, X_train.shape[0]), n_init=1, max_iter=40, random_state=42)
            km.fit(X_sub[:, i, :])
            self.centroids[i] = km.cluster_centers_.astype(np.float32)

    def quantize(self, X: np.ndarray) -> np.ndarray:
        X_sub = X.reshape(X.shape[0], self.m, self.d_sub)
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)
        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            c = self.centroids[i]
            dists = np.sum(sub_vecs ** 2, axis=1, keepdims=True) + np.sum(c ** 2, axis=1, keepdims=True).T - 2.0 * np.dot(sub_vecs, c.T)
            codes[:, i] = np.argmin(dists, axis=1)
        return codes

    def compute_reconstruction_mse(self, X: np.ndarray) -> float:
        codes = self.quantize(X)
        X_sub = X.reshape(X.shape[0], self.m, self.d_sub)
        rec = np.zeros_like(X_sub)
        for i in range(self.m):
            rec[:, i, :] = self.centroids[i][codes[:, i]]
        return float(np.mean((X - rec.reshape(X.shape[0], self.d)) ** 2))

    def ingest(self, X: np.ndarray):
        self.codes.append(self.quantize(X))
        self.raw_vectors.append(np.ascontiguousarray(X, dtype=np.float32))

    def search_adc_only(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        all_codes = np.vstack(self.codes)
        n = all_codes.shape[0]
        q_sub = query.reshape(self.m, self.d_sub)
        lut = np.zeros((1, self.m, self.k), dtype=np.float32)
        for sub_i in range(self.m):
            lut[0, sub_i] = np.sum((self.centroids[sub_i] - q_sub[sub_i]) ** 2, axis=1)
        epochs = np.zeros(n, dtype=np.uint64)
        map_arr = np.zeros(1, dtype=np.uint64)
        dists = _fast_unified_multi_epoch_adc(all_codes, epochs, lut, map_arr, self.m, self.k)
        final_k = min(top_k, n)
        top_idx = np.argpartition(dists, final_k - 1)[:final_k]
        sorted_order = top_idx[np.argsort(dists[top_idx])]
        return sorted_order.astype(np.uint64), dists[sorted_order]

    def search(self, query: np.ndarray, top_k: int = 10, candidate_pool: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        all_codes = np.vstack(self.codes)
        n = all_codes.shape[0]
        q_sub = query.reshape(self.m, self.d_sub)
        lut = np.zeros((1, self.m, self.k), dtype=np.float32)
        for sub_i in range(self.m):
            lut[0, sub_i] = np.sum((self.centroids[sub_i] - q_sub[sub_i]) ** 2, axis=1)
        epochs = np.zeros(n, dtype=np.uint64)
        map_arr = np.zeros(1, dtype=np.uint64)
        dists = _fast_unified_multi_epoch_adc(all_codes, epochs, lut, map_arr, self.m, self.k)
        cand_k = min(max(top_k * 4, candidate_pool), n)
        cand_idx = np.argpartition(dists, cand_k - 1)[:cand_k]
        all_raw = np.vstack(self.raw_vectors)
        cand_vecs = all_raw[cand_idx]
        exact_dists = _fast_exact_rerank(cand_vecs, query)
        final_k = min(top_k, cand_k)
        best_loc = np.argpartition(exact_dists, final_k - 1)[:final_k]
        sorted_order = best_loc[np.argsort(exact_dists[best_loc])]
        return cand_idx[sorted_order].astype(np.uint64), exact_dists[sorted_order]


class BlueGreenRebuildBaseline:
    """
    Industry-standard baseline: Serves queries from Index A while 
    training and rebuilding Index B in the background, doubling peak RAM.
    """
    def __init__(self, d: int = 128, m: int = 8, k: int = 256):
        self.d = d
        self.m = m
        self.k = k
        self.active_index = StaticPQEngine(d, m, k)
        self.accumulated_vectors = []

    def fit_initial(self, X_train: np.ndarray):
        self.active_index.fit(X_train)
        self.active_index.ingest(X_train)
        self.accumulated_vectors.append(X_train)

    def ingest_and_maybe_rebuild(self, X_batch: np.ndarray, rebuild: bool = False) -> Tuple[float, float]:
        self.active_index.ingest(X_batch)
        self.accumulated_vectors.append(X_batch)
        
        rebuild_time_ms = 0.0
        peak_memory_mb = (len(np.vstack(self.accumulated_vectors)) * self.m) / (1024 * 1024)

        if rebuild:
            t0 = time.perf_counter()
            all_vecs = np.vstack(self.accumulated_vectors)
            # Memory doubles during background training
            peak_memory_mb *= 2.0
            
            shadow_index = StaticPQEngine(self.d, self.m, self.k)
            shadow_index.fit(all_vecs[:min(100000, len(all_vecs))])
            for chunk in self.accumulated_vectors:
                shadow_index.ingest(chunk)
            
            self.active_index = shadow_index
            rebuild_time_ms = (time.perf_counter() - t0) * 1000.0

        return rebuild_time_ms, peak_memory_mb

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        return self.active_index.search(query, top_k=top_k)


def run_sift_benchmark():
    print("=" * 75)
    print("      SIFT-1M STREAMING BENCHMARK (AO-PQ v4.2 MVCC PROTOCOL)")
    print("=" * 75)
    raw_sift = download_sift1m()
    
    print("[*] Clustering SIFT vectors to construct continuous streaming drift...")
    cluster_sorter = MiniBatchKMeans(n_clusters=25, batch_size=4096, random_state=42)
    cluster_labels = cluster_sorter.fit_predict(raw_sift)
    sort_idx = np.argsort(cluster_labels)
    sift_stream = raw_sift[sort_idx]

    d, m, k = 128, 8, 256
    num_initial = 100000
    batch_size = 10000
    num_batches = 90

    X_init = sift_stream[:num_initial]
    stream_batches = [
        sift_stream[num_initial + i * batch_size : num_initial + (i + 1) * batch_size]
        for i in range(num_batches)
    ]

    print(f"[1/3] Initializing engines on {num_initial:,} bootstrap SIFT vectors...")
    adaptive_engine = AdaptiveOnlinePQ(d=d, m=m, k=k, lr=0.12, momentum=0.85, drift_threshold=0.015, max_active_window=4)
    static_engine = StaticPQEngine(d=d, m=m, k=k)
    bluegreen_engine = BlueGreenRebuildBaseline(d=d, m=m, k=k)

    adaptive_engine.fit_initial(X_init)
    adaptive_engine.ingest_stream_batch(X_init)

    static_engine.fit(X_init)
    static_engine.ingest(X_init)

    bluegreen_engine.fit_initial(X_init)

    print(f"[2/3] Streaming {num_batches} drifting batches (1,000,000 total vectors)...")
    metrics_records = []
    adp_individual_latencies = []
    static_individual_latencies = []

    for b_idx, batch in enumerate(tqdm(stream_batches, desc="Streaming Ingestion")):
        adp_mse = adaptive_engine.compute_reconstruction_mse(batch, adaptive_engine.shadow_codebook)
        static_mse = static_engine.compute_reconstruction_mse(batch)

        adaptive_engine.ingest_stream_batch(batch)
        static_engine.ingest(batch)
        
        # Periodic Blue-Green Rebuild every 20 batches (~200k vectors)
        trigger_bg = ((b_idx + 1) % 20 == 0)
        bg_rebuild_ms, bg_peak_mb = bluegreen_engine.ingest_and_maybe_rebuild(batch, rebuild=trigger_bg)

        # Generational Staleness Tracking
        staleness = adaptive_engine.get_generational_staleness()

        # Query evaluation using on-manifold queries
        adp_recalls_twostage, static_recalls_twostage, bg_recalls_twostage = [], [], []
        adp_recalls_pure, static_recalls_pure = [], []
        adp_latencies_adc, static_latencies_adc = [], []

        curr_n = num_initial + (b_idx + 1) * batch_size
        curr_corpus = sift_stream[:curr_n]

        sample_indices = np.random.choice(len(batch), size=min(20, len(batch)), replace=False)
        query_samples = batch[sample_indices] + np.random.normal(0.0, 5.0, size=(len(sample_indices), d)).astype(np.float32)

        for q in query_samples:
            gt_dists = np.sum((curr_corpus - q) ** 2, axis=1)
            gt_top10 = set(np.argpartition(gt_dists, 10)[:10])

            # Two-Stage Search
            pred_adp, _ = adaptive_engine.search(q, top_k=10)
            pred_sta, _ = static_engine.search(q, top_k=10, candidate_pool=80)
            pred_bg, _ = bluegreen_engine.search(q, top_k=10)

            adp_recalls_twostage.append(len(gt_top10.intersection(set(pred_adp))) / 10.0)
            static_recalls_twostage.append(len(gt_top10.intersection(set(pred_sta))) / 10.0)
            bg_recalls_twostage.append(len(gt_top10.intersection(set(pred_bg))) / 10.0)

            # Pure ADC Timed Search
            t0 = time.perf_counter()
            p_adp_ids, _ = adaptive_engine.search_adc_only(q, top_k=10)
            t_adp = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            p_sta_ids, _ = static_engine.search_adc_only(q, top_k=10)
            t_sta = (time.perf_counter() - t0) * 1000.0

            adp_latencies_adc.append(t_adp)
            static_latencies_adc.append(t_sta)
            adp_individual_latencies.append(t_adp)
            static_individual_latencies.append(t_sta)

            adp_recalls_pure.append(len(gt_top10.intersection(set(p_adp_ids))) / 10.0)
            static_recalls_pure.append(len(gt_top10.intersection(set(p_sta_ids))) / 10.0)

        metrics_records.append({
            "Batch": b_idx + 1,
            "Total_Vectors": curr_n,
            "Adaptive_MSE": adp_mse,
            "Static_MSE": static_mse,
            "Adaptive_Recall10": np.mean(adp_recalls_twostage) * 100.0,
            "Static_Recall10": np.mean(static_recalls_twostage) * 100.0,
            "BlueGreen_Recall10": np.mean(bg_recalls_twostage) * 100.0,
            "Adaptive_Recall10_Pure": np.mean(adp_recalls_pure) * 100.0,
            "Static_Recall10_Pure": np.mean(static_recalls_pure) * 100.0,
            "Adaptive_Latency_ms": np.mean(adp_latencies_adc),
            "Static_Latency_ms": np.mean(static_latencies_adc),
            "Max_Epoch_Lag": staleness["max_epoch_lag"],
            "Pct_Stale_Records": staleness["pct_records_stale_gt_1"],
            "BlueGreen_Peak_RAM_MB": bg_peak_mb
        })

    adaptive_engine.migration_queue.join()
    mem = adaptive_engine.get_memory_footprint()

    print("\n" + "=" * 75)
    print("                    PHYSICAL MEMORY BREAKDOWN")
    print("=" * 75)
    print(f" Total Quantized Vector Records : {mem['total_records']:,}")
    print(f" Standalone PQ Index Memory     : {mem['pq_index_bytes'] / (1024 * 1024):.2f} MB")
    print(f" Raw Uncompressed Cache Memory  : {mem['raw_cache_bytes'] / (1024 * 1024):.2f} MB")
    print(f" Total Engine Allocated Memory  : {mem['total_allocated_bytes'] / (1024 * 1024):.2f} MB")
    print(f" Equivalent Flat Index Memory   : {mem['equivalent_flat_bytes'] / (1024 * 1024):.2f} MB")
    print(f" Standalone PQ Compression Ratio: {mem['pq_standalone_compression_ratio']:.2f}%")
    print("=" * 75)

    adaptive_engine.close()

    df_metrics = pd.DataFrame(metrics_records)
    os.makedirs("results", exist_ok=True)
    df_metrics.to_csv("results/sift1m_benchmark_metrics.csv", index=False)

    print("[3/3] Generating publication-grade empirical figures...")
    sns.set_theme(style="ticks", font_scale=1.1)

    # Figure 1: Reconstruction MSE Tracking
    plt.figure(figsize=(8, 4))
    plt.plot(df_metrics["Batch"], df_metrics["Static_MSE"], label="Static PQ Baseline", color="#d9534f", linestyle="--", linewidth=1.8)
    plt.plot(df_metrics["Batch"], df_metrics["Adaptive_MSE"], label="AO-PQ Multi-Epoch (Ours)", color="#0275d8", linewidth=2.2)
    plt.title("SIFT-1M (1,000,000 Vectors) Reconstruction MSE under Streaming Drift", fontweight="bold", fontsize=11)
    plt.xlabel("Streaming Batch Number (10,000 vectors/batch)")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig1_reconstruction_mse.png", dpi=300)
    plt.close()

    # Figure 2: Matched 4-Way Recall Retention + BlueGreen Baseline
    plt.figure(figsize=(8, 4))
    plt.plot(df_metrics["Batch"], df_metrics["Adaptive_Recall10"], label="AO-PQ Two-Stage (Ours)", color="#0275d8", linewidth=2.0)
    plt.plot(df_metrics["Batch"], df_metrics["BlueGreen_Recall10"], label="Blue-Green Rebuild Baseline", color="#5cb85c", linestyle="-.", linewidth=1.8)
    plt.plot(df_metrics["Batch"], df_metrics["Static_Recall10"], label="Static PQ Baseline", color="#f0ad4e", linestyle="--", linewidth=1.8)
    plt.axhline(85.0, color="gray", linestyle="--", alpha=0.7, label="SLA Target (85%)")
    plt.title("SIFT-1M (1,000,000 Vectors) Recall@10 vs. Blue-Green Rebuild", fontweight="bold", fontsize=11)
    plt.xlabel("Streaming Batch Number (10,000 vectors/batch)")
    plt.ylabel("Empirical Recall@10 (%)")
    plt.ylim(0, 105)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig2_recall_retention.png", dpi=300)
    plt.close()

    # Figure 3: Generational Staleness Tracking (Property 3 Verification)
    plt.figure(figsize=(8, 4))
    plt.plot(df_metrics["Batch"], df_metrics["Pct_Stale_Records"], label="Records Stale > 1 Epoch (%)", color="#d9534f", linewidth=2.0)
    plt.axhline(5.0, color="gray", linestyle=":", label="Staleness Ceiling (5%)")
    plt.title("Generational Staleness Bounded by Out-of-Line COW Compaction", fontweight="bold", fontsize=11)
    plt.xlabel("Streaming Batch Number (10,000 vectors/batch)")
    plt.ylabel("Corpus Percentage Stale (%)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig5_generational_staleness.png", dpi=300)
    plt.close()

    print("\n[SUCCESS] SIFT-1M Benchmark Completed (1,000,000 Vectors).")
    print(f" -> Total Codebook Promotions (Swaps): {adaptive_engine.total_swaps}")
    print(f" -> Total Background Compactions Handled: {adaptive_engine.total_compactions}")


if __name__ == "__main__":
    run_sift_benchmark()
