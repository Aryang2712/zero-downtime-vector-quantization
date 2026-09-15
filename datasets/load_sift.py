"""
SIFT-1M Full 1,000,000 Vector Streaming Benchmark (AO-PQ v4.1)
==============================================================
Evaluates continuous streaming drift across the complete 1,000,000 SIFT vectors:
- Figure 1: Reconstruction MSE tracking across 90 streaming drift batches.
- Figure 2: Matched 4-Way Recall@10 retention curve (on-manifold evaluation).
- Figure 3: Pure ADC latency distribution boxplot (N=1,800 queries).
- Figure 4: Search latency scalability vs. Segmented PQ Store (100% vector coverage).
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


class SegmentedPQStore:
    """Segmented Scatter-Gather baseline with fast mini-batch fitting."""
    def __init__(self, d: int = 128, m: int = 8, k: int = 256):
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.segments = []

    def ingest(self, X: np.ndarray):
        X_sub = X.reshape(X.shape[0], self.m, self.d_sub)
        sample_n = min(1024, X.shape[0])
        sample_sub = X_sub[:sample_n]
        
        c = np.zeros((self.m, self.k, self.d_sub), dtype=np.float32)
        for i in range(self.m):
            km = MiniBatchKMeans(
                n_clusters=self.k,
                batch_size=min(512, sample_n),
                n_init=1,
                max_iter=15,
                random_state=42
            )
            km.fit(sample_sub[:, i, :])
            c[i] = km.cluster_centers_.astype(np.float32)

        codes = np.zeros((X.shape[0], self.m), dtype=np.uint8)
        for i in range(self.m):
            sub_v = X_sub[:, i, :]
            cen = c[i]
            dists = (
                np.sum(sub_v ** 2, axis=1, keepdims=True)
                + np.sum(cen ** 2, axis=1, keepdims=True).T
                - 2.0 * np.dot(sub_v, cen.T)
            )
            codes[:, i] = np.argmin(dists, axis=1)

        self.segments.append({"centroids": c, "codes": codes, "size": X.shape[0]})

    def search_adc_only(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        seg_results = []
        q_sub = query.reshape(self.m, self.d_sub)
        for s in self.segments:
            lut = np.zeros((1, self.m, self.k), dtype=np.float32)
            for sub_i in range(self.m):
                lut[0, sub_i] = np.sum((s["centroids"][sub_i] - q_sub[sub_i]) ** 2, axis=1)
            epochs = np.zeros(s["size"], dtype=np.uint64)
            map_arr = np.zeros(1, dtype=np.uint64)
            dists = _fast_unified_multi_epoch_adc(s["codes"], epochs, lut, map_arr, self.m, self.k)
            k_s = min(top_k, s["size"])
            idx = np.argpartition(dists, k_s - 1)[:k_s]
            seg_results.append(dists[idx])
        all_dists = np.concatenate(seg_results)
        final_k = min(top_k, len(all_dists))
        best_idx = np.argpartition(all_dists, final_k - 1)[:final_k]
        sorted_order = best_idx[np.argsort(all_dists[best_idx])]
        return np.arange(final_k, dtype=np.uint64), all_dists[sorted_order]


def run_sift_benchmark():
    print("=" * 75)
    print("      SIFT-1M FULL 1,000,000 VECTOR STREAMING BENCHMARK (v4.1)")
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

    print(f"[1/3] Fitting bootstrap codebooks on initial {num_initial:,} SIFT vectors...")
    adaptive_engine = AdaptiveOnlinePQ(d=d, m=m, k=k, lr=0.12, momentum=0.85, drift_threshold=0.015, max_active_window=4)
    static_engine = StaticPQEngine(d=d, m=m, k=k)
    segmented_engine = SegmentedPQStore(d=d, m=m, k=k)

    adaptive_engine.fit_initial(X_init)
    adaptive_engine.ingest_stream_batch(X_init)

    static_engine.fit(X_init)
    static_engine.ingest(X_init)
    segmented_engine.ingest(X_init)

    print(f"[2/3] Streaming {num_batches} drifting batches (1,000,000 total vectors)...")
    metrics_records = []
    adp_individual_latencies = []
    static_individual_latencies = []

    for b_idx, batch in enumerate(tqdm(stream_batches, desc="Streaming Ingestion")):
        adp_mse = adaptive_engine.compute_reconstruction_mse(batch, adaptive_engine.shadow_codebook)
        static_mse = static_engine.compute_reconstruction_mse(batch)

        adaptive_engine.ingest_stream_batch(batch)
        static_engine.ingest(batch)
        segmented_engine.ingest(batch)

        # Query evaluation using on-manifold queries with perturbation
        adp_recalls_twostage, static_recalls_twostage = [], []
        adp_recalls_pure, static_recalls_pure = [], []
        adp_latencies_adc, static_latencies_adc, seg_latencies_adc = [], [], []

        curr_n = num_initial + (b_idx + 1) * batch_size
        curr_corpus = sift_stream[:curr_n]

        # Sample 20 on-manifold queries from the current batch with Gaussian noise
        sample_indices = np.random.choice(len(batch), size=min(20, len(batch)), replace=False)
        query_samples = batch[sample_indices] + np.random.normal(0.0, 5.0, size=(len(sample_indices), d)).astype(np.float32)

        for q in query_samples:
            # Ground truth calculation on full indexed corpus
            gt_dists = np.sum((curr_corpus - q) ** 2, axis=1)
            gt_top10 = set(np.argpartition(gt_dists, 10)[:10])

            # Two-Stage Search with Dynamic Candidate Scaling
            pred_adp, _ = adaptive_engine.search(q, top_k=10)
            pred_sta, _ = static_engine.search(q, top_k=10, candidate_pool=80)
            adp_recalls_twostage.append(len(gt_top10.intersection(set(pred_adp))) / 10.0)
            static_recalls_twostage.append(len(gt_top10.intersection(set(pred_sta))) / 10.0)

            # Pure ADC Timed Search
            t0 = time.perf_counter()
            p_adp_ids, _ = adaptive_engine.search_adc_only(q, top_k=10)
            t_adp = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            p_sta_ids, _ = static_engine.search_adc_only(q, top_k=10)
            t_sta = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            segmented_engine.search_adc_only(q, top_k=10)
            t_seg = (time.perf_counter() - t0) * 1000.0

            adp_latencies_adc.append(t_adp)
            static_latencies_adc.append(t_sta)
            seg_latencies_adc.append(t_seg)
            adp_individual_latencies.append(t_adp)
            static_individual_latencies.append(t_sta)

            adp_recalls_pure.append(len(gt_top10.intersection(set(p_adp_ids))) / 10.0)
            static_recalls_pure.append(len(gt_top10.intersection(set(p_sta_ids))) / 10.0)

        metrics_records.append({
            "Batch": b_idx + 1,
            "Total_Vectors": num_initial + (b_idx + 1) * batch_size,
            "Adaptive_MSE": adp_mse,
            "Static_MSE": static_mse,
            "Adaptive_Recall10_TwoStage": np.mean(adp_recalls_twostage) * 100.0,
            "Static_Recall10_TwoStage": np.mean(static_recalls_twostage) * 100.0,
            "Adaptive_Recall10_PureADC": np.mean(adp_recalls_pure) * 100.0,
            "Static_Recall10_PureADC": np.mean(static_recalls_pure) * 100.0,
            "Adaptive_Latency_ms": np.mean(adp_latencies_adc),
            "Static_Latency_ms": np.mean(static_latencies_adc),
            "Segmented_Latency_ms": np.mean(seg_latencies_adc)
        })

    adaptive_engine.migration_queue.join()
    
    # Print Exact Memory Footprint
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

    # Figure 1: Reconstruction MSE Tracking
    print("[3/3] Generating publication-grade empirical figures...")
    sns.set_theme(style="ticks", font_scale=1.1)

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

    # Figure 2: Matched 4-Way Recall Retention
    plt.figure(figsize=(8, 4))
    plt.plot(df_metrics["Batch"], df_metrics["Adaptive_Recall10_TwoStage"], label="AO-PQ (ADC + Exact Rerank)", color="#0275d8", linewidth=2.0)
    plt.plot(df_metrics["Batch"], df_metrics["Static_Recall10_TwoStage"], label="Static PQ (ADC + Exact Rerank)", color="#f0ad4e", linestyle="-.", linewidth=1.8)
    plt.plot(df_metrics["Batch"], df_metrics["Adaptive_Recall10_PureADC"], label="AO-PQ (Pure ADC Only)", color="#5cb85c", linestyle="--", linewidth=1.6)
    plt.plot(df_metrics["Batch"], df_metrics["Static_Recall10_PureADC"], label="Static PQ (Pure ADC Only)", color="#d9534f", linestyle=":", linewidth=1.6)
    plt.axhline(85.0, color="gray", linestyle="--", alpha=0.7, label="SLA Target (85%)")
    plt.title("SIFT-1M (1,000,000 Vectors) Recall@10: Algorithmic & Pipeline Ablation", fontweight="bold", fontsize=11)
    plt.xlabel("Streaming Batch Number (10,000 vectors/batch)")
    plt.ylabel("Empirical Recall@10 (%)")
    plt.ylim(0, 105)
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig2_recall_retention.png", dpi=300)
    plt.close()

    # Figure 3: Pure ADC Latency Distribution
    plt.figure(figsize=(7, 4))
    df_box = pd.DataFrame({
        "Static PQ (Pure ADC)": static_individual_latencies,
        "AO-PQ (Pure ADC Multi-Epoch)": adp_individual_latencies
    })
    sns.boxplot(data=df_box, palette=["#1f77b4", "#5cb85c"], showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
    plt.title(f"SIFT-1M Pure ADC Latency Distribution (N={len(adp_individual_latencies):,} Queries)", fontweight="bold", fontsize=11)
    plt.ylabel("Query Latency (ms)")
    plt.xlabel("System")
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.tight_layout()
    plt.savefig("results/sift_fig3_latency_distribution.png", dpi=300)
    plt.close()

    # Figure 4: Search Scalability vs Segmented Store
    plt.figure(figsize=(8, 4))
    plt.plot(df_metrics["Total_Vectors"], df_metrics["Segmented_Latency_ms"], label="Segmented PQ Store (Scatter-Gather)", color="#d9534f", linestyle="--", linewidth=1.8)
    plt.plot(df_metrics["Total_Vectors"], df_metrics["Adaptive_Latency_ms"], label="Unified AO-PQ (JIT Multi-Epoch)", color="#0275d8", linewidth=2.2)
    plt.title("SIFT-1M Pure ADC Search Latency Scaling (Segmented vs Unified)", fontweight="bold", fontsize=11)
    plt.xlabel("Total Indexed Vectors in Memory (100% Coverage)")
    plt.ylabel("Query Latency (ms)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/sift_fig4_scalability.png", dpi=300)
    plt.close()

    print("\n[SUCCESS] Full SIFT-1M Scale Benchmark Completed (1,000,000 Vectors).")
    print(f" -> Total Codebook Swaps: {adaptive_engine.total_swaps}")
    print(f" -> Total Background Compactions: {adaptive_engine.total_compactions}")


if __name__ == "__main__":
    run_sift_benchmark()
