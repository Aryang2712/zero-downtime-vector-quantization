"""
Streaming Baselines & Pure/Reranked Search Evaluation (AO-PQ v3.7)
=================================================================
Includes complete index coverage across sealed segments AND unsealed current_buf.
"""

import os
import sys
import time
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import MiniBatchKMeans


class StaticPQBaseline:
    """Static Single-Codebook PQ Baseline."""
    def __init__(self, d: int = 128, m: int = 8, k: int = 256):
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.centroids = np.zeros((m, k, self.d_sub), dtype=np.float32)
        self.codes: List[np.ndarray] = []
        self.raw_vectors: List[np.ndarray] = []
        self.total_records = 0

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X.reshape(X.shape[0], self.m, self.d_sub))

    def fit(self, X_train: np.ndarray):
        X_sub = self._decompose(X_train)
        for i in range(self.m):
            km = MiniBatchKMeans(n_clusters=self.k, batch_size=2048, random_state=42)
            km.fit(X_sub[:, i, :])
            self.centroids[i] = km.cluster_centers_.astype(np.float32)

    def quantize(self, X: np.ndarray) -> np.ndarray:
        X_sub = self._decompose(X)
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)
        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            c = self.centroids[i]
            dists = np.sum(sub_vecs**2, axis=1, keepdims=True) + np.sum(c**2, axis=1, keepdims=True).T - 2.0 * np.dot(sub_vecs, c.T)
            codes[:, i] = np.argmin(dists, axis=1)
        return codes

    def compute_reconstruction_mse(self, X: np.ndarray) -> float:
        codes = self.quantize(X)
        X_sub = self._decompose(X)
        reconstructed = np.zeros_like(X_sub)
        for i in range(self.m):
            reconstructed[:, i, :] = self.centroids[i][codes[:, i]]
        return float(np.mean((X - reconstructed.reshape(X.shape[0], self.d)) ** 2))

    def ingest(self, X_batch: np.ndarray):
        codes = self.quantize(X_batch)
        self.codes.append(codes)
        self.raw_vectors.append(X_batch)
        self.total_records += X_batch.shape[0]

    def search(self, query: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Pure ADC Search on Static PQ."""
        q_sub = query.reshape(self.m, self.d_sub)
        lut = np.zeros((self.m, self.k), dtype=np.float32)
        for i in range(self.m):
            lut[i] = np.sum((self.centroids[i] - q_sub[i]) ** 2, axis=1)

        all_codes = np.vstack(self.codes)
        n = all_codes.shape[0]
        dists = np.zeros(n, dtype=np.float32)
        for j in range(self.m):
            dists += lut[j, all_codes[:, j]]

        final_k = min(top_k, n)
        top_idx = np.argpartition(dists, final_k - 1)[:final_k]
        return top_idx[np.argsort(dists[top_idx])]

    def search_reranked(self, query: np.ndarray, top_k: int = 10, candidate_pool: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """Matched Two-Stage Search on Static PQ Baseline."""
        all_raw = np.vstack(self.raw_vectors)
        candidate_k = min(max(top_k * 4, candidate_pool), self.total_records)
        cand_indices = self.search(query, top_k=candidate_k)

        cand_vecs = all_raw[cand_indices]
        exact_dists = np.sum((cand_vecs - query) ** 2, axis=1)

        final_k = min(top_k, candidate_k)
        best_local = np.argpartition(exact_dists, final_k - 1)[:final_k]
        sorted_order = best_local[np.argsort(exact_dists[best_local])]
        return cand_indices[sorted_order], exact_dists[sorted_order]


class SegmentedPQBaseline:
    """
    Segmented PQ Baseline with Scatter-Gather Search Scanning.
    Guarantees full index coverage by querying sealed segments AND unsealed current_buf.
    """
    def __init__(self, d: int = 128, m: int = 8, k: int = 256, segment_size: int = 5000):
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.segment_size = segment_size
        self.segments: List[Tuple[np.ndarray, np.ndarray]] = []
        self.current_buf: List[np.ndarray] = []
        self.current_buf_centroids: Optional[np.ndarray] = None
        self.total_records = 0

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X.reshape(X.shape[0], self.m, self.d_sub))

    def ingest(self, X_batch: np.ndarray):
        self.current_buf.append(X_batch)
        buf_len = sum(b.shape[0] for b in self.current_buf)
        
        if buf_len >= self.segment_size:
            seg_data = np.vstack(self.current_buf)
            static_seg = StaticPQBaseline(self.d, self.m, self.k)
            static_seg.fit(seg_data)
            static_seg.ingest(seg_data)
            self.segments.append((static_seg.centroids, np.vstack(static_seg.codes)))
            self.current_buf = []
            self.current_buf_centroids = None
        else:
            # Maintain active sub-centroid view for unsealed buffer
            buf_data = np.vstack(self.current_buf)
            if buf_data.shape[0] >= self.k:
                static_active = StaticPQBaseline(self.d, self.m, self.k)
                static_active.fit(buf_data)
                self.current_buf_centroids = static_active.centroids
            elif len(self.segments) > 0:
                self.current_buf_centroids = self.segments[-1][0]
            else:
                self.current_buf_centroids = np.zeros((self.m, self.k, self.d_sub), dtype=np.float32)

        self.total_records += X_batch.shape[0]

    def search(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scatter-gather multi-segment search scan across ALL indexed vectors
        (Sealed Segments + Active Buffer).
        """
        if self.total_records == 0:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)

        q_sub = query.reshape(self.m, self.d_sub)
        all_dists, all_ids = [], []
        curr_offset = 0

        # 1. Scan Sealed Segments
        for centroids, codes in self.segments:
            lut = np.zeros((self.m, self.k), dtype=np.float32)
            for i in range(self.m):
                lut[i] = np.sum((centroids[i] - q_sub[i]) ** 2, axis=1)

            n_seg = codes.shape[0]
            dists = np.zeros(n_seg, dtype=np.float32)
            for j in range(self.m):
                dists += lut[j, codes[:, j]]

            all_dists.append(dists)
            all_ids.append(np.arange(curr_offset, curr_offset + n_seg, dtype=np.uint64))
            curr_offset += n_seg

        # 2. Scan Unsealed Current Buffer (Ensures 100% vector coverage)
        if len(self.current_buf) > 0:
            buf_data = np.vstack(self.current_buf)
            n_buf = buf_data.shape[0]
            buf_sub = self._decompose(buf_data)
            
            c_buf = self.current_buf_centroids if self.current_buf_centroids is not None else np.zeros((self.m, self.k, self.d_sub), dtype=np.float32)
            lut = np.zeros((self.m, self.k), dtype=np.float32)
            for i in range(self.m):
                lut[i] = np.sum((c_buf[i] - q_sub[i]) ** 2, axis=1)

            # Quantize unsealed batch
            buf_codes = np.zeros((n_buf, self.m), dtype=np.uint8)
            for i in range(self.m):
                sub_vecs = buf_sub[:, i, :]
                dists_c = np.sum(sub_vecs**2, axis=1, keepdims=True) + np.sum(c_buf[i]**2, axis=1, keepdims=True).T - 2.0 * np.dot(sub_vecs, c_buf[i].T)
                buf_codes[:, i] = np.argmin(dists_c, axis=1)

            buf_dists = np.zeros(n_buf, dtype=np.float32)
            for j in range(self.m):
                buf_dists += lut[j, buf_codes[:, j]]

            all_dists.append(buf_dists)
            all_ids.append(np.arange(curr_offset, curr_offset + n_buf, dtype=np.uint64))

        merged_dists = np.concatenate(all_dists)
        merged_ids = np.concatenate(all_ids)

        final_k = min(top_k, merged_dists.shape[0])
        top_idx = np.argpartition(merged_dists, final_k - 1)[:final_k]
        sorted_order = top_idx[np.argsort(merged_dists[top_idx])]
        return merged_ids[sorted_order], merged_dists[sorted_order]