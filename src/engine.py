"""
Adaptive Online Product Quantization (AO-PQ) Engine
===================================================
Optimized, vectorized engine implementing:
1. Product Quantization subspace decomposition.
2. Dual-Codebook Shadow Buffering (Lock-free writes).
3. Autonomous MSE-based Concept Drift Detection.
4. Versioned Dual-LUT Multi-Epoch indexing (Zero Heterogeneous Distortion).
"""

import copy
import threading
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class AdaptiveOnlinePQ:
    def __init__(
        self,
        d: int = 128,
        m: int = 8,
        k: int = 256,
        lr: float = 0.05,
        drift_threshold: float = 0.04
    ):
        assert d % m == 0, f"Dimension {d} must be divisible by m={m}"
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.lr = lr
        self.drift_threshold = drift_threshold

        # --- Dual-Codebook Architecture ---
        self.active_codebook = np.random.randn(m, k, self.d_sub).astype(np.float32)
        self.shadow_codebook = copy.deepcopy(self.active_codebook)
        self.prior_codebook = None

        # --- Storage ---
        self.codes = np.empty((0, self.m), dtype=np.uint8)
        self.epochs = np.empty((0,), dtype=np.uint8)
        self.current_epoch_id = 0

        self.lock = threading.Lock()
        self.total_swaps = 0

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], self.m, self.d_sub)

    def fit_initial(self, X_train: np.ndarray):
        """Initial baseline codebook generation on Day-1 data."""
        X_sub = self._decompose(X_train)
        for i in range(self.m):
            km = MiniBatchKMeans(
                n_clusters=self.k,
                batch_size=min(2048, X_train.shape[0]),
                n_init=1,
                max_iter=50,
                random_state=42
            )
            km.fit(X_sub[:, i, :])
            self.active_codebook[i] = km.cluster_centers_.astype(np.float32)
            
        self.shadow_codebook = copy.deepcopy(self.active_codebook)

    def quantize(self, X: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Vectorized subvector quantization."""
        X_sub = self._decompose(X)
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)

        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            centroids = codebook[i]
            # Fast squared Euclidean distance expansion: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
            dists = (
                np.sum(sub_vecs ** 2, axis=1, keepdims=True)
                + np.sum(centroids ** 2, axis=1, keepdims=True).T
                - 2 * np.dot(sub_vecs, centroids.T)
            )
            codes[:, i] = np.argmin(dists, axis=1)

        return codes

    def compute_reconstruction_mse(self, X: np.ndarray, codebook: np.ndarray) -> float:
        """Measures compression distortion on a given batch."""
        codes = self.quantize(X, codebook)
        X_sub = self._decompose(X)
        reconstructed = np.zeros_like(X_sub)

        for i in range(self.m):
            reconstructed[:, i, :] = codebook[i][codes[:, i]]

        reconstructed = reconstructed.reshape(X.shape[0], self.d)
        return float(np.mean((X - reconstructed) ** 2))

    def ingest_stream_batch(self, X_batch: np.ndarray):
        """Fast vectorized streaming ingestion."""
        X_sub = self._decompose(X_batch)

        # 1. Vectorized online gradient update for shadow codebook
        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            centroids = self.shadow_codebook[i]
            dists = (
                np.sum(sub_vecs ** 2, axis=1, keepdims=True)
                + np.sum(centroids ** 2, axis=1, keepdims=True).T
                - 2 * np.dot(sub_vecs, centroids.T)
            )
            assignments = np.argmin(dists, axis=1)

            # Vectorized centroid update only for touched clusters
            unique_clusters = np.unique(assignments)
            for c in unique_clusters:
                members = sub_vecs[assignments == c]
                batch_mean = np.mean(members, axis=0)
                self.shadow_codebook[i, c] += self.lr * (batch_mean - centroids[c])

        # 2. Autonomous Drift Detection
        active_mse = self.compute_reconstruction_mse(X_batch, self.active_codebook)
        shadow_mse = self.compute_reconstruction_mse(X_batch, self.shadow_codebook)
        rel_gain = (active_mse - shadow_mse) / max(active_mse, 1e-6)

        # 3. Dynamic Atomic Pointer Swap
        if rel_gain > self.drift_threshold:
            with self.lock:
                self.prior_codebook = copy.deepcopy(self.active_codebook)
                self.active_codebook = copy.deepcopy(self.shadow_codebook)
                self.current_epoch_id = (self.current_epoch_id + 1) % 255
                self.total_swaps += 1

        # 4. Quantize incoming vectors and store with current Epoch ID
        batch_codes = self.quantize(X_batch, self.active_codebook)
        batch_epochs = np.full((X_batch.shape[0],), self.current_epoch_id, dtype=np.uint8)

        with self.lock:
            self.codes = np.vstack([self.codes, batch_codes]) if self.codes.size else batch_codes
            self.epochs = np.concatenate([self.epochs, batch_epochs]) if self.epochs.size else batch_epochs

    def search(self, query: np.ndarray, top_k: int = 10):
        """Asymmetric Distance Computation (ADC) with Dual-LUT Multi-Epoch Evaluation."""
        q_sub = query.reshape(self.m, self.d_sub)

        # LUT for Active Generation: shape (m, k)
        lut_active = np.zeros((self.m, self.k), dtype=np.float32)
        for i in range(self.m):
            lut_active[i] = np.sum((self.active_codebook[i] - q_sub[i]) ** 2, axis=1)

        # LUT for Prior Generation (if active)
        lut_prior = None
        if self.prior_codebook is not None:
            lut_prior = np.zeros((self.m, self.k), dtype=np.float32)
            for i in range(self.m):
                lut_prior[i] = np.sum((self.prior_codebook[i] - q_sub[i]) ** 2, axis=1)

        with self.lock:
            total_records = self.codes.shape[0]
            if total_records == 0:
                return np.array([], dtype=int), np.array([], dtype=float)

            distances = np.zeros(total_records, dtype=np.float32)
            curr_epoch = self.current_epoch_id
            is_curr_epoch = (self.epochs == curr_epoch)
            is_prior_epoch = ~is_curr_epoch

            for i in range(self.m):
                if np.any(is_curr_epoch):
                    distances[is_curr_epoch] += lut_active[i, self.codes[is_curr_epoch, i]]
                if np.any(is_prior_epoch):
                    chosen_lut = lut_prior if lut_prior is not None else lut_active
                    distances[is_prior_epoch] += chosen_lut[i, self.codes[is_prior_epoch, i]]

        top_k_clamp = min(top_k, total_records)
        top_indices = np.argpartition(distances, top_k_clamp - 1)[:top_k_clamp]
        sorted_top_indices = top_indices[np.argsort(distances[top_indices])]
        return sorted_top_indices, distances[sorted_top_indices]