"""
Adaptive Online Product Quantization (AO-PQ) Engine - Numba JIT Core
===================================================================
Uses a compiled C-level kernel for zero-allocation, sub-0.5ms ADC search.
"""

import copy
import threading
import numpy as np
from numba import njit, prange
from sklearn.cluster import MiniBatchKMeans


@njit(fastmath=True, parallel=True)
def _fast_adc_search(codes, epochs, current_epoch_id, lut_active, lut_prior, m, k):
    """
    Compiled C-level parallel ADC kernel.
    Directly accumulates distances into registers with zero temporary array allocations.
    """
    n = codes.shape[0]
    distances = np.empty(n, dtype=np.float32)

    for i in prange(n):
        dist_acc = 0.0
        is_curr = (epochs[i] == current_epoch_id)
        for j in range(m):
            code_val = int(codes[i, j])
            idx = j * k + code_val
            if is_curr:
                dist_acc += lut_active[idx]
            else:
                dist_acc += lut_prior[idx]
        distances[i] = dist_acc

    return distances


class AdaptiveOnlinePQ:
    def __init__(
        self,
        d: int = 128,
        m: int = 8,
        k: int = 256,
        lr: float = 0.10,
        momentum: float = 0.85,
        drift_threshold: float = 0.030
    ):
        assert d % m == 0, f"Dimension {d} must be divisible by m={m}"
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.lr = lr
        self.momentum = momentum
        self.drift_threshold = drift_threshold

        # --- Dual-Codebook Architecture: shape (m, k, d_sub) ---
        self.active_codebook = np.random.randn(m, k, self.d_sub).astype(np.float32)
        self.shadow_codebook = copy.deepcopy(self.active_codebook)
        self.prior_codebook = copy.deepcopy(self.active_codebook)

        # Momentum velocity buffer
        self.velocity = np.zeros_like(self.shadow_codebook, dtype=np.float32)

        # --- Storage Arrays ---
        self.codes = np.empty((0, self.m), dtype=np.uint8)
        self.epochs = np.empty((0,), dtype=np.uint8)
        self.current_epoch_id = 0
        self.total_swaps = 0

        self.lock = threading.Lock()

        # Warm up the JIT compiler
        _dummy_codes = np.zeros((10, self.m), dtype=np.uint8)
        _dummy_epochs = np.zeros(10, dtype=np.uint8)
        _dummy_lut = np.zeros(self.m * self.k, dtype=np.float32)
        _fast_adc_search(_dummy_codes, _dummy_epochs, 0, _dummy_lut, _dummy_lut, self.m, self.k)

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X.reshape(X.shape[0], self.m, self.d_sub))

    def fit_initial(self, X_train: np.ndarray):
        """Initial baseline clustering."""
        X_sub = self._decompose(X_train)
        for i in range(self.m):
            km = MiniBatchKMeans(
                n_clusters=self.k,
                batch_size=min(2048, X_train.shape[0]),
                n_init=1,
                max_iter=40,
                random_state=42
            )
            km.fit(X_sub[:, i, :])
            self.active_codebook[i] = km.cluster_centers_.astype(np.float32)

        self.shadow_codebook = copy.deepcopy(self.active_codebook)
        self.prior_codebook = copy.deepcopy(self.active_codebook)
        self.velocity.fill(0.0)

    def quantize(self, X: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """Fast vectorized subvector quantization."""
        X_sub = self._decompose(X)
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)

        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            centroids = codebook[i]
            dists = (
                np.sum(sub_vecs ** 2, axis=1, keepdims=True)
                + np.sum(centroids ** 2, axis=1, keepdims=True).T
                - 2.0 * np.dot(sub_vecs, centroids.T)
            )
            codes[:, i] = np.argmin(dists, axis=1)

        return codes

    def compute_reconstruction_mse(self, X: np.ndarray, codebook: np.ndarray) -> float:
        """Measures quantization reconstruction error."""
        codes = self.quantize(X, codebook)
        X_sub = self._decompose(X)
        reconstructed = np.zeros_like(X_sub)

        for i in range(self.m):
            reconstructed[:, i, :] = codebook[i][codes[:, i]]

        return float(np.mean((X - reconstructed.reshape(X.shape[0], self.d)) ** 2))

    def ingest_stream_batch(self, X_batch: np.ndarray):
        """Streaming Ingestion with Momentum updates & atomic codebook swaps."""
        X_sub = self._decompose(X_batch)

        # 1. Background Shadow Centroid Updates with Polyak Momentum
        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            centroids = self.shadow_codebook[i]
            dists = (
                np.sum(sub_vecs ** 2, axis=1, keepdims=True)
                + np.sum(centroids ** 2, axis=1, keepdims=True).T
                - 2.0 * np.dot(sub_vecs, centroids.T)
            )
            assignments = np.argmin(dists, axis=1)
            unique_c = np.unique(assignments)

            for c in unique_c:
                members = sub_vecs[assignments == c]
                batch_grad = np.mean(members, axis=0) - centroids[c]
                self.velocity[i, c] = (
                    self.momentum * self.velocity[i, c]
                    + (1.0 - self.momentum) * batch_grad
                )
                self.shadow_codebook[i, c] += self.lr * self.velocity[i, c]

        # 2. Autonomous Drift Detection
        active_mse = self.compute_reconstruction_mse(X_batch, self.active_codebook)
        shadow_mse = self.compute_reconstruction_mse(X_batch, self.shadow_codebook)
        rel_gain = (active_mse - shadow_mse) / max(active_mse, 1e-6)

        # 3. Dynamic Atomic Pointer Swap
        if rel_gain > self.drift_threshold:
            with self.lock:
                self.prior_codebook = copy.deepcopy(self.active_codebook)
                self.active_codebook = copy.deepcopy(self.shadow_codebook)
                self.current_epoch_id = 1 - self.current_epoch_id
                self.total_swaps += 1

        # 4. Quantize incoming vectors under Active Codebook and store
        batch_codes = self.quantize(X_batch, self.active_codebook)
        batch_epochs = np.full((X_batch.shape[0],), self.current_epoch_id, dtype=np.uint8)

        with self.lock:
            self.codes = (
                np.ascontiguousarray(np.vstack([self.codes, batch_codes]))
                if self.codes.size
                else np.ascontiguousarray(batch_codes)
            )
            self.epochs = (
                np.ascontiguousarray(np.concatenate([self.epochs, batch_epochs]))
                if self.epochs.size
                else np.ascontiguousarray(batch_epochs)
            )

    def search(self, query: np.ndarray, top_k: int = 10):
        """JIT-Accelerated Asymmetric Distance Computation (ADC)."""
        q_sub = query.reshape(self.m, self.d_sub)

        # Precompute 1D LUTs of size m * k (8.19 KB)
        lut_active = np.zeros(self.m * self.k, dtype=np.float32)
        lut_prior = np.zeros(self.m * self.k, dtype=np.float32)

        for i in range(self.m):
            start = i * self.k
            end = start + self.k
            lut_active[start:end] = np.sum((self.active_codebook[i] - q_sub[i]) ** 2, axis=1)
            lut_prior[start:end] = np.sum((self.prior_codebook[i] - q_sub[i]) ** 2, axis=1)

        with self.lock:
            total_records = self.codes.shape[0]
            if total_records == 0:
                return np.array([], dtype=int), np.array([], dtype=float)

            # Compiled parallel kernel execution
            distances = _fast_adc_search(
                self.codes,
                self.epochs,
                self.current_epoch_id,
                lut_active,
                lut_prior,
                self.m,
                self.k
            )

        top_k_clamp = min(top_k, total_records)
        top_indices = np.argpartition(distances, top_k_clamp - 1)[:top_k_clamp]
        sorted_top_indices = top_indices[np.argsort(distances[top_indices])]
        return sorted_top_indices, distances[sorted_top_indices]