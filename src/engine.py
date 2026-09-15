"""
Adaptive Online Product Quantization (AO-PQ) Engine - v4.1 Dynamic Core
=======================================================================
Features:
- Immutable Codebook Snapshots with 64-bit Monotonic Epoch IDs.
- Dynamic Chunk Arena Storage (geometric buffer expansion).
- Thread-safe JIT Distance Scanning with GIL released (nogil=True).
- Dual-Mode Search: search() (Two-Stage Refinement) & search_adc_only() (Pure ADC).
- Robust drift thresholding with absolute delta guards.
"""

import os
import sys
import copy
import time
import queue
import enum
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from numba import njit
from sklearn.cluster import MiniBatchKMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.storage import ChunkedVectorStorage, ImmutableChunk


@njit(fastmath=True, nogil=True)
def _fast_unified_multi_epoch_adc(codes, epochs, epoch_lut_matrix, epoch_id_map, m, k):
    """
    SIMD-vectorized Multi-Epoch Distance Scanner with GIL released.
    Thread-safe for concurrent multi-reader access.
    """
    n = codes.shape[0]
    distances = np.empty(n, dtype=np.float32)
    num_epochs = epoch_id_map.shape[0]

    for i in range(n):
        e_id = epochs[i]

        lut_idx = -1
        for slot in range(num_epochs):
            if epoch_id_map[slot] == e_id:
                lut_idx = slot
                break

        if lut_idx == -1:
            distances[i] = np.nan
            continue

        dist_acc = 0.0
        for j in range(m):
            code_val = int(codes[i, j])
            dist_acc += epoch_lut_matrix[lut_idx, j, code_val]
        distances[i] = dist_acc

    return distances


@njit(fastmath=True, nogil=True)
def _fast_exact_rerank(candidate_vectors, query):
    """Euclidean distance refinement on extracted candidate vectors with GIL released."""
    k_cand = candidate_vectors.shape[0]
    d = candidate_vectors.shape[1]
    exact_dists = np.empty(k_cand, dtype=np.float32)

    for i in range(k_cand):
        acc = 0.0
        for dim in range(d):
            diff = candidate_vectors[i, dim] - query[dim]
            acc += diff * diff
        exact_dists[i] = acc

    return exact_dists


class EpochState(enum.Enum):
    PREPARING = 0
    ACTIVE = 1
    RETIRED = 2
    MIGRATING = 3
    RECLAIMABLE = 4
    PURGED = 5


class CodebookSnapshot:
    """Immutable snapshot of a PQ codebook at a specific epoch generation."""
    def __init__(self, epoch_id: int, centroids: np.ndarray):
        self.epoch_id = np.uint64(epoch_id)
        self._data = np.ascontiguousarray(centroids, dtype=np.float32).copy()
        self._data.flags.writeable = False
        self.created_at = time.time()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape


class EpochManager:
    """Coordinates monotonic epoch progression, lifetime states, and reclamation."""
    def __init__(self, max_active_window: int = 4):
        self.max_active_window = max_active_window
        self.registry: Dict[int, CodebookSnapshot] = {}
        self.epoch_states: Dict[int, EpochState] = {}
        self.ref_counts: Dict[int, int] = {}
        self.active_epoch_id: int = 0
        self.lock = threading.RLock()

    def register_initial_epoch(self, codebook_data: np.ndarray) -> CodebookSnapshot:
        with self.lock:
            snapshot = CodebookSnapshot(0, codebook_data)
            self.registry[0] = snapshot
            self.epoch_states[0] = EpochState.ACTIVE
            self.ref_counts[0] = 0
            self.active_epoch_id = 0
            return snapshot

    def prepare_snapshot(self, next_epoch_id: int, codebook_data: np.ndarray) -> CodebookSnapshot:
        return CodebookSnapshot(next_epoch_id, codebook_data)

    def publish_promoted_snapshot(self, snapshot: CodebookSnapshot):
        """Atomic publication critical section."""
        with self.lock:
            new_id = int(snapshot.epoch_id)
            prev_id = self.active_epoch_id

            self.registry[new_id] = snapshot
            self.epoch_states[new_id] = EpochState.ACTIVE
            self.ref_counts[new_id] = 0

            if prev_id in self.epoch_states:
                self.epoch_states[prev_id] = EpochState.RETIRED

            self.active_epoch_id = new_id

    def get_search_snapshot_view(self) -> Dict[int, CodebookSnapshot]:
        """Provides an immutable dictionary view of all searchable snapshots."""
        with self.lock:
            searchable = {}
            for e_id, snap in self.registry.items():
                state = self.epoch_states.get(e_id, EpochState.PURGED)
                refs = self.ref_counts.get(e_id, 0)
                if state in (EpochState.ACTIVE, EpochState.RETIRED, EpochState.MIGRATING) or refs > 0:
                    searchable[e_id] = snap
            return searchable

    def increment_ref(self, epoch_id: int, count: int = 1):
        with self.lock:
            self.ref_counts[epoch_id] = self.ref_counts.get(epoch_id, 0) + count

    def decrement_ref(self, epoch_id: int, count: int = 1):
        with self.lock:
            if epoch_id in self.ref_counts:
                self.ref_counts[epoch_id] = max(0, self.ref_counts[epoch_id] - count)
                if self.ref_counts[epoch_id] == 0:
                    if self.epoch_states.get(epoch_id) == EpochState.MIGRATING:
                        self.epoch_states[epoch_id] = EpochState.RECLAIMABLE
                        self._attempt_reclaim(epoch_id)

    def _attempt_reclaim(self, epoch_id: int):
        with self.lock:
            if (self.epoch_states.get(epoch_id) == EpochState.RECLAIMABLE and 
                self.ref_counts.get(epoch_id, 0) == 0):
                self.epoch_states[epoch_id] = EpochState.PURGED
                if epoch_id in self.registry:
                    del self.registry[epoch_id]

    def get_migration_candidates(self) -> List[int]:
        with self.lock:
            active_ids = sorted([
                e for e, state in self.epoch_states.items() 
                if state in (EpochState.ACTIVE, EpochState.RETIRED)
            ])
            candidates = []
            if len(active_ids) > self.max_active_window:
                excess = len(active_ids) - self.max_active_window
                for e_id in active_ids[:excess]:
                    if self.epoch_states[e_id] == EpochState.RETIRED:
                        self.epoch_states[e_id] = EpochState.MIGRATING
                        candidates.append(e_id)
            return candidates


class AdaptiveOnlinePQ:
    """
    Adaptive Online Product Quantization Engine (v4.1 Dynamic Core).
    Supports zero-downtime codebook replacement with dynamic arena storage.
    """
    def __init__(
        self,
        d: int = 128,
        m: int = 8,
        k: int = 256,
        lr: float = 0.12,
        momentum: float = 0.85,
        drift_threshold: float = 0.015,
        max_active_window: int = 4,
        chunk_capacity: int = 32768,
        initial_arena_capacity: int = 131072
    ):
        assert d % m == 0, f"Vector dimension {d} must be divisible by m={m}"
        self.d = d
        self.m = m
        self.d_sub = d // m
        self.k = k
        self.lr = lr
        self.momentum = momentum
        self.drift_threshold = drift_threshold

        self.epoch_manager = EpochManager(max_active_window=max_active_window)
        self.storage = ChunkedVectorStorage(
            chunk_capacity=chunk_capacity,
            m=m,
            initial_arena_capacity=initial_arena_capacity
        )

        self._raw_chunks: List[np.ndarray] = []
        self._raw_count = 0
        self._raw_lock = threading.Lock()

        self.shadow_codebook = np.zeros((m, k, self.d_sub), dtype=np.float32, order='C')
        self.velocity = np.zeros_like(self.shadow_codebook, dtype=np.float32, order='C')
        self.total_swaps = 0
        self.total_compactions = 0

        self.migration_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._migration_worker = threading.Thread(target=self._async_migration_loop, daemon=True)
        self._migration_worker.start()

        # Warm up JIT kernels
        _d_codes = np.zeros((10, self.m), dtype=np.uint8)
        _d_epochs = np.zeros(10, dtype=np.uint64)
        _d_lut = np.zeros((1, self.m, self.k), dtype=np.float32)
        _d_map = np.zeros(1, dtype=np.uint64)
        _fast_unified_multi_epoch_adc(_d_codes, _d_epochs, _d_lut, _d_map, self.m, self.k)

        _d_vecs = np.zeros((10, self.d), dtype=np.float32)
        _d_q = np.zeros(self.d, dtype=np.float32)
        _fast_exact_rerank(_d_vecs, _d_q)

    def _decompose(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X.reshape(X.shape[0], self.m, self.d_sub))

    def fit_initial(self, X_train: np.ndarray):
        X_sub = self._decompose(X_train)
        init_centroids = np.zeros((self.m, self.k, self.d_sub), dtype=np.float32, order='C')

        for i in range(self.m):
            km = MiniBatchKMeans(
                n_clusters=self.k,
                batch_size=min(2048, X_train.shape[0]),
                n_init=1,
                max_iter=40,
                random_state=42
            )
            km.fit(X_sub[:, i, :])
            init_centroids[i] = km.cluster_centers_.astype(np.float32)

        self.shadow_codebook = init_centroids.copy()
        self.shadow_codebook.flags.writeable = True
        self.velocity.fill(0.0)
        self.epoch_manager.register_initial_epoch(init_centroids)

    def quantize(self, X: np.ndarray, codebook_data: np.ndarray) -> np.ndarray:
        X_sub = self._decompose(X)
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)

        for i in range(self.m):
            sub_vecs = X_sub[:, i, :]
            centroids = codebook_data[i]
            dists = (
                np.sum(sub_vecs ** 2, axis=1, keepdims=True)
                + np.sum(centroids ** 2, axis=1, keepdims=True).T
                - 2.0 * np.dot(sub_vecs, centroids.T)
            )
            codes[:, i] = np.argmin(dists, axis=1)

        return codes

    def compute_reconstruction_mse(self, X: np.ndarray, codebook_data: np.ndarray) -> float:
        codes = self.quantize(X, codebook_data)
        X_sub = self._decompose(X)
        reconstructed = np.zeros_like(X_sub)

        for i in range(self.m):
            reconstructed[:, i, :] = codebook_data[i][codes[:, i]]

        return float(np.mean((X - reconstructed.reshape(X.shape[0], self.d)) ** 2))

    def ingest_stream_batch(self, X_batch: np.ndarray, force_swap: bool = False):
        X_sub = self._decompose(X_batch)
        N_batch = X_batch.shape[0]

        # 1. Update Shadow Centroids
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

        # 2. Evaluate Drift Against Active Snapshot
        curr_active_id = self.epoch_manager.active_epoch_id
        active_snapshot = self.epoch_manager.registry[curr_active_id]

        active_mse = self.compute_reconstruction_mse(X_batch, active_snapshot.data)
        shadow_mse = self.compute_reconstruction_mse(X_batch, self.shadow_codebook)
        
        mse_diff = active_mse - shadow_mse
        rel_gain = mse_diff / max(active_mse, 1e-6)

        # 3. Decoupled Promotion with stationary noise guard
        is_significant_drift = (rel_gain > self.drift_threshold) and (mse_diff > 1e-4)
        if is_significant_drift or force_swap:
            next_epoch_id = curr_active_id + 1
            prepared_snapshot = self.epoch_manager.prepare_snapshot(
                next_epoch_id, self.shadow_codebook
            )
            self.epoch_manager.publish_promoted_snapshot(prepared_snapshot)
            self.total_swaps += 1
            self.velocity.fill(0.0)

            candidates = self.epoch_manager.get_migration_candidates()
            for cand in candidates:
                self.migration_queue.put(cand)

        # 4. Ingest raw vectors into chunk list
        with self._raw_lock:
            self._raw_chunks.append(np.ascontiguousarray(X_batch, dtype=np.float32))
            self._raw_count += N_batch

        # 5. Storage Append
        active_id = self.epoch_manager.active_epoch_id
        active_snap = self.epoch_manager.registry[active_id]

        batch_codes = self.quantize(X_batch, active_snap.data)
        global_ids = np.arange(
            self.storage.total_records,
            self.storage.total_records + N_batch,
            dtype=np.uint64
        )

        self.storage.append_batch(batch_codes, active_id, global_ids)
        self.epoch_manager.increment_ref(active_id, N_batch)

    def _async_migration_loop(self):
        while not self._stop_event.is_set():
            try:
                target_epoch = self.migration_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._migrate_epoch_records(target_epoch)
            self.migration_queue.task_done()

    def _migrate_epoch_records(self, old_epoch_id: int):
        with self.epoch_manager.lock:
            if old_epoch_id not in self.epoch_manager.registry:
                return
            old_snapshot = self.epoch_manager.registry[old_epoch_id]
            target_epoch_id = self.epoch_manager.active_epoch_id
            target_snapshot = self.epoch_manager.registry[target_epoch_id]

        migrated_total = self.storage.migrate_all_records(
            old_epoch_id=old_epoch_id,
            target_epoch_id=target_epoch_id,
            old_snapshot_data=old_snapshot.data,
            target_snapshot_data=target_snapshot.data,
            quantize_fn=self.quantize,
            d=self.d,
            m=self.m,
            d_sub=self.d_sub
        )

        self.epoch_manager.decrement_ref(old_epoch_id, migrated_total)
        self.epoch_manager.increment_ref(target_epoch_id, migrated_total)
        self.total_compactions += 1

    def search_adc_only(self, query: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Pure Multi-Epoch ADC search without candidate refinement."""
        assert query.shape[0] == self.d, f"Query dimension mismatch: expected {self.d}"
        q_sub = query.reshape(self.m, self.d_sub)

        flat_codes, flat_epochs, flat_gids, total_n = self.storage.get_unified_search_view()
        if total_n == 0:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)

        active_snapshots = self.epoch_manager.get_search_snapshot_view()
        epoch_ids = list(active_snapshots.keys())
        epoch_id_map = np.array(epoch_ids, dtype=np.uint64)
        epoch_lut_matrix = np.zeros((len(epoch_ids), self.m, self.k), dtype=np.float32)

        for slot, e_id in enumerate(epoch_ids):
            cb_data = active_snapshots[e_id].data
            for sub_i in range(self.m):
                epoch_lut_matrix[slot, sub_i] = np.sum((cb_data[sub_i] - q_sub[sub_i]) ** 2, axis=1)

        all_dists = _fast_unified_multi_epoch_adc(
            flat_codes, flat_epochs, epoch_lut_matrix, epoch_id_map, self.m, self.k
        )

        if np.isnan(all_dists).any():
            active_snapshots = self.epoch_manager.get_search_snapshot_view()
            epoch_ids = list(active_snapshots.keys())
            epoch_id_map = np.array(epoch_ids, dtype=np.uint64)
            epoch_lut_matrix = np.zeros((len(epoch_ids), self.m, self.k), dtype=np.float32)
            for slot, e_id in enumerate(epoch_ids):
                cb_data = active_snapshots[e_id].data
                for sub_i in range(self.m):
                    epoch_lut_matrix[slot, sub_i] = np.sum((cb_data[sub_i] - q_sub[sub_i]) ** 2, axis=1)
            all_dists = _fast_unified_multi_epoch_adc(
                flat_codes, flat_epochs, epoch_lut_matrix, epoch_id_map, self.m, self.k
            )

        final_k = min(top_k, total_n)
        top_indices = np.argpartition(all_dists, final_k - 1)[:final_k]
        sorted_order = top_indices[np.argsort(all_dists[top_indices])]
        return flat_gids[sorted_order], all_dists[sorted_order]

    def _get_raw_vector(self, gid: int) -> np.ndarray:
        """Retrieves raw vector from chunk list by global ID."""
        curr = 0
        for chunk in self._raw_chunks:
            if gid < curr + chunk.shape[0]:
                return chunk[gid - curr]
            curr += chunk.shape[0]
        raise IndexError(f"Global ID {gid} out of range in raw vector store.")

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        candidate_pool: int = 80
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-stage search with multi-epoch ADC scan and Euclidean refinement."""
        assert query.shape[0] == self.d, f"Query dimension mismatch: expected {self.d}"
        q_sub = query.reshape(self.m, self.d_sub)

        flat_codes, flat_epochs, flat_gids, total_n = self.storage.get_unified_search_view()
        if total_n == 0:
            return np.array([], dtype=np.uint64), np.array([], dtype=np.float32)

        active_snapshots = self.epoch_manager.get_search_snapshot_view()
        epoch_ids = list(active_snapshots.keys())
        num_epochs = len(epoch_ids)
        epoch_id_map = np.array(epoch_ids, dtype=np.uint64)
        epoch_lut_matrix = np.zeros((num_epochs, self.m, self.k), dtype=np.float32)

        for slot, e_id in enumerate(epoch_ids):
            cb_data = active_snapshots[e_id].data
            for sub_i in range(self.m):
                epoch_lut_matrix[slot, sub_i] = np.sum((cb_data[sub_i] - q_sub[sub_i]) ** 2, axis=1)

        all_dists = _fast_unified_multi_epoch_adc(
            flat_codes, flat_epochs, epoch_lut_matrix, epoch_id_map, self.m, self.k
        )

        if np.isnan(all_dists).any():
            active_snapshots = self.epoch_manager.get_search_snapshot_view()
            epoch_ids = list(active_snapshots.keys())
            epoch_id_map = np.array(epoch_ids, dtype=np.uint64)
            epoch_lut_matrix = np.zeros((len(epoch_ids), self.m, self.k), dtype=np.float32)
            for slot, e_id in enumerate(epoch_ids):
                cb_data = active_snapshots[e_id].data
                for sub_i in range(self.m):
                    epoch_lut_matrix[slot, sub_i] = np.sum((cb_data[sub_i] - q_sub[sub_i]) ** 2, axis=1)
            all_dists = _fast_unified_multi_epoch_adc(
                flat_codes, flat_epochs, epoch_lut_matrix, epoch_id_map, self.m, self.k
            )

        candidate_k = min(max(top_k * 4, candidate_pool), total_n)
        candidate_indices = np.argpartition(all_dists, candidate_k - 1)[:candidate_k]
        candidate_ids = flat_gids[candidate_indices]

        # Exact Refinement on retrieved candidates
        with self._raw_lock:
            if candidate_ids.size > 0 and candidate_ids.max() < self._raw_count:
                candidate_vecs = np.vstack([self._get_raw_vector(int(cid)) for cid in candidate_ids])
                exact_dists = _fast_exact_rerank(candidate_vecs, query)

                final_k = min(top_k, candidate_k)
                best_local = np.argpartition(exact_dists, final_k - 1)[:final_k]
                sorted_order = best_local[np.argsort(exact_dists[best_local])]

                return candidate_ids[sorted_order], exact_dists[sorted_order]

        final_k = min(top_k, candidate_k)
        sorted_order = candidate_indices[np.argsort(all_dists[candidate_indices])][:final_k]
        return flat_gids[sorted_order], all_dists[sorted_order]

    def close(self):
        self._stop_event.set()
        if self._migration_worker.is_alive():
            self._migration_worker.join(timeout=1.0)
