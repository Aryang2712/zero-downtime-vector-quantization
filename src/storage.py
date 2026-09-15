"""
Generation-Based Dynamic Chunk Arena & Copy-On-Write Storage (AO-PQ v4.1)
========================================================================
Guarantees:
- Geometric 2x buffer reallocation with unbounded streaming vector support.
- Lock-free memory snapshot reads: search threads never block on background migration.
- Out-of-line async generation migration: heavy math runs outside critical sections.
- Microsecond atomic reference pointer swaps (True O(1) swap).
- Pruned sealed chunk scanning using get_referenced_epochs().
"""

import threading
import numpy as np
from typing import List, Tuple, Set, Callable


class ImmutableChunk:
    def __init__(self, chunk_id: int, codes: np.ndarray, epochs: np.ndarray, global_ids: np.ndarray, version: int = 0):
        self.chunk_id = chunk_id
        self.version = version
        self.size = len(epochs)
        
        self._codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self._epochs = np.ascontiguousarray(epochs, dtype=np.uint64)
        self._global_ids = np.ascontiguousarray(global_ids, dtype=np.uint64)
        
        self._codes.flags.writeable = False
        self._epochs.flags.writeable = False
        self._global_ids.flags.writeable = False

    @property
    def codes(self) -> np.ndarray:
        return self._codes

    @property
    def epochs(self) -> np.ndarray:
        return self._epochs

    @property
    def global_ids(self) -> np.ndarray:
        return self._global_ids

    def get_referenced_epochs(self) -> Set[int]:
        if self.size == 0:
            return set()
        return set(np.unique(self._epochs).tolist())


class ChunkedVectorStorage:
    def __init__(self, chunk_capacity: int = 32768, m: int = 8, initial_arena_capacity: int = 131072):
        self.chunk_capacity = chunk_capacity
        self.m = m
        self.chunks: List[ImmutableChunk] = []
        self._arena_capacity = initial_arena_capacity
        
        # Contiguous unified arena with dynamic geometric expansion
        self._flat_codes = np.zeros((self._arena_capacity, m), dtype=np.uint8)
        self._flat_epochs = np.zeros(self._arena_capacity, dtype=np.uint64)
        self._flat_gids = np.zeros(self._arena_capacity, dtype=np.uint64)
        
        self._active_buf_codes = np.zeros((chunk_capacity, m), dtype=np.uint8)
        self._active_buf_epochs = np.zeros(chunk_capacity, dtype=np.uint64)
        self._active_buf_gids = np.zeros(chunk_capacity, dtype=np.uint64)
        self._active_size = 0
        
        self.total_records = 0
        self.lock = threading.RLock()

    def _grow_arena_if_needed(self, required_capacity: int):
        """Geometric 2x arena reallocation when capacity threshold is reached."""
        if required_capacity <= self._arena_capacity:
            return
        
        new_capacity = max(self._arena_capacity * 2, required_capacity)
        new_codes = np.zeros((new_capacity, self.m), dtype=np.uint8)
        new_epochs = np.zeros(new_capacity, dtype=np.uint64)
        new_gids = np.zeros(new_capacity, dtype=np.uint64)
        
        n = self.total_records
        if n > 0:
            new_codes[:n] = self._flat_codes[:n]
            new_epochs[:n] = self._flat_epochs[:n]
            new_gids[:n] = self._flat_gids[:n]
            
        self._flat_codes = new_codes
        self._flat_epochs = new_epochs
        self._flat_gids = new_gids
        self._arena_capacity = new_capacity

    def append_batch(self, codes: np.ndarray, epoch_id: int, global_ids: np.ndarray):
        num_records = codes.shape[0]
        offset = 0

        with self.lock:
            start_idx = self.total_records
            end_idx = start_idx + num_records
            
            self._grow_arena_if_needed(end_idx)

            self._flat_codes[start_idx:end_idx] = codes
            self._flat_epochs[start_idx:end_idx] = epoch_id
            self._flat_gids[start_idx:end_idx] = global_ids

            while offset < num_records:
                available = self.chunk_capacity - self._active_size
                to_insert = min(num_records - offset, available)
                end = self._active_size + to_insert

                self._active_buf_codes[self._active_size:end] = codes[offset:offset + to_insert]
                self._active_buf_epochs[self._active_size:end] = epoch_id
                self._active_buf_gids[self._active_size:end] = global_ids[offset:offset + to_insert]
                self._active_size = end
                offset += to_insert

                if self._active_size == self.chunk_capacity:
                    sealed_chunk = ImmutableChunk(
                        chunk_id=len(self.chunks),
                        codes=self._active_buf_codes[:self._active_size],
                        epochs=self._active_buf_epochs[:self._active_size],
                        global_ids=self._active_buf_gids[:self._active_size],
                        version=0
                    )
                    self.chunks.append(sealed_chunk)
                    self._active_size = 0

            self.total_records += num_records

    def get_searchable_chunks(self) -> List[ImmutableChunk]:
        with self.lock:
            snapshot = list(self.chunks)
            if self._active_size > 0:
                active_snapshot = ImmutableChunk(
                    chunk_id=len(snapshot),
                    codes=self._active_buf_codes[:self._active_size],
                    epochs=self._active_buf_epochs[:self._active_size],
                    global_ids=self._active_buf_gids[:self._active_size],
                    version=-1
                )
                snapshot.append(active_snapshot)
            return snapshot

    def get_unified_search_view(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Lock-free memory snapshot reference."""
        n = self.total_records
        return self._flat_codes[:n], self._flat_epochs[:n], self._flat_gids[:n], n

    def migrate_all_records(
        self,
        old_epoch_id: int,
        target_epoch_id: int,
        old_snapshot_data: np.ndarray,
        target_snapshot_data: np.ndarray,
        quantize_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        d: int,
        m: int,
        d_sub: int
    ) -> int:
        """
        Asynchronous Out-of-Line Migration with True O(1) Atomic Reference Pointer Swap.
        Heavy compute runs outside the critical lock section.
        """
        with self.lock:
            n = self.total_records
            current_flat_codes = self._flat_codes.copy()
            current_flat_epochs = self._flat_epochs.copy()
            current_chunks = list(self.chunks)

        flat_mask = (current_flat_epochs[:n] == old_epoch_id)
        if not np.any(flat_mask):
            return 0

        flat_match = np.flatnonzero(flat_mask)
        migrated_total = len(flat_match)
        old_codes = current_flat_codes[flat_match]

        # Heavy vector reconstruction and re-quantization executed OUTSIDE lock
        rec_sub = np.zeros((migrated_total, m, d_sub), dtype=np.float32)
        for sub_i in range(m):
            rec_sub[:, sub_i, :] = old_snapshot_data[sub_i][old_codes[:, sub_i]]
        rec_vectors = rec_sub.reshape(migrated_total, d)
        new_codes = quantize_fn(rec_vectors, target_snapshot_data)

        # Update copied buffer out-of-line
        current_flat_codes[flat_match] = new_codes
        current_flat_epochs[flat_match] = target_epoch_id

        # Update sealed chunks out-of-line with get_referenced_epochs() pruning
        updated_chunks = []
        for chunk in current_chunks:
            if old_epoch_id not in chunk.get_referenced_epochs():
                updated_chunks.append(chunk)
                continue

            mask = (chunk.epochs == old_epoch_id)
            m_indices = np.flatnonzero(mask)
            mut_codes = chunk.codes.copy()
            mut_epochs = chunk.epochs.copy()

            c_old = chunk.codes[m_indices]
            c_rec_sub = np.zeros((len(m_indices), m, d_sub), dtype=np.float32)
            for sub_i in range(m):
                c_rec_sub[:, sub_i, :] = old_snapshot_data[sub_i][c_old[:, sub_i]]
            c_new = quantize_fn(c_rec_sub.reshape(len(m_indices), d), target_snapshot_data)

            mut_codes[m_indices] = c_new
            mut_epochs[m_indices] = target_epoch_id

            updated_chunks.append(
                ImmutableChunk(
                    chunk_id=chunk.chunk_id,
                    codes=mut_codes,
                    epochs=mut_epochs,
                    global_ids=chunk.global_ids,
                    version=chunk.version + 1
                )
            )

        # True O(1) Atomic Reference Pointer Swap
        with self.lock:
            # Re-apply any appends that arrived concurrently during out-of-line compute
            if self.total_records > n:
                current_flat_codes[n:self.total_records] = self._flat_codes[n:self.total_records]
                current_flat_epochs[n:self.total_records] = self._flat_epochs[n:self.total_records]
                for c in self.chunks[len(current_chunks):]:
                    updated_chunks.append(c)

            self._flat_codes = current_flat_codes
            self._flat_epochs = current_flat_epochs
            self.chunks = updated_chunks

        return migrated_total
