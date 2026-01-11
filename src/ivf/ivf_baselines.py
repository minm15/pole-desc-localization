import numpy as np
import scipy.spatial.distance
import faiss
from typing import Dict, List, Optional, Tuple

class BaselineStandardL2:
    """
    Baseline 1: Standard L2 IVF using FAISS.
    Uses the original descriptors with standard Euclidean distance (K-Means).
    This treats 0 as just a numerical value, ignoring sparsity structure.
    """
    def __init__(self):
        self.nlist = 128
        self.nprobe = 4
        self.seed = 2025
        self.niter = 20
        self.desc_dim = 64
        
        # Internal storage
        self._lists: Dict[int, List[Tuple[int, int]]] = {} 
        self.centroids: Optional[np.ndarray] = None
        self._map_descs: Optional[np.ndarray] = None

    def build(self, map_descs: np.ndarray, map_ids: Optional[np.ndarray] = None) -> Dict[str, object]:
        self._map_descs = map_descs
        N, D = map_descs.shape
        self.desc_dim = D
        if map_ids is None: map_ids = np.arange(N, dtype=np.int64)

        # --- Standard K-Means Training (FAISS) ---
        # 使用 FAISS 直接進行訓練
        kmeans = faiss.Kmeans(D, self.nlist, niter=self.niter, seed=self.seed, verbose=False)
        X_float = map_descs.astype(np.float32)
        kmeans.train(X_float)
        self.centroids = kmeans.centroids

        # --- Assign Buckets ---
        # 使用 FAISS 內建的 index 進行 search (比 scipy cdist 快很多)
        _, labels = kmeans.index.search(X_float, 1)
        labels = labels.flatten()

        # Build Inverted Lists
        self._lists = {i: [] for i in range(self.nlist)}
        for idx, label in enumerate(labels):
            mid = int(map_ids[idx])
            self._lists[label].append((mid, int(idx)))

        # Stats
        sizes = [len(l) for l in self._lists.values()]
        return {
            "nlist": self.nlist,
            "std_dev": float(np.std(sizes)),
            "empty_buckets": int(np.sum(np.array(sizes) == 0))
        }

    def candidates_for_query(self, q_desc: np.ndarray, max_cands=None, dedup=True, expand_if_empty=True) -> List[int]:
        # 1. Find nearest centroids (L2) using Scipy (fast enough for single query)
        # 也可以轉成 faiss search，但在 pure python loop 裡這樣寫 overhead 較小
        dists = scipy.spatial.distance.cdist(q_desc[None, :], self.centroids, metric='euclidean')[0]
        
        # 2. Get top nprobe buckets
        nearest_buckets = np.argsort(dists)[:self.nprobe]
        
        # 3. Collect candidates
        candidates = []
        for bid in nearest_buckets:
            candidates.extend([mid for mid, _ in self._lists[bid]])
            
        if dedup: candidates = list(set(candidates))
        return candidates

    def centroids_bits(self):
        # Visualization helper
        if self.centroids is None: return None
        return (self.centroids != 0).astype(np.uint8)


class BaselineBinaryHamming:
    """
    Baseline 2: Binary Presence + Hamming Distance using FAISS (Approximation).
    Converts descriptors to 0/1 bits.
    Uses FAISS K-Means (L2 on binary data) to cluster, which approximates Hamming clustering.
    """
    def __init__(self):
        self.nlist = 128
        self.nprobe = 4
        self.seed = 2025
        self.niter = 20
        self.desc_dim = 64
        
        self._lists: Dict[int, List[Tuple[int, int]]] = {}
        self.centroids: Optional[np.ndarray] = None 
        self._map_descs: Optional[np.ndarray] = None

    def build(self, map_descs: np.ndarray, map_ids: Optional[np.ndarray] = None) -> Dict[str, object]:
        self._map_descs = map_descs
        N, D = map_descs.shape
        self.desc_dim = D
        if map_ids is None: map_ids = np.arange(N, dtype=np.int64)

        # --- Binarize Data ---
        X_bin = (map_descs != 0).astype(np.float32)

        # --- K-Means on Binary Data (FAISS) ---
        kmeans = faiss.Kmeans(D, self.nlist, niter=self.niter, seed=self.seed, verbose=False)
        kmeans.train(X_bin)
        self.centroids = kmeans.centroids 
        # Note: Centroids become "float means" (e.g. 0.5), representing presence probability

        # --- Assign Buckets ---
        # L2 distance on binary vectors is an effective proxy for Hamming distance clustering
        _, labels = kmeans.index.search(X_bin, 1)
        labels = labels.flatten()

        self._lists = {i: [] for i in range(self.nlist)}
        for idx, label in enumerate(labels):
            mid = int(map_ids[idx])
            self._lists[label].append((mid, int(idx)))

        sizes = [len(l) for l in self._lists.values()]
        return {
            "nlist": self.nlist,
            "std_dev": float(np.std(sizes)),
            "empty_buckets": int(np.sum(np.array(sizes) == 0))
        }

    def candidates_for_query(self, q_desc: np.ndarray, max_cands=None, dedup=True, expand_if_empty=True) -> List[int]:
        # 1. Binarize Query
        q_bin = (q_desc != 0).astype(np.float32).reshape(1, -1)
        
        # 2. Find nearest buckets
        # Using Euclidean distance to float centroids is standard for Soft-Hamming clustering
        dists = scipy.spatial.distance.cdist(q_bin, self.centroids, metric='euclidean')[0]
        
        # 3. Get top nprobe
        nearest_buckets = np.argsort(dists)[:self.nprobe]
        
        candidates = []
        for bid in nearest_buckets:
            candidates.extend([mid for mid, _ in self._lists[bid]])
            
        if dedup: candidates = list(set(candidates))
        return candidates
        
    def centroids_bits(self):
        if self.centroids is None: return None
        # Threshold float centroids back to bits for visualization
        return (self.centroids > 0.5).astype(np.uint8)