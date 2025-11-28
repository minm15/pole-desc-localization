# ===============================
# ZeroAwareIVF: value+presence gated IVF (drop-in for KMeansIVF)
# ===============================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

desc_dim = 62


# -------- helpers (same semantics as your original) --------
def _to_presence_bits(descs: np.ndarray) -> np.ndarray:
    """
    Return binary presence bits for (N,64) uint8 descriptors:
    1 if dim is non-zero, else 0.
    """
    assert descs.ndim == 2 and descs.shape[1] == desc_dim, "Expect (N,64)"
    return (descs != 0).astype(np.uint8)


def _idf_weights(bits01: np.ndarray, smooth: float = 0.5) -> np.ndarray:
    """
    Compute simple IDF-like weights per dimension.

    IDF(d) = log( (N + 1) / (df_d + smooth) ) , clamped at >= 0.

    Args:
        bits01: (N,64) presence matrix in {0,1}
        smooth: small constant to avoid division by zero

    Returns:
        (64,) float32 array of per-dimension weights
    """
    N = float(bits01.shape[0])
    df = bits01.sum(axis=0).astype(np.float64)
    w = np.log((N + 1.0) / (df + smooth))
    w[w < 0] = 0.0
    return w.astype(np.float32)


def _nonzero_unitvar_scale(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-dimension scale computed from non-zero values:
        scale[d] = 1 / std(X[ X[:,d]!=0, d ])
    If a dimension has <2 non-zero samples, fall back to 1.0.

    Args:
        X: (N,64) float array

    Returns:
        (64,) float32 array of per-dimension scales
    """
    assert X.ndim == 2 and X.shape[1] == desc_dim
    stds = np.zeros(desc_dim, dtype=np.float32)
    for d in range(desc_dim):
        col = X[:, d]
        nz = col[col != 0]
        if nz.size > 1:
            stds[d] = np.std(nz.astype(np.float32))
        else:
            stds[d] = 1.0
    stds = np.where(stds > eps, stds, 1.0)
    return (1.0 / stds).astype(np.float32)


def _kmeans_l2_numpy(X: np.ndarray, k: int, niter: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal L2 k-means fallback (no FAISS):
    - k-means++ initialization
    - Lloyd iterations
    - reseed empty clusters

    Args:
        X: (N,D) float32 matrix
        k: number of centroids
        niter: number of iterations
        seed: RNG seed

    Returns:
        centroids: (k,D) float32
        labels:    (N,)  int32
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    k = min(k, max(1, N))

    # k-means++ init
    centroids = np.empty((k, D), dtype=np.float32)
    i0 = rng.randint(0, N)
    centroids[0] = X[i0]
    dist = np.full(N, np.inf, dtype=np.float64)
    for j in range(1, k):
        d = np.sum((X - centroids[j - 1]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        probs = dist / (dist.sum() + 1e-12)
        idx = rng.choice(N, p=probs)
        centroids[j] = X[idx]
    labels = np.zeros(N, dtype=np.int32)

    for _ in range(niter):
        # assignment
        x2 = (X ** 2).sum(axis=1, keepdims=True)          # (N,1)
        c2 = (centroids ** 2).sum(axis=1, keepdims=True)  # (k,1)
        d2 = x2 + c2.T - 2.0 * (X @ centroids.T)          # (N,k)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for j in range(k):
            idx = np.where(labels == j)[0]
            if idx.size:
                centroids[j] = X[idx].mean(axis=0)
            else:
                centroids[j] = X[rng.randint(0, N)]
    return centroids.astype(np.float32), labels


class ZeroAwareIVF:
    """
    IVF over an augmented (value + presence) space; drop-in replacement for KMeansIVF.

    Augmentation (per dimension d):
        - value feature:    v_d_norm  = v_d * (1/std_nonzero_d)
        - presence feature: z_d       = 1[v_d != 0]
        - per-dim weight:   w_d = IDF(bits)  (optional)
        - concatenated vector:
              [ alpha * w_d * v_d_norm ,  beta * w_d * z_d ]  in R^(2*64=128)

    Intuition:
        - Presence mismatches receive an explicit penalty via beta * w_d.
        - When both sides are non-zero, distance is driven by value difference,
          making 0.2 vs 0.1 much closer than 0.1 vs 0.

    Interface compatibility with KMeansIVF:
        - Attributes: nlist, nprobe, seed, niter
        - Methods: build, candidates_for_query, match_within_lists_equal_nonzero,
                   build_stats, centroids_bits, assigned_list_ids
    """

    def __init__(self, outdir: Optional[Path] = None):
        # Public knobs (compatible names)
        self.nlist: int = 128
        self.nprobe: int = 4
        self.seed: int = 2025
        self.niter: int = 20

        # Additional weights (tunable)
        self.alpha_value: float = 1.0   # weight for value differences
        self.beta_presence: float = 5.0 # penalty for presence mismatch
        self.use_idf: bool = True       # apply per-dimension IDF weights

        self.outdir = Path(outdir) if outdir is not None else None

        # Internal state
        self._map_descs: Optional[np.ndarray] = None           # (N,64) uint8
        self._valid_mask: Optional[np.ndarray] = None          # (N,) bool
        self._orig_idx_kept: Optional[np.ndarray] = None       # (N_kept,)
        self._rows_by_mapid: Dict[int, List[int]] = {}
        self._lists: Dict[int, List[Tuple[int, int]]] = {}     # lid -> [(map_id, row_idx), ...]

        # Transforms / scales
        self._val_scale: Optional[np.ndarray] = None           # (64,) value scaling
        self._idf_w: Optional[np.ndarray] = None               # (64,) IDF weights
        self._aug_centroids: Optional[np.ndarray] = None       # (K,128) centroids
        self._labels_kept: Optional[np.ndarray] = None         # (N_kept,) list ids
        self._build_stats: Dict[str, object] = {}

    # ------------- public API -------------
    def build(self, map_descs: np.ndarray, map_ids: Optional[np.ndarray] = None) -> Dict[str, object]:
        """
        Build IVF index over augmented (value+presence) space.

        Steps:
          1) Filter out all-zero descriptors.
          2) Compute per-dim value scale from non-zero values.
          3) Compute per-dim IDF weights (optional).
          4) Build augmented vectors: concat( alpha*w*v_norm , beta*w*z ).
          5) Train k-means centroids in augmented space (FAISS if available; numpy fallback).
          6) Assign each kept row to its nearest centroid and build inverted lists.

        Args:
            map_descs: (N,64) uint8 descriptor matrix.
            map_ids  : (N,) int64 ids; if None, use row indices.

        Returns:
            dict with basic build statistics.
        """
        assert map_descs.ndim == 2 and map_descs.shape[1] == desc_dim, "map_descs must be (N,64)"
        self._map_descs = map_descs
        N = map_descs.shape[0]

        if map_ids is None:
            map_ids = np.arange(N, dtype=np.int64)
        else:
            map_ids = map_ids.astype(np.int64, copy=False)
            assert map_ids.shape[0] == N

        # Keep non-all-zero rows
        valid = np.any(map_descs != 0, axis=1)
        kept = np.nonzero(valid)[0]
        if kept.size == 0:
            raise ValueError("ZeroAwareIVF.build(): no non-zero descriptors to index.")
        self._valid_mask = valid
        self._orig_idx_kept = kept

        X = map_descs[kept].astype(np.float32)                 # (Nk,64)
        Z = _to_presence_bits(map_descs[kept]).astype(np.float32)  # (Nk,64)

        # Per-dimension scaling from non-zero stats
        self._val_scale = _nonzero_unitvar_scale(X)            # (64,)
        Xn = X * self._val_scale[None, :]

        # IDF weights
        if self.use_idf:
            self._idf_w = _idf_weights(Z.astype(np.uint8))     # (64,)
        else:
            self._idf_w = np.ones(desc_dim, dtype=np.float32)

        Wv = (self.alpha_value * self._idf_w)[None, :]         # (1,64)
        Wz = (self.beta_presence * self._idf_w)[None, :]       # (1,64)

        # Augmented vectors: concat(value, presence)
        Xa = np.concatenate([Wv * Xn, Wz * Z], axis=1).astype(np.float32)  # (Nk,128)

        # Use K = min(nlist, Nk)
        K = min(self.nlist, max(1, Xa.shape[0]))

        # Train in augmented space (prefer FAISS; fallback to numpy)
        try:
            import faiss  # type: ignore
            d = Xa.shape[1]
            clus = faiss.Clustering(d, K)
            clus.seed = self.seed
            clus.niter = self.niter
            index = faiss.IndexFlatL2(d)
            clus.train(Xa, index)
            C = faiss.vector_to_array(clus.centroids).reshape(K, d).astype(np.float32)
            # Assign
            _, I = index.search(Xa, 1)
            labels = I[:, 0].astype(np.int32)
        except Exception:
            C, labels = _kmeans_l2_numpy(Xa, k=K, niter=self.niter, seed=self.seed)

        self._aug_centroids = C
        self._labels_kept = labels

        # Build inverted lists
        self._lists.clear()
        self._rows_by_mapid.clear()
        for j, lid in enumerate(labels):
            orig_row = int(kept[j])
            mid = int(map_ids[orig_row])
            self._lists.setdefault(int(lid), []).append((mid, orig_row))
            self._rows_by_mapid.setdefault(mid, []).append(orig_row)

        # Stats
        list_sizes = {lid: len(v) for lid, v in self._lists.items()}
        num_nonempty = len(self._lists)
        avg_list_size = (sum(list_sizes.values()) / max(1, num_nonempty)) if num_nonempty else 0.0
        largest_list_size = max(list_sizes.values()) if num_nonempty else 0
        largest_list_ids = [lid for lid, sz in list_sizes.items() if sz == largest_list_size]
        self._build_stats = {
            "nlist": int(K),
            "nonempty_lists": int(num_nonempty),
            "avg_list_size": float(avg_list_size),
            "largest_list_size": int(largest_list_size),
            "largest_list_ids": [int(x) for x in largest_list_ids],
        }
        return dict(self._build_stats)

    def candidates_for_query(
        self,
        q_desc: np.ndarray,
        max_cands: Optional[int] = None,
        dedup: bool = True,
        expand_if_empty: bool = True,
    ) -> List[int]:
        """
        Probe top-nprobe augmented-space centroids for a single query,
        then collect candidate map_ids from corresponding inverted lists.

        Args:
            q_desc        : (64,) uint8 query descriptor
            max_cands     : optional cap of returned candidate count
            dedup         : if True, stable-unique map_ids
            expand_if_empty: if True and no probed lists contain entries, scan all lists

        Returns:
            List[int] of candidate map_ids (deduped if requested)
        """
        assert q_desc.ndim == 1 and q_desc.shape[0] == desc_dim and q_desc.dtype == np.uint8, \
            "q_desc must be (64,) uint8"
        assert self._aug_centroids is not None and self._val_scale is not None and self._idf_w is not None

        qv = q_desc.astype(np.float32)
        qz = (qv != 0).astype(np.float32)
        qv *= self._val_scale  # normalize values

        q_aug = np.concatenate([
            (self.alpha_value * self._idf_w) * qv,
            (self.beta_presence * self._idf_w) * qz
        ], axis=0).astype(np.float32)  # (128,)

        C = self._aug_centroids  # (K,128)
        d2 = ((C - q_aug[None, :]) ** 2).sum(axis=1)  # (K,)
        nprobe = max(1, min(self.nprobe, C.shape[0]))
        probe_lids = np.argpartition(d2, nprobe - 1)[:nprobe]
        probe_lids = probe_lids[np.argsort(d2[probe_lids])].tolist()

        pairs: List[Tuple[int, int]] = []
        for lid in probe_lids:
            if lid in self._lists:
                pairs.extend(self._lists[lid])

        if not pairs and expand_if_empty:
            for _, lst in self._lists.items():
                pairs.extend(lst)

        mids = [mid for (mid, _row) in pairs]
        if dedup:
            seen: set[int] = set()
            out: List[int] = []
            for m in mids:
                if m not in seen:
                    seen.add(m)
                    out.append(m)
            mids = out

        if max_cands is not None and len(mids) > max_cands:
            mids = mids[:max_cands]
        return mids

    def match_within_lists_equal_nonzero(
        self,
        q_desc: np.ndarray,
        max_cands: Optional[int] = None,
        expand_if_empty: bool = True,
    ) -> Tuple[int, int]:
        """
        Compatibility helper: pick a single best map_id inside probed lists
        using "equal & non-zero" score (same semantics as your KMeansIVF helper).

        score = sum( (map_desc == q_desc) & (map_desc != 0) )

        Returns:
            (best_map_id, candidate_count); (-1, 0) if none
        """
        assert self._map_descs is not None, "Map descriptors not stored"
        mids = self.candidates_for_query(
            q_desc, max_cands=max_cands, dedup=False, expand_if_empty=expand_if_empty
        )
        if not mids:
            return -1, 0

        cand_rows: List[int] = []
        cand_mids: List[int] = []
        for mid in mids:
            rows = self._rows_by_mapid.get(mid, [])
            if rows:
                cand_rows.append(rows[0])
                cand_mids.append(mid)

        if not cand_rows:
            return -1, 0

        cand_mat = self._map_descs[np.asarray(cand_rows, dtype=np.int64)]
        eq = (cand_mat == q_desc[None, :])
        nonzero = (cand_mat != 0)
        scores = (eq & nonzero).sum(axis=1)
        best_idx = int(np.argmax(scores))
        return int(cand_mids[best_idx]), len(cand_rows)

    # ---- diagnostics / compatibility ----
    def build_stats(self) -> Dict[str, object]:
        """Return build-time statistics."""
        return dict(self._build_stats)

    def centroids_bits(self) -> Optional[np.ndarray]:
        """
        Return a rough presence visualization of centroids for compatibility:
        take the presence half (last 64 dims) and threshold at >0.
        """
        if self._aug_centroids is None:
            return None
        pres = self._aug_centroids[:, desc_dim:]  # (K,64)
        return (pres > 0).astype(np.uint8)

    def assigned_list_ids(self) -> Optional[np.ndarray]:
        """
        Return (N_kept,) int32 centroid labels for kept rows; None if not built.
        """
        return None if self._labels_kept is None else self._labels_kept.copy()