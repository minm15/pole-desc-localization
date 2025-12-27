# kmeans_ivf.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

DESC_DIM = 62

# ===============================
# Binary / presence helpers
# ===============================
def to_presence_bits(descs: np.ndarray) -> np.ndarray:
    """
    Convert (N,64) uint8 descriptors to presence bits in {0,1}:
    1 means the dimension is non-zero (valid), 0 means absent.
    """
    assert descs.ndim == 2 and descs.shape[1] == DESC_DIM, "Expect (N,64) uint8"
    return (descs != 0).astype(np.uint8)


def packbits_be(bits01: np.ndarray) -> np.ndarray:
    """
    Pack (N,64) bits (0/1) into (N,8) bytes using big-endian bit order.
    Suitable for faiss.IndexBinary* (d_bits=64).
    """
    assert bits01.ndim == 2 and bits01.shape[1] == DESC_DIM, "Expect (N,64) bits"
    return np.packbits(bits01, axis=1, bitorder="big")


# Precomputed popcount for bytes 0..255
_POPCOUNT_8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_pairwise_packed(A8: np.ndarray, B8: np.ndarray, block: int = 4096) -> np.ndarray:
    """
    Brute-force pairwise Hamming distance between packed arrays in blocks to avoid N^2 memory spikes.
    A8: (na,8) uint8, B8: (nb,8) uint8 -> returns (na, nb) int32.
    Note: FAISS path is preferred; this helper is here to preserve a memory-friendly fallback style.
    """
    assert A8.ndim == 2 and B8.ndim == 2 and A8.shape[1] == 8 and B8.shape[1] == 8
    na, nb = A8.shape[0], B8.shape[0]
    out = np.empty((na, nb), dtype=np.int32)

    for s in range(0, na, block):
        e = min(s + block, na)
        xor = A8[s:e, None, :] ^ B8[None, :, :]
        # popcount per byte, sum across 8 bytes
        pc = _POPCOUNT_8[xor].sum(axis=2).astype(np.int32)
        out[s:e, :] = pc
    return out


def assign_to_centroids_hamming(
    P8: np.ndarray, C8: np.ndarray, topk: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign packed presence vectors (P8) to nearest binary centroids (C8) by Hamming,
    using FAISS IndexBinaryFlat(64). No NumPy fallback.

    Returns:
        D: (N, topk) Hamming distances (int32)
        I: (N, topk) centroid indices (int32)
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS is required for assign_to_centroids_hamming(...). "
            "Please install 'faiss-cpu' (or 'faiss-gpu'), e.g.:\n"
            "  pip install faiss-cpu"
        ) from e

    assert P8.ndim == 2 and P8.shape[1] == 8 and P8.dtype == np.uint8, "Expect (N,8) uint8"
    assert C8.ndim == 2 and C8.shape[1] == 8 and C8.dtype == np.uint8, "Expect (M,8) uint8"
    topk = max(1, min(topk, C8.shape[0]))

    index = faiss.IndexBinaryFlat(64)
    index.add(C8)
    D, I = index.search(P8, topk)
    return D.astype(np.int32), I.astype(np.int32)


def train_binary_kmeans_l2_round(
    P_bits: np.ndarray,
    nlist: int,
    seed: int = 2025,
    niter: int = 15,
    verbose: bool = True,
) -> np.ndarray:
    """
    Train 'nlist' centroids in the {0,1} presence space by FAISS L2 k-means,
    then round each dimension at 0.5 to obtain binary centroids.

    Returns:
        C_bits: (nlist, 64) uint8 in {0,1}

    Raises:
        RuntimeError if FAISS is not installed.
    """
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS is required for train_binary_kmeans_l2_round(...). "
            "Please install 'faiss-cpu' (or 'faiss-gpu'), e.g.:\n"
            "  pip install faiss-cpu"
        ) from e

    assert P_bits.ndim == 2 and P_bits.shape[1] == DESC_DIM, "Expect presence bits (N,64)"
    X = P_bits.astype(np.float32, copy=False)
    d = X.shape[1]

    clus = faiss.Clustering(d, nlist)
    clus.seed = seed
    clus.niter = niter
    index = faiss.IndexFlatL2(d)
    clus.train(X, index)

    Cf = faiss.vector_to_array(clus.centroids).reshape(nlist, d)
    C = (Cf >= 0.5).astype(np.uint8)
    if verbose:
        print(f"[KMeansIVF] FAISS k-means done (niter={niter}); rounded @0.5")
    return C


# ===============================
# Core IVF class
# ===============================
class KMeansIVF:
    """
    Presence-only (Hamming-routed) IVF for (N,64) uint8 descriptors.
    - Centroids are trained by L2 k-means in presence space and rounded to {0,1}.
    - Assignment/probing use Hamming in packed presence space (FAISS-only).
    - Inverted lists store (map_id, orig_row_idx) for each centroid.

    Parameters are intentionally hard-coded in __init__ for simplicity,
    but you can override instance attributes after construction if needed.
    """

    def __init__(self, outdir: Optional[Path] = None):
        # ---- Hard-coded IVF parameters (you can override after init if desired) ----
        self.nlist: int = 128     # number of centroids/lists
        self.nprobe: int = 4      # lists to probe at query
        self.seed: int = 2025      # RNG seed for k-means
        self.niter: int = 15       # k-means iterations

        # Output/diagnostic directory (optional)
        self.outdir = Path(outdir) if outdir is not None else None

        # ---- Internal state ----
        self._centroids_bits: Optional[np.ndarray] = None       # (nlist,64) uint8 in {0,1}
        self._centroids_packed: Optional[np.ndarray] = None     # (nlist,8)  uint8
        self._lists: Dict[int, List[Tuple[int, int]]] = {}      # lid -> [(map_id, orig_row_idx), ...]
        self._rows_by_mapid: Dict[int, List[int]] = {}          # map_id -> [orig_row_idx, ...]
        self._map_descs: Optional[np.ndarray] = None            # original map descs (N,64) uint8
        self._valid_mask: Optional[np.ndarray] = None           # (N,) bool for non-zero descs
        self._orig_idx_kept: Optional[np.ndarray] = None        # indices of kept rows
        self._build_stats: Dict[str, object] = {}

    # ---------- Public API ----------
    def build(self, map_descs: np.ndarray, map_ids: Optional[np.ndarray] = None) -> Dict[str, object]:
        """
        Build IVF from map descriptors.
        - Filters out all-zero descriptors (no information in presence space).
        - Trains binary centroids in presence space.
        - Assigns each kept row to its nearest centroid by Hamming and builds inverted lists.

        Args:
            map_descs: (N,64) uint8 descriptor matrix.
            map_ids  : (N,) int64 ids mapping to your original map entities. If None, use row indices.

        Returns:
            build_stats: dict with nlist, nonempty_lists, avg_list_size, largest_list_size, largest_list_ids.

        Raises:
            ValueError if there are no valid (non-zero) descriptors.
        """
        print(map_descs.shape)
        assert map_descs.ndim == 2 and map_descs.shape[1] == DESC_DIM, \
            "map_descs must be (N,64) uint8"

        self._map_descs = map_descs
        N = map_descs.shape[0]
        if map_ids is None:
            map_ids = np.arange(N, dtype=np.int64)
        else:
            map_ids = map_ids.astype(np.int64, copy=False)
            assert map_ids.shape[0] == N

        # 1) Filter all-zero descriptors (no presence)
        valid = np.any(map_descs != 0, axis=1)
        kept_idx = np.nonzero(valid)[0]
        if kept_idx.size == 0:
            raise ValueError("KMeansIVF.build(): No valid (non-zero) descriptors to index.")

        self._valid_mask = valid
        self._orig_idx_kept = kept_idx

        # 2) Train centroids in presence space and round to {0,1} (FAISS-only)
        P_bits = to_presence_bits(map_descs[kept_idx])
        C_bits = train_binary_kmeans_l2_round(
            P_bits, nlist=self.nlist, seed=self.seed, niter=self.niter, verbose=True
        )
        self._centroids_bits = C_bits.astype(np.uint8, copy=False)
        self._centroids_packed = packbits_be(self._centroids_bits)

        # 3) Assign kept rows to nearest centroid by Hamming (presence-packed; FAISS-only)
        P8 = packbits_be(P_bits)
        _, I = assign_to_centroids_hamming(P8, self._centroids_packed, topk=1)
        labels = I[:, 0].astype(np.int64)

        # 4) Build inverted lists and map_id -> rows index
        self._lists.clear()
        self._rows_by_mapid.clear()
        for j, lid in enumerate(labels):
            orig_row = int(kept_idx[j])
            mid = int(map_ids[orig_row])
            self._lists.setdefault(int(lid), []).append((mid, orig_row))
            self._rows_by_mapid.setdefault(mid, []).append(orig_row)

        # 5) Stats
        list_sizes = {lid: len(v) for lid, v in self._lists.items()}
        num_nonempty = len(self._lists)
        avg_list_size = (sum(list_sizes.values()) / max(1, num_nonempty)) if num_nonempty else 0.0
        largest_list_size = max(list_sizes.values()) if num_nonempty else 0
        largest_list_ids = [lid for lid, sz in list_sizes.items() if sz == largest_list_size]

        self._build_stats = {
            "nlist": int(self.nlist),
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
        Return candidate map_ids for a single query descriptor.
        - Probe top-nprobe lists by Hamming in presence space (FAISS).
        - Aggregate (map_id, row_idx) pairs from those lists.
        - Optionally de-duplicate and cap candidate count.

        Args:
            q_desc         : (64,) uint8 query descriptor.
            max_cands      : optional cap for number of returned candidates.
            dedup          : if True, unique by map_id while preserving order.
            expand_if_empty: if True and no list is selected (rare), fall back to scan all lists.

        Returns:
            List[int] of candidate map_ids.
        """
        assert q_desc.ndim == 1 and q_desc.shape[0] == DESC_DIM and q_desc.dtype == np.uint8, \
            "q_desc must be (64,) uint8"
        assert self._centroids_packed is not None and self._lists is not None, "Index not built"

        Q_bits = to_presence_bits(q_desc[None, :])
        Q8 = packbits_be(Q_bits)  # (1,8)

        # Probe top-nprobe lists (cap by number of centroids)
        topk = max(1, min(self.nprobe, self._centroids_packed.shape[0]))
        _, Iq = assign_to_centroids_hamming(Q8, self._centroids_packed, topk=topk)
        probe_lids = [int(x) for x in Iq[0]]

        # Aggregate candidates from probed lists
        pairs: List[Tuple[int, int]] = []
        for lid in probe_lids:
            if lid in self._lists:
                pairs.extend(self._lists[lid])

        # Optional fallback: if empty, scan all lists
        if not pairs and expand_if_empty:
            for _lid, lst in self._lists.items():
                pairs.extend(lst)

        # Extract map_ids; optional stable dedup
        mids = [mid for (mid, _row) in pairs]
        if dedup:
            seen = set()
            mids_dedup: List[int] = []
            for m in mids:
                if m not in seen:
                    seen.add(m)
                    mids_dedup.append(m)
            mids = mids_dedup

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
        Optional convenience: select a single best map_id inside probed lists
        using the "equal & non-zero" score:
            score = sum( (map_desc == q_desc) & (map_desc != 0) )

        Returns:
            (best_map_id, candidate_count); (-1, 0) if none.
        """
        assert self._map_descs is not None, "Map descriptors not stored"

        # Get candidate map_ids (preserving duplicates to keep ordering preference)
        mids = self.candidates_for_query(q_desc, max_cands=max_cands, dedup=False, expand_if_empty=expand_if_empty)
        if not mids:
            return -1, 0

        # Rebuild candidate row list via map_id -> rows; pick the first row per map_id
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
        best_mid = int(cand_mids[best_idx])
        return best_mid, len(cand_rows)

    # ---------- Optional: diagnostics / artifacts ----------
    def build_stats(self) -> Dict[str, object]:
        """Return build-time statistics."""
        return dict(self._build_stats)

    def centroids_bits(self) -> Optional[np.ndarray]:
        """Return (nlist,64) uint8 centroids in {0,1}, or None if not built."""
        return None if self._centroids_bits is None else self._centroids_bits.copy()

    def assigned_list_ids(self) -> Optional[np.ndarray]:
        """
        Return (N_kept,) int32 labels assigning each kept map row to a centroid.
        None if not built or if there are no valid rows.
        """
        if self._centroids_packed is None or self._orig_idx_kept is None or self._map_descs is None:
            return None
        P_bits = to_presence_bits(self._map_descs[self._orig_idx_kept])
        P8 = packbits_be(P_bits)
        _, I = assign_to_centroids_hamming(P8, self._centroids_packed, topk=1)
        return I[:, 0].astype(np.int32)