import numpy as np
from sklearn.cluster import KMeans

def score_based_feature_match(new_pole, new_desc, global_poles, global_descs,
                              pos_thresh, score_thresh, score_tol):
    """
    Checks if a new feature exists in the global map using a position-first,
    score-second logic.
    Returns: int (index in global map) or -1 (no match)
    """
    if global_poles.shape[0] == 0:
        return -1

    # 1. Filter by Position FIRST
    pos_dists = np.linalg.norm(global_poles[:, :2] - new_pole[:2], axis=1)
    position_candidate_indices = np.where(pos_dists < pos_thresh)[0]

    if position_candidate_indices.size == 0:
        return -1

    # 2. Filter by Score SECOND
    nearby_descs = global_descs[position_candidate_indices]
    non_zero_mask = (nearby_descs != 0) & (new_desc != 0)
    tolerance_mask = np.abs(nearby_descs - new_desc) <= score_tol
    
    scores = np.sum(non_zero_mask & tolerance_mask, axis=1)

    # 3. Decision
    passing_scores_mask = (scores >= score_thresh)
    
    if np.any(passing_scores_mask):
        passing_indices_local = np.where(passing_scores_mask)[0]
        # Return the best match among candidates
        best_local_idx = passing_indices_local[np.argmax(scores[passing_scores_mask])]
        best_global_idx = position_candidate_indices[best_local_idx]
        return best_global_idx
    else:
        return -1

def merge_cluster_descriptors(poleparams, descs_array, threshold=0.2):
    """
    Merge logic for descriptors within a cluster.
    Currently returns input directly (can be expanded).
    """
    keep = np.any(descs_array != 0, axis=1)  
    return poleparams[keep], descs_array[keep]

def build_descmap_index(descmap, k):
    """
    Build inverted index based on descriptor dimension 0.
    """
    dim0 = descmap[:, 0]
    edges = np.quantile(dim0, np.linspace(0, 1, k+1))
    descmap_index = {i: [] for i in range(k)}
    
    for j, v in enumerate(dim0):
        bin_idx = np.searchsorted(edges, v, side='right') - 1
        bin_idx = max(0, min(bin_idx, k-1))
        descmap_index[bin_idx].append(j)
    return descmap_index, edges

def quantize_descmap(descmap: np.ndarray, bits: int = 4):
    """
    Quantize descriptor map into buckets.
    """
    if bits < 1: raise ValueError("bits must be >= 1")
    num_buckets = 1 << bits
    flat = descmap.ravel()
    nonzero_mask = (flat != 0.0)
    nonzeros = flat[nonzero_mask]

    if nonzeros.size == 0:
        return np.zeros_like(descmap, dtype=int), np.array([])

    quantiles = np.linspace(0.0, 1.0, num_buckets)[1:-1]
    thresholds = np.quantile(nonzeros, quantiles)
    idx = np.digitize(nonzeros, thresholds, right=False)
    qvals = idx + 1
    qflat = np.zeros_like(flat, dtype=int)
    qflat[nonzero_mask] = qvals

    return qflat.reshape(descmap.shape), thresholds

def quantize_descmap_uniform(descmap: np.ndarray, bits: int = 6):
    """
    Quantize descriptor map into buckets using uniform intervals.
    """
    if bits < 1: raise ValueError("bits must be >= 1")
    num_buckets = 1 << bits
    flat = descmap.ravel()
    nonzero_mask = (flat != 0.0)
    nonzeros = flat[nonzero_mask]

    if nonzeros.size == 0:
        return np.zeros_like(descmap, dtype=int), np.array([])

    min_val = nonzeros.min()
    max_val = nonzeros.max()

    edges = np.linspace(min_val, max_val, num_buckets + 1)
    thresholds = edges[1:-1]

    idx = np.digitize(nonzeros, thresholds, right=False)

    qvals = idx + 1
    
    qflat = np.zeros_like(flat, dtype=int)
    qflat[nonzero_mask] = qvals

    return qflat.reshape(descmap.shape), thresholds

def quantize_descmap_zscore(descmap: np.ndarray, bits: int = 4, k: float = 3.0):
    """
    Return interface unchanged: (qmap, thresholds)

    thresholds: length (2^bits - 2) (e.g. 14 when bits=4),
    suitable for quantize_descriptor(vec, thresholds) which does:
        idx = digitize(vec[nz], thresholds); q = idx + 1

    Behavior:
      - zeros stay 0 in qmap
      - positives are quantized into 1..(2^bits-1)
      - tail-high (>U) naturally maps to the highest bin
      - tail-low (<=L) naturally maps to the lowest nonzero bin (1)
        (cannot map positives to bin0 without changing quantize_descriptor)
    """
    if bits < 2:
        raise ValueError("bits must be >= 2")
    B = 1 << bits
    n_thr = B - 2  # e.g. 14

    flat = descmap.ravel()
    qflat = np.zeros_like(flat, dtype=int)

    pos_mask = (flat > 0.0)
    x = flat[pos_mask]
    if x.size == 0:
        return qflat.reshape(descmap.shape), np.array([])

    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0.0:
        qflat[pos_mask] = 1
        return qflat.reshape(descmap.shape), np.array([])

    L = mu - k * sigma
    U = mu + k * sigma

    # middle range used to compute quantile thresholds
    xin = x[(x > L) & (x <= U)]
    if xin.size == 0:
        # fallback: quantile over all positives
        qs = np.linspace(0.0, 1.0, B)[1:-1]  # length B-2
        thresholds = np.quantile(x, qs).astype(float)
    else:
        qs = np.linspace(0.0, 1.0, B)[1:-1]  # length B-2 (14)
        thresholds = np.quantile(xin, qs).astype(float)

    # make thresholds strictly increasing to reduce empty bins when ties exist
    eps = np.finfo(float).eps * max(1.0, float(U) if np.isfinite(U) else 1.0)
    thresholds = np.maximum.accumulate(thresholds)
    for i in range(1, thresholds.size):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + eps

    # quantize positives using these thresholds (same rule as quantize_descriptor)
    idx = np.digitize(x, thresholds, right=False)  # 0..(B-2)
    qflat[pos_mask] = idx + 1                      # 1..(B-1)

    return qflat.reshape(descmap.shape), thresholds

# def quantize_descmap_kmeans(descmap: np.ndarray, bits: int = 4, clip_percentile: float = 0.95):
#     """
#     Hybrid Strategy: Clipped Log + K-Means.
    
#     Why this works:
#       1. Dedicate the last bin (Bin 15) exclusively to the top (1-clip)% outliers.
#          This prevents the 'long tail' from stealing centroids from the dense peak.
#       2. Apply Log + K-Means ONLY on the remaining 99% of data.
#          This forces high resolution (14 bins) on the dense region where accuracy matters most.
    
#     Args:
#         clip_percentile: e.g., 0.99 means top 1% data goes to the last bin.
#     """
#     if bits < 2:
#         raise ValueError("bits must be >= 2")
    
#     B = 1 << bits
#     n_bins_total = B - 1        # 15 bins (1..15)
    
#     flat = descmap.ravel()
#     qflat = np.zeros_like(flat, dtype=int)
    
#     # Filter positives
#     pos_mask = (flat > 0.0)
#     x = flat[pos_mask]
    
#     if x.size == 0:
#         return qflat.reshape(descmap.shape), np.array([])
    
#     # --- Step 1: Handle the Tail (Clipping) ---
#     # Find the boundary for the top 1% (or user defined %)
#     # This value becomes the "Gatekeeper" to the last bin.
#     tail_threshold = np.quantile(x, clip_percentile)
    
#     # Split data: Body (99%) vs Tail (1%)
#     x_body = x[x <= tail_threshold]
    
#     # Check if we have enough unique data in body to do clustering
#     unique_body = np.unique(x_body)
    
#     # We reserve 1 bin for the tail, so we have (n_bins_total - 1) for the body
#     n_bins_body = n_bins_total - 1  # e.g., 14 bins
    
#     final_thresholds = []

#     if unique_body.size <= n_bins_body:
#         # Fallback if body is too sparse: just use Quantile or unique values
#         qs = np.linspace(0, 1, n_bins_body + 1)[1:]
#         body_thresholds = np.quantile(x_body, qs)
#     else:
#         # --- Step 2: Log + K-Means on the Body ---
#         # Log transform focuses resolution on small values
#         x_body_log = np.log(x_body)
#         X_train = x_body_log.reshape(-1, 1)
        
#         # Run K-Means only on the body
#         kmeans = KMeans(n_clusters=n_bins_body, n_init=10, random_state=42)
#         kmeans.fit(X_train)
        
#         # Get centroids in Log domain
#         centroids = np.sort(kmeans.cluster_centers_.flatten())
        
#         # Calculate boundaries (midpoints) in Log domain
#         # These separate bins 1..14
#         log_bounds = (centroids[:-1] + centroids[1:]) / 2.0
#         body_thresholds = np.exp(log_bounds)
    
#     # --- Step 3: Combine Thresholds ---
#     # The thresholds are: [Body Boundaries] + [Tail Threshold]
#     # Body boundaries split bins 1 to 14.
#     # Tail threshold splits bin 14 from bin 15.
    
#     thresholds = np.concatenate([body_thresholds, [tail_threshold]])
    
#     # --- Safety: Ensure strictly increasing ---
#     thresholds = np.maximum.accumulate(thresholds)
#     eps = np.finfo(float).eps * max(1.0, float(x.max()))
#     for i in range(1, thresholds.size):
#         if thresholds[i] <= thresholds[i - 1]:
#             thresholds[i] = thresholds[i - 1] + eps

#     # --- Quantize ---
#     # digitize returns 0..14. +1 makes it 1..15
#     idxs = np.digitize(x, thresholds, right=False)
#     qflat[pos_mask] = idxs + 1
    
#     return qflat.reshape(descmap.shape), thresholds


def quantize_descriptor(vec: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Quantize a single descriptor vector.
    """
    q = np.zeros_like(vec, dtype=int)
    nz = (vec != 0.0)
    if thresholds.size > 0:
        idxs = np.digitize(vec[nz], thresholds, right=False)
        q[nz] = idxs + 1
    return q

def quantize_xy_to_u6_shared(x, y, min_x, min_y, R, MAX=90):
    QMAX = MAX
    if R <= 0.0:
        return np.uint8(0), np.uint8(0)

    qx = int(np.rint((float(x) - float(min_x)) * QMAX / float(R)))
    qy = int(np.rint((float(y) - float(min_y)) * QMAX / float(R)))

    qx = 0 if qx < 0 else (MAX if qx > MAX else qx)
    qy = 0 if qy < 0 else (MAX if qy > MAX else qy)
    return np.uint8(qx), np.uint8(qy)

def quantize_xy_array_to_u6_shared(xy, min_x, min_y, R, MAX=90):
    if R <= 0.0:
        return np.zeros((xy.shape[0], 2), dtype=np.uint8)

    qx = np.rint((xy[:, 0].astype(np.float64) - float(min_x)) * MAX / float(R)).astype(np.int32)
    qy = np.rint((xy[:, 1].astype(np.float64) - float(min_y)) * MAX / float(R)).astype(np.int32)

    qx = np.clip(qx, 0, MAX).astype(np.uint8)
    qy = np.clip(qy, 0, MAX).astype(np.uint8)
    return np.stack([qx, qy], axis=1)