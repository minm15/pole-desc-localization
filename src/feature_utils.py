import numpy as np

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

def quantize_descmap(descmap: np.ndarray, bits: int = 6):
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