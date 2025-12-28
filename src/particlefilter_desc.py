import numpy as np
import scipy
import util
import feature_utils

class particlefilter:
    def __init__(self, count, start, posrange, angrange, 
            polemeans, polevar, descmap, descxy, descmap_index, edges, quant, T_w_o=np.identity(4), d_max = 2.0, descxy_u8=None, geo_qparams=None):
        self.p_min = 0.01
        self.d_max = d_max
        self.minneff = 0.5
        self.estimatetype = 'best'
        self.count = count
        r = np.random.uniform(low=0.0, high=posrange, size=[self.count, 1])
        angle = np.random.uniform(low=-np.pi, high=np.pi, size=[self.count, 1])
        xy = r * np.hstack([np.cos(angle), np.sin(angle)])
        dxyp = np.hstack([xy, np.random.uniform(
            low=-angrange, high=angrange, size=[self.count, 1])])
        self.particles = np.matmul(start, util.xyp2ht(dxyp))
        self.weights = np.full(self.count, 1.0 / self.count)
        self.polemeans = polemeans
        self.descmap = descmap
        self.descxy = descxy
        self.descmap_index = descmap_index
        self.edges = edges
        self.quant = quant
        self.descxy_u8 = descxy_u8
        self.geo_qparams = geo_qparams
        self.poledist = scipy.stats.norm(loc=0.0, scale=np.sqrt(polevar))
        self.kdtree = scipy.spatial.cKDTree(polemeans[:, :2], leafsize=3)
        self.T_w_o = T_w_o
        self.T_o_w = util.invert_ht(self.T_w_o)

    @property
    def neff(self):
        return 1.0 / (np.sum(self.weights**2.0) * self.count)

    def update_motion(self, mean, cov):
        T_r0_r1 = util.xyp2ht(
            np.random.multivariate_normal(mean, cov, self.count))
        self.particles = np.matmul(self.particles, T_r0_r1)

    def update_measurement(self, desc, poleparams, resample=True):
        est_pose = self.estimate_pose()
        matches = self.matcher_quant(desc, est_pose) if self.quant else self.matcher(desc, est_pose)
        # matches = self.matcher_quant(desc) if self.quant else self.matcher(desc)
        M = poleparams.shape[0]
        polepos_r = np.hstack([poleparams[:, :2], np.zeros([M, 1]), np.ones([M, 1])]).T
        
        # update weight according to the matching result
        local_idx  = np.array([i for i, _ in matches], dtype=int)       # shape (M,)
        global_idx = np.array([j for _, j in matches], dtype=int)       # shape (M,)

        # construct the subset of polepos_r, only keep the matched pole
        polepos_r_sub = polepos_r[:, local_idx]                         # shape (4, M)

        # construct the subset of global descxy
        desc_sub = self.descxy[global_idx, :2]                           # shape (M, 2)

        # for each particle, perform matrix operation once, (count, 4, M)
        # represents the pole (4, M) for each particle (count)
        polepos_w_all = np.einsum('pij,jk->pik', self.particles, polepos_r_sub)

        # calclate the distance: (count, M)
        diffs = polepos_w_all[:, :2, :] - desc_sub.T[None, :, :]         # broadcast → (count, 2, M)
        dists = np.linalg.norm(diffs, axis=1)                            # (count, M)
        dists = np.minimum(dists, self.d_max)  
        #print('dist: \n', dists[0])

        weights_factor = np.prod(self.poledist.pdf(dists) + 0.2, axis=1) # shape (count,)

        self.weights *= weights_factor
        
        self.weights /= np.sum(self.weights)
        if resample and self.neff < self.minneff:
            self.resample()
            
    def update_measurement_knn(self, poleparams, resample=True):
        n = poleparams.shape[0]
        polepos_r = np.hstack(
            [poleparams[:, :2], np.zeros([n, 1]), np.ones([n, 1])]).T
        for i in range(self.count):
            polepos_w = self.particles[i].dot(polepos_r)
            d, _ = self.kdtree.query(
                polepos_w[:2].T, k=1, distance_upper_bound=1.0)
            #print('dist: ', d)
            self.weights[i] *= np.prod(
                self.poledist.pdf(np.clip(d, 0.0, 1.0)) + 0.2)
        self.weights /= np.sum(self.weights)

        if resample and self.neff < self.minneff:
            self.resample()

    def estimate_pose(self):
        if self.estimatetype == 'mean':
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights),
                    util.average_angles(xyp[:, 2], weights=self.weights)])
            return self.T_w_o.dot(util.xyp2ht(mean))
        if self.estimatetype == 'max':
            return self.particles[np.argmax(self.weights)]
        if self.estimatetype == 'best':
            i = np.argsort(self.weights)[-int(0.1 * self.count):]
            xyp = util.ht2xyp(np.matmul(self.T_o_w, self.particles[i]))
            mean = np.hstack(
                [np.average(xyp[:, :2], axis=0, weights=self.weights[i]),
                    util.average_angles(xyp[:, 2], weights=self.weights[i])])                
            return self.T_w_o.dot(util.xyp2ht(mean))

    def resample(self):
        cumsum = np.cumsum(self.weights)
        pos = np.random.rand() / self.count
        idx = np.empty(self.count, dtype=np.int)
        ics = 0
        for i in range(self.count):
            while cumsum[ics] < pos:
                ics += 1
            idx[i] = ics
            pos += 1.0 / self.count
        self.particles = self.particles[idx]
        self.weights[:] = 1.0 / self.count
        
    def _score_tol_nonzero(self, q: np.ndarray, C: np.ndarray, tol: float = 0.2) -> np.ndarray:
        """
        Tolerance & non-zero score for non-quantized descriptors:
        score = count of dims where both non-zero and |C - q| <= tol
        """
        nz = (C != 0) & (q[None, :] != 0)
        diffs = np.abs(C - q[None, :])
        return (nz & (diffs <= tol)).sum(axis=1)

    def _score_equal_nonzero(self, q: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Equal & non-zero score for quantized descriptors:
        score = count of dims where both non-zero and C == q
        """
        nz = (C != 0)
        eq = (C == q[None, :])
        return (eq & nz).sum(axis=1)
    
    # def matcher(self, local_descs: np.ndarray):
    #     """
    #     IVF-only matcher for non-quantized descriptors.
    #     For each local row, retrieve ALL candidates from the probed lists (no cap),
    #     score with tolerance & non-zero, pick the best row index.
    #     Returns: List[(i_local, j_global_row)]
    #     """
    #     matches: list[tuple[int, int]] = []
    #     ivf = self.descmap_index  # KMeansIVF object
    #     for i, d1 in enumerate(local_descs):
    #         # candidates_for_query with max_cands=None -> no cap; dedup=True is fine since map_id == row index
    #         cand_rows = ivf.candidates_for_query(d1.astype(np.uint8), max_cands=None, dedup=True, expand_if_empty=True)
    #         if not cand_rows:
    #             matches.append((i, -1))
    #             continue
    #         C = self.descmap[cand_rows]  # (K,64)
    #         scores = self._score_tol_nonzero(d1, C, tol=0.2)
    #         best_k = int(np.argmax(scores))
    #         matches.append((i, int(cand_rows[best_k])))
    #     return matches
    
    def matcher(self, local_descs: np.ndarray, current_pose_w: np.ndarray, search_radius: float = 20.0):
        matches: list[tuple[int, int]] = []
        ivf = self.descmap_index
        
        rx, ry = current_pose_w[0, 3], current_pose_w[1, 3]
        
        for i, d1 in enumerate(local_descs):
            cand_rows = ivf.candidates_for_query(d1.astype(np.uint8), max_cands=None, dedup=True, expand_if_empty=True)
            if not cand_rows:
                matches.append((i, -1))
                continue

            cand_indices = np.array(cand_rows, dtype=int)
            cand_pos = self.descxy[cand_indices, :2] # (N_cand, 2)
            
            dist_sq = (cand_pos[:, 0] - rx)**2 + (cand_pos[:, 1] - ry)**2
            valid_mask = dist_sq < (search_radius ** 2)
            
            valid_cands = cand_indices[valid_mask]
            
            if len(valid_cands) == 0:
                matches.append((i, -1))
                continue
                
            C = self.descmap[valid_cands]
            scores = self._score_tol_nonzero(d1, C, tol=0.2)
            best_idx_in_subset = int(np.argmax(scores))

            matches.append((i, int(valid_cands[best_idx_in_subset])))
            
        return matches


    def matcher_quant(self, local_descs: np.ndarray, current_pose_w: np.ndarray, search_radius: float = 20.0):
        matches: list[tuple[int, int]] = []
        ivf = self.descmap_index

        min_x, min_y, R = self.geo_qparams
        rx, ry = current_pose_w[0, 3], current_pose_w[1, 3]
        qrx, qry = feature_utils.quantize_xy_to_u6_shared(rx, ry, min_x, min_y, R)

        for i, d1 in enumerate(local_descs):
            cand_rows = ivf.candidates_for_query(d1.astype(np.uint8), max_cands=None, dedup=True, expand_if_empty=True)
            if not cand_rows:
                matches.append((i, -1))
                continue

            cand_indices = np.array(cand_rows, dtype=np.int32)

            cand_xy = self.descxy_u8[cand_indices]  # (Ncand,2) uint8
            mx = cand_xy[:, 0].astype(np.int32)
            my = cand_xy[:, 1].astype(np.int32)
            qx = int(qrx); qy = int(qry)
            geo_mask = (mx >= qx - 2) & (mx <= qx + 2) & (my >= qy - 2) & (my <= qy + 2)
            valid_cands = cand_indices[geo_mask]

            if len(valid_cands) == 0:
                matches.append((i, -1))
                continue

            # --- scoring ---
            C = self.descmap[valid_cands]  # (K,64)
            scores = self._score_equal_nonzero(d1, C)
            best_k = int(np.argmax(scores))
            matches.append((i, int(valid_cands[best_k])))

        return matches