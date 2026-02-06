import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance

class IVFProfiler:
    def __init__(self, ivf_object, save_dir, nprobe=None):
        """
        Initializes the IVF Profiler.
        
        Args:
            ivf_object: Your IVF instance (ZeroAwareIVF or KMeansIVF)
            save_dir: Directory to save analysis results
            nprobe: IVF probe count (defaults to 1 if not found)
        """
        self.ivf = ivf_object
        self.save_dir = save_dir
        self.slow_frames_log = []
        
        # Attempt to get nprobe
        self.nprobe = getattr(self.ivf, 'nprobe', nprobe)
        if self.nprobe is None:
            self.nprobe = 1
            print("[IVFProfiler] Warning: nprobe not found in object, defaulting to 1.")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def analyze_bucket_distribution(self):
        """
        Stats: Counts descriptors per bucket (centroid).
        Saves raw bucket sizes to .npy for external comparison.
        """
        print("[IVFProfiler] Analyzing bucket distribution...")
        
        bucket_sizes = []
        nlist = getattr(self.ivf, 'nlist', 0)

        # --- Case 1: ZeroAwareIVF ---
        if hasattr(self.ivf, '_lists') and isinstance(self.ivf._lists, dict):
            bucket_sizes = [0] * nlist
            for lid, items in self.ivf._lists.items():
                if 0 <= lid < nlist:
                    bucket_sizes[lid] = len(items)
        
        # --- Case 2: Generic list of lists ---
        elif hasattr(self.ivf, 'invlists') and isinstance(self.ivf.invlists, (list, np.ndarray)):
            bucket_sizes = [len(l) for l in self.ivf.invlists]
        
        # --- Case 3: FAISS Wrapped ---
        elif hasattr(self.ivf, 'index') and hasattr(self.ivf.index, 'invlists'):
            nlist = self.ivf.index.nlist
            bucket_sizes = [self.ivf.index.invlists.list_size(i) for i in range(nlist)]
        
        else:
            print("[IVFProfiler] Error: Cannot access internal lists.")
            return

        bucket_sizes = np.array(bucket_sizes)
        if len(bucket_sizes) == 0:
            print("[IVFProfiler] Warning: nlist is 0 or buckets are empty.")
            return

        stats = {
            "mean_size": float(np.mean(bucket_sizes)),
            "std_dev": float(np.std(bucket_sizes)),
            "max_size": int(np.max(bucket_sizes)),
            "empty_buckets": int(np.sum(bucket_sizes == 0))
        }
        print(f"[IVFProfiler] Stats: {json.dumps(stats, indent=2)}")

        n_buckets = len(bucket_sizes)
        plt.figure(figsize=(10, 6))
        plt.bar(range(n_buckets), bucket_sizes, width=1.0, color='skyblue', edgecolor='black', linewidth=0.5)
        plt.title(f"IVF Bucket Size Distribution (Total: {np.sum(bucket_sizes)})")
        plt.axhline(y=stats['mean_size'], color='r', linestyle='--', label=f"Mean: {stats['mean_size']:.1f}")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "ivf_bucket_histogram.png"))
        plt.close()

        raw_data_path = os.path.join(self.save_dir, "bucket_sizes.npy")
        np.save(raw_data_path, bucket_sizes)
        print(f"[IVFProfiler] Raw bucket data saved to: {raw_data_path}")

    def monitor_frame(self, t_elapsed_sec, query_descs, frame_idx, t_now):
        """
        Monitors frame time. If > 200ms, logs detailed query info.
        For ZeroAwareIVF, we need to replicate the 'Augmented' query mapping.
        """
        THRESHOLD_MS = 200.0
        t_ms = t_elapsed_sec * 1000.0
        
        if t_ms > THRESHOLD_MS:
            print(f"[IVFProfiler] Slow frame detected! Frame {frame_idx}, Time: {t_ms:.2f}ms. Logging details...")
            
            # Ensure we have bucket sizes
            if not hasattr(self, 'cached_bucket_sizes'):
                # Try to run analysis to populate cache, or fail gracefully
                print("[IVFProfiler] Warning: bucket stats not cached, skipping detailed breakdown.")
                return
            
            bucket_sizes = self.cached_bucket_sizes

            # --- ZeroAwareIVF Specific Logic ---
            # We need to map query descs to centroids using the AUGMENTED space logic
            if hasattr(self.ivf, '_aug_centroids') and hasattr(self.ivf, '_val_scale'):
                # 1. Prepare Query in Augmented Space
                # Logic copied from ZeroAwareIVF.candidates_for_query
                qv = query_descs.astype(np.float32)
                qz = (qv != 0).astype(np.float32)
                # Normalize values
                qv = qv * self.ivf._val_scale[None, :] 
                
                # Weights
                alpha = self.ivf.alpha_value
                beta = self.ivf.beta_presence
                idf = self.ivf._idf_w
                
                # Stack augmented vector: (N_query, 128)
                # left part: alpha * idf * v_norm
                left = (alpha * idf)[None, :] * qv
                # right part: beta * idf * z
                right = (beta * idf)[None, :] * qz
                
                q_aug = np.concatenate([left, right], axis=1).astype(np.float32)
                
                centroids = self.ivf._aug_centroids # (K, 128)
            
            # --- Generic/KMeansIVF Logic (Standard L2) ---
            elif hasattr(self.ivf, 'centroids'):
                 q_aug = query_descs.astype(np.float32)
                 centroids = self.ivf.centroids
            else:
                 print("[IVFProfiler] Cannot find centroids for distance calculation.")
                 return

            # Compute Distances (cdist)
            # q_aug: (N, D), centroids: (K, D)
            dists = scipy.spatial.distance.cdist(q_aug, centroids, metric='euclidean')

            frame_log = {
                "frame_idx": int(frame_idx),
                "timestamp": float(t_now),
                "duration_ms": t_ms,
                "num_queries": len(query_descs),
                "queries": []
            }
            
            # Log top buckets for each query
            for q_idx in range(len(query_descs)):
                nearest_indices = np.argsort(dists[q_idx])[:self.nprobe]
                
                probed_buckets_info = []
                total_candidates = 0
                for bucket_id in nearest_indices:
                    size = bucket_sizes[bucket_id] if bucket_id < len(bucket_sizes) else 0
                    probed_buckets_info.append({
                        "bucket_id": int(bucket_id),
                        "size": int(size)
                    })
                    total_candidates += size
                
                frame_log["queries"].append({
                    "query_idx": q_idx,
                    "total_candidates_in_buckets": int(total_candidates),
                    "probed_buckets": probed_buckets_info
                })
            
            self.slow_frames_log.append(frame_log)
            self.save_log()

    def save_log(self):
        out_path = os.path.join(self.save_dir, "ivf_slow_query_analysis.json")
        with open(out_path, 'w') as f:
            json.dump(self.slow_frames_log, f, indent=2)