# src/export_gem5.py
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import utils_.feature_utils as feature_utils


# ---------------------------
# Binary writers
# ---------------------------

def write_bin_raw(path: Path, arr: np.ndarray) -> None:
    """Write raw binary only (no metadata)."""
    path = Path(path)
    arr_c = np.ascontiguousarray(arr)
    with open(path, "wb") as f:
        f.write(arr_c.tobytes(order="C"))


def write_bin_meta(path: Path, arr: np.ndarray) -> Dict[str, Any]:
    """Write raw binary and return JSON metadata dict."""
    path = Path(path)
    arr_c = np.ascontiguousarray(arr)

    # enforce little-endian for numeric types (safe on x86; explicit for portability)
    if arr_c.dtype.kind in ("i", "u", "f") and arr_c.dtype.byteorder in (">", "!"):
        arr_c = arr_c.byteswap().newbyteorder("<")

    with open(path, "wb") as f:
        f.write(arr_c.tobytes(order="C"))

    return {
        "file": path.name,
        "dtype": str(arr_c.dtype),
        "shape": list(arr_c.shape),
        "c_order": True,
        "bytes": int(arr_c.nbytes),
    }


def file_bytes(path: Path) -> int:
    p = Path(path)
    return p.stat().st_size if p.is_file() else 0


# ---------------------------
# IVF extraction
# ---------------------------

def extract_ivf_lists_and_centroids(ivf):
    """
    Return:
      K: int
      centroids: (K,D) float32
      postings_offsets: (K+1,) uint32
      postings_map_ids: (total,) int32
      extra: dict including optional arrays (val_scale, idf_w) + scalars
    """
    extra: Dict[str, Any] = {}
    
    # 紀錄 class name 方便在 map.json 辨識是用哪個方法產生的
    extra["ivf_impl"] = ivf.__class__.__name__

    # -------------------------------------------------------
    # 1. ZeroAwareIVF (Has _aug_centroids and _lists)
    # -------------------------------------------------------
    if hasattr(ivf, "_aug_centroids") and hasattr(ivf, "_lists"):
        C = getattr(ivf, "_aug_centroids", None)
        lists = getattr(ivf, "_lists", None)
        if C is None or lists is None:
            raise ValueError("ZeroAwareIVF missing _aug_centroids/_lists. Did you call build()?")

        centroids = np.asarray(C, dtype=np.float32)
        K = int(centroids.shape[0])

        # parameters needed to project query into the same augmented space
        vs = getattr(ivf, "_val_scale", None)
        iw = getattr(ivf, "_idf_w", None)
        if vs is not None:
            extra["val_scale"] = np.asarray(vs, dtype=np.float32)  # (64,)
        if iw is not None:
            extra["idf_w"] = np.asarray(iw, dtype=np.float32)      # (64,)

        extra["alpha_value"] = float(getattr(ivf, "alpha_value", 1.0))
        extra["beta_presence"] = float(getattr(ivf, "beta_presence", 5.0))
        extra["use_idf"] = int(bool(getattr(ivf, "use_idf", True)))

        # postings: lid -> [(map_id, row_idx), ...]
        offsets = np.zeros((K + 1,), dtype=np.uint32)
        all_ids = []
        cur = 0
        for lid in range(K):
            entries = lists.get(lid, [])
            mids = [int(mid) for (mid, _row) in entries]  # store only map_id
            all_ids.extend(mids)
            cur += len(mids)
            offsets[lid + 1] = cur

        postings_map_ids = np.asarray(all_ids, dtype=np.int32)
        return K, centroids, offsets, postings_map_ids, extra

    # -------------------------------------------------------
    # 2. Baselines (StandardL2 / BinaryHamming)
    #    特徵: 有 centroids (numpy) 和 _lists (dict)
    # -------------------------------------------------------
    if hasattr(ivf, "centroids") and hasattr(ivf, "_lists"):
        centroids = np.asarray(ivf.centroids, dtype=np.float32)
        K = int(centroids.shape[0])
        lists = getattr(ivf, "_lists") # Access the internal dict

        offsets = np.zeros((K + 1,), dtype=np.uint32)
        all_ids = []
        cur = 0
        for lid in range(K):
            # Baseline 的 _lists 結構也是 [(mid, row_idx), ...]
            entries = lists.get(lid, [])
            mids = [int(mid) for (mid, _row) in entries]
            all_ids.extend(mids)
            cur += len(mids)
            offsets[lid + 1] = cur
        
        postings_map_ids = np.asarray(all_ids, dtype=np.int32)
        
        # BinaryHamming 的 centroids 是 float (means)，但在 gem5 裡
        # 我們一樣用 L2 distance 去算 (作為 soft hamming)，所以不需要額外轉換
        return K, centroids, offsets, postings_map_ids, extra

    # -------------------------------------------------------
    # 3. Generic/Legacy KMeansIVF (best-effort)
    #    特徵: 有 centroids 和 lists (無底線)
    # -------------------------------------------------------
    if hasattr(ivf, "centroids"):
        centroids = np.asarray(ivf.centroids, dtype=np.float32)
        K = int(centroids.shape[0])
        if hasattr(ivf, "lists"):
            lists = ivf.lists
            offsets = np.zeros((K + 1,), dtype=np.uint32)
            all_ids = []
            cur = 0
            for lid in range(K):
                # 假設 legacy lists 直接存 [mid, mid, ...] 或類似結構
                # 這裡為了保險起見，假設是 list of IDs
                raw_list = lists.get(lid, [])
                # 嘗試判斷內容物
                if raw_list and isinstance(raw_list[0], (tuple, list)):
                     mids = [int(x[0]) for x in raw_list]
                else:
                     mids = [int(x) for x in raw_list]
                     
                all_ids.extend(mids)
                cur += len(mids)
                offsets[lid + 1] = cur
            postings_map_ids = np.asarray(all_ids, dtype=np.int32)
            return K, centroids, offsets, postings_map_ids, extra

        raise NotImplementedError(
            "Detected ivf.centroids but missing ivf.lists or ivf._lists."
        )

    raise NotImplementedError(
        f"Unsupported IVF object type: {type(ivf)}. "
        "Expected ZeroAwareIVF, BaselineStandardL2, or BaselineBinaryHamming."
    )


# ---------------------------
# MAP export
# ---------------------------

def export_ivf_map(
    out_dir: str,
    *,
    descmap_f32: np.ndarray,     # (N,64) float
    descxy_f32: np.ndarray,      # (N,2) float (x,y)
    ivf,                         # built IVF object
    desc_bits: int = 4,
    geo_max: int = 90,
    geo_qparams: Optional[Tuple[float, float, float]] = None,  # (min_x, min_y, R)
) -> Tuple[str, np.ndarray]:
    """
    Export IVF map (for gem5):
      - map_desc_q4.bin          (N,64) uint8 0..15
      - map_geo_grid.bin         (N,2)  uint8 0..geo_max
      - ivf_centroids.bin        (K,D)  float32
      - ivf_postings_offsets.bin (K+1)  uint32
      - ivf_postings_map_ids.bin (total) int32
      - optional ivf_val_scale.bin / ivf_idf_w.bin
      - map.json

    Returns:
      (map_json_path, thresholds_q4)
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    if desc_bits != 4:
        raise ValueError("export_ivf_map currently expects desc_bits=4 for gem5 input.")

    # 1) descriptor quantization to q4 (0..15, 0 for zero)
    desc_q, thresholds = feature_utils.quantize_descmap(descmap_f32, bits=desc_bits)
    map_desc_q4 = np.asarray(desc_q, dtype=np.uint8)

    # 2) geometry quantization (shared span)
    if geo_qparams is None:
        min_x = float(descxy_f32[:, 0].min())
        max_x = float(descxy_f32[:, 0].max())
        min_y = float(descxy_f32[:, 1].min())
        max_y = float(descxy_f32[:, 1].max())
        R = float(max(max_x - min_x, max_y - min_y))
        geo_qparams = (min_x, min_y, R)

    min_x, min_y, R = geo_qparams
    map_geo_grid = feature_utils.quantize_xy_array_to_u6_shared(
        descxy_f32, min_x, min_y, R, MAX=int(geo_max)
    ).astype(np.uint8, copy=False)

    # 3) IVF export
    K, centroids, offsets, postings_map_ids, extra = extract_ivf_lists_and_centroids(ivf)

    # 4) write bins + json
    meta: Dict[str, Any] = {
        "version": 1,
        "endianness": "little",
        "desc_bits": int(desc_bits),
        "geo_grid": {
            "dtype": "uint8",
            "max": int(geo_max),
            "qparams": [float(min_x), float(min_y), float(R)],
            "meaning": "grid_x, grid_y in [0..max]; computed by shared span R",
        },
        "ivf": {
            "nlist": int(K),
            "space": "aug128_l2" if int(centroids.shape[1]) == 128 else f"l2_dim{int(centroids.shape[1])}",
            "extra": {},
        },
        "bins": {},
        "desc_quant": {
            "method": "quantile_digitize_nonzero",
            "thresholds": [float(x) for x in np.asarray(thresholds).ravel()],
            "note": "q=0 means zero; nonzero mapped to 1..15 for 4-bit",
        },
    }

    meta["bins"]["map_desc_q4"] = write_bin_meta(outp / "map_desc_q4.bin", map_desc_q4)
    meta["bins"]["map_geo_grid"] = write_bin_meta(outp / "map_geo_grid.bin", map_geo_grid)
    meta["bins"]["ivf_centroids"] = write_bin_meta(outp / "ivf_centroids.bin", np.asarray(centroids, dtype=np.float32))
    meta["bins"]["ivf_postings_offsets"] = write_bin_meta(outp / "ivf_postings_offsets.bin", np.asarray(offsets, dtype=np.uint32))
    meta["bins"]["ivf_postings_map_ids"] = write_bin_meta(outp / "ivf_postings_map_ids.bin", np.asarray(postings_map_ids, dtype=np.int32))

    for k, v in extra.items():
        if isinstance(v, np.ndarray):
            key = f"ivf_{k}"
            meta["bins"][key] = write_bin_meta(outp / f"{key}.bin", v)
            meta["ivf"]["extra"][k] = {"ref": key}
        else:
            meta["ivf"]["extra"][k] = v

    json_path = outp / "map.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(json_path), np.asarray(thresholds)


# ---------------------------
# QUERY export (stream)
# ---------------------------

class QueryExportWriter:
    """
    Export query stream:
      - query_desc_q4.bin         : (Q,64)  uint8
      - query_step_pose_grid.bin  : (S,2)   uint8
      - query_step_offsets.bin    : (S+1,)  uint32   (prefix sum into Q)
      - query_step_counts.bin     : (S,)    uint16
      - query_step_relodo_i.bin   : (S,)    int32
      - query_step_t_now.bin      : (S,)    float64  (optional)
      - query.json
    """
    def __init__(
        self,
        out_dir: str,
        sessionname: str,
        desc_bits: int,
        geo_max: int,
        geo_qparams: Tuple[float, float, float],  # (min_x, min_y, R)
        map_json_ref: Optional[str] = None,
        save_t_now: bool = True,
        overwrite: bool = True,
    ):
        self.outp = Path(out_dir)
        self.outp.mkdir(parents=True, exist_ok=True)

        self.sessionname = sessionname
        self.desc_bits = int(desc_bits)
        self.geo_max = int(geo_max)
        self.geo_qparams = (float(geo_qparams[0]), float(geo_qparams[1]), float(geo_qparams[2]))
        self.map_json_ref = map_json_ref
        self.save_t_now = bool(save_t_now)

        mode = "wb" if overwrite else "ab"
        self.f_desc = open(self.outp / "query_desc_q4.bin", mode)
        self.f_pose = open(self.outp / "query_step_pose_grid.bin", mode)

        self.offsets = [0]   # prefix sum in rows
        self.counts = []     # uint16
        self.relodo_i = []   # int32
        self.t_now = []      # float64

        self.total_rows = 0
        self.steps = 0

    def add_step(
        self,
        qrx: int,
        qry: int,
        desc_q4: np.ndarray,     # (M,64) uint8 0..15
        relodo_i: int,
        t_now: Optional[float] = None,
    ) -> None:
        if desc_q4 is None:
            return
        desc_q4 = np.asarray(desc_q4)
        if desc_q4.ndim != 2 or desc_q4.shape[1] != 64:
            raise ValueError(f"desc_q4 must be (M,64), got {desc_q4.shape}")
        if desc_q4.dtype != np.uint8:
            desc_q4 = desc_q4.astype(np.uint8, copy=False)

        M = int(desc_q4.shape[0])
        if M <= 0:
            return

        # write desc rows
        self.f_desc.write(desc_q4.tobytes(order="C"))

        # write pose grid for this step
        pose = np.array([[int(qrx), int(qry)]], dtype=np.uint8)
        self.f_pose.write(pose.tobytes(order="C"))

        # meta
        self.total_rows += M
        self.steps += 1
        self.offsets.append(self.total_rows)
        self.counts.append(M)
        self.relodo_i.append(int(relodo_i))
        if self.save_t_now:
            self.t_now.append(float(t_now) if t_now is not None else float("nan"))

    def close(self) -> str:
        self.f_desc.close()
        self.f_pose.close()

        offsets = np.asarray(self.offsets, dtype=np.uint32)
        counts = np.asarray(self.counts, dtype=np.uint16)
        relodo_i = np.asarray(self.relodo_i, dtype=np.int32)

        write_bin_raw(self.outp / "query_step_offsets.bin", offsets)
        write_bin_raw(self.outp / "query_step_counts.bin", counts)
        write_bin_raw(self.outp / "query_step_relodo_i.bin", relodo_i)
        if self.save_t_now:
            write_bin_raw(self.outp / "query_step_t_now.bin", np.asarray(self.t_now, dtype=np.float64))

        j: Dict[str, Any] = {
            "version": 1,
            "endianness": "little",
            "session": self.sessionname,
            "desc_bits": int(self.desc_bits),
            "geo_grid": {
                "dtype": "uint8",
                "max": int(self.geo_max),
                "qparams": [float(self.geo_qparams[0]), float(self.geo_qparams[1]), float(self.geo_qparams[2])],
                "meaning": "grid_x, grid_y in [0..max]; computed by shared span R",
            },
            "ivf": {
                "centroids_ref": "use map.json/ivf_centroids.bin",
                "postings_ref": "use map.json/ivf_postings_*.bin",
                "map_json_ref": self.map_json_ref,
            },
            "bins": {
                "query_desc_q4": {
                    "file": "query_desc_q4.bin",
                    "dtype": "uint8",
                    "shape": [int(self.total_rows), 64],
                    "c_order": True,
                    "bytes": file_bytes(self.outp / "query_desc_q4.bin"),
                },
                "query_step_pose_grid": {
                    "file": "query_step_pose_grid.bin",
                    "dtype": "uint8",
                    "shape": [int(self.steps), 2],
                    "c_order": True,
                    "bytes": file_bytes(self.outp / "query_step_pose_grid.bin"),
                },
                "query_step_offsets": {
                    "file": "query_step_offsets.bin",
                    "dtype": "uint32",
                    "shape": [int(self.steps) + 1],
                    "c_order": True,
                    "bytes": file_bytes(self.outp / "query_step_offsets.bin"),
                },
                "query_step_counts": {
                    "file": "query_step_counts.bin",
                    "dtype": "uint16",
                    "shape": [int(self.steps)],
                    "c_order": True,
                    "bytes": file_bytes(self.outp / "query_step_counts.bin"),
                },
                "query_step_relodo_i": {
                    "file": "query_step_relodo_i.bin",
                    "dtype": "int32",
                    "shape": [int(self.steps)],
                    "c_order": True,
                    "bytes": file_bytes(self.outp / "query_step_relodo_i.bin"),
                },
            },
        }
        if self.save_t_now:
            j["bins"]["query_step_t_now"] = {
                "file": "query_step_t_now.bin",
                "dtype": "float64",
                "shape": [int(self.steps)],
                "c_order": True,
                "bytes": file_bytes(self.outp / "query_step_t_now.bin"),
            }

        with open(self.outp / "query.json", "w") as f:
            json.dump(j, f, indent=2)

        return str(self.outp / "query.json")


def export_queries_begin(
    out_dir: str,
    sessionname: str,
    *,
    desc_bits: int,
    geo_max: int,
    geo_qparams: Tuple[float, float, float],
    map_json_ref: Optional[str],
    overwrite: bool = True,
) -> QueryExportWriter:
    return QueryExportWriter(
        out_dir=out_dir,
        sessionname=sessionname,
        desc_bits=desc_bits,
        geo_max=geo_max,
        geo_qparams=geo_qparams,
        map_json_ref=map_json_ref,
        save_t_now=True,
        overwrite=overwrite,
    )