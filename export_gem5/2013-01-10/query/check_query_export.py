#!/usr/bin/env python3
import os
import json
import numpy as np
from pathlib import Path

_DTYPE = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}

def _bytes_of(dtype_str: str) -> int:
    dt = _DTYPE.get(dtype_str)
    if dt is None:
        raise ValueError(f"Unsupported dtype in json: {dtype_str}")
    return np.dtype(dt).itemsize

def _file_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0

def _expect_bytes(dtype: str, shape) -> int:
    n = 1
    for x in shape:
        n *= int(x)
    return n * _bytes_of(dtype)

def _read_bin(path: Path, dtype: str, shape):
    dt = _DTYPE[dtype]
    expected = _expect_bytes(dtype, shape)
    actual = _file_bytes(path)
    if actual != expected:
        raise ValueError(f"Size mismatch for {path.name}: expected {expected} bytes, got {actual} bytes")
    arr = np.fromfile(path, dtype=dt)
    arr = arr.reshape(tuple(shape), order="C")
    return arr

def _ok(msg):  print(f"[ OK ] {msg}")
def _bad(msg): print(f"[FAIL] {msg}")

def main():
    folder = Path(os.getcwd())
    jpath = folder / "query.json"
    if not jpath.is_file():
        _bad(f"Missing query.json in {folder}")
        raise SystemExit(1)

    with open(jpath, "r") as f:
        meta = json.load(f)

    print(f"[info] folder: {folder}")
    print(f"[info] version: {meta.get('version')}, endianness: {meta.get('endianness')}, desc_bits: {meta.get('desc_bits')}")
    sess = meta.get("session", None)
    if sess is not None:
        print(f"[info] session: {sess}")

    # ---- basic required fields ----
    bins = meta.get("bins", {})
    required = [
        "query_desc_q4",
        "query_step_pose_grid",
        "query_step_offsets",
        "query_step_counts",
        "query_step_relodo_i",
    ]
    missing = [k for k in required if k not in bins]
    if missing:
        _bad(f"query.json missing bins: {missing}")
        raise SystemExit(1)

    # ---- check each bin exists and json bytes matches computed ----
    print("\n== Basic bin file checks ==")
    for k, b in bins.items():
        f = b["file"]
        dtype = b["dtype"]
        shape = b["shape"]
        bytes_json = int(b.get("bytes", -1))
        path = folder / f

        if not path.is_file():
            _bad(f"Missing file for '{k}': {f}")
            raise SystemExit(1)
        else:
            _ok(f"Found meta for '{k}' -> {f}")

        bytes_calc = _expect_bytes(dtype, shape)
        if bytes_json != -1 and bytes_json != bytes_calc:
            _bad(f"{f}: json bytes ({bytes_json}) != computed ({bytes_calc})")
            raise SystemExit(1)
        else:
            _ok(f"{f}: json bytes matches computed ({bytes_calc}B)")

        bytes_file = _file_bytes(path)
        if bytes_file != bytes_calc:
            _bad(f"{f}: file size ({bytes_file}) != expected ({bytes_calc})")
            raise SystemExit(1)
        else:
            _ok(f"{f}: file size matches expected ({bytes_calc}B)")

    # ---- load core arrays ----
    desc_meta = bins["query_desc_q4"]
    pose_meta = bins["query_step_pose_grid"]
    off_meta  = bins["query_step_offsets"]
    cnt_meta  = bins["query_step_counts"]
    rid_meta  = bins["query_step_relodo_i"]
    tnow_meta = bins.get("query_step_t_now", None)

    Q = int(desc_meta["shape"][0])
    S = int(pose_meta["shape"][0])

    offsets = _read_bin(folder / off_meta["file"], off_meta["dtype"], off_meta["shape"])
    counts  = _read_bin(folder / cnt_meta["file"], cnt_meta["dtype"], cnt_meta["shape"])
    poses   = _read_bin(folder / pose_meta["file"], pose_meta["dtype"], pose_meta["shape"])
    relodo  = _read_bin(folder / rid_meta["file"], rid_meta["dtype"], rid_meta["shape"])
    # desc is big; mmap-style load
    desc = _read_bin(folder / desc_meta["file"], desc_meta["dtype"], desc_meta["shape"])

    if tnow_meta is not None:
        tnow = _read_bin(folder / tnow_meta["file"], tnow_meta["dtype"], tnow_meta["shape"])
    else:
        tnow = None

    print("\n== Step structure consistency ==")
    # offsets length should be S+1
    if offsets.shape[0] != S + 1:
        _bad(f"offsets length = {offsets.shape[0]} but steps S={S} => expect S+1={S+1}")
        raise SystemExit(1)
    _ok(f"Offsets length = {offsets.shape[0]} => S={S}")

    # monotonic and starts at 0
    if offsets[0] != 0:
        _bad(f"offsets[0] != 0 (got {offsets[0]})")
        raise SystemExit(1)
    _ok("offsets[0] == 0")

    if np.any(offsets[1:] < offsets[:-1]):
        _bad("Offsets are not monotonic non-decreasing")
        raise SystemExit(1)
    _ok("Offsets are monotonic non-decreasing")

    # offsets[-1] equals total desc rows
    if int(offsets[-1]) != Q:
        _bad(f"offsets[-1] = {int(offsets[-1])} but total_rows Q={Q}")
        raise SystemExit(1)
    _ok(f"offsets[-1] matches total_rows (Q={Q})")

    # counts length should be S
    if counts.shape[0] != S:
        _bad(f"counts length = {counts.shape[0]} but steps S={S}")
        raise SystemExit(1)
    _ok("counts length == steps")

    # counts sum should equal Q
    s_counts = int(counts.astype(np.int64).sum())
    if s_counts != Q:
        _bad(f"sum(counts) = {s_counts} but Q={Q}")
        raise SystemExit(1)
    _ok("sum(counts) == total_rows")

    # poses length should be S and values in range
    if poses.shape[0] != S or poses.shape[1] != 2:
        _bad(f"poses shape is {poses.shape}, expected ({S},2)")
        raise SystemExit(1)
    _ok("pose grid shape ok")

    # relodo length should be S
    if relodo.shape[0] != S:
        _bad(f"relodo_i length {relodo.shape[0]} != S={S}")
        raise SystemExit(1)
    _ok("relodo_i length == steps")

    if tnow is not None and tnow.shape[0] != S:
        _bad(f"t_now length {tnow.shape[0]} != S={S}")
        raise SystemExit(1)
    if tnow is not None:
        _ok("t_now length == steps")

    # Optional: check per-step slice boundaries
    # offsets[i+1]-offsets[i] should equal counts[i]
    d = offsets[1:] - offsets[:-1]
    if not np.all(d.astype(np.int64) == counts.astype(np.int64)):
        bad_idx = np.where(d.astype(np.int64) != counts.astype(np.int64))[0][:10]
        _bad(f"offsets diff != counts at indices (show first 10): {bad_idx.tolist()}")
        raise SystemExit(1)
    _ok("offsets diffs match counts per step")

    # ---- value range checks ----
    print("\n== Descriptor q4 sanity ==")
    vmin = int(desc.min()) if desc.size else 0
    vmax = int(desc.max()) if desc.size else 0
    print(f"[stats] desc rows={Q}, dim=64, range=[{vmin},{vmax}]")
    if vmax > 15 or vmin < 0:
        _bad("query_desc_q4 has values out of [0,15]")
        raise SystemExit(1)
    _ok("query_desc_q4 value range looks like 4-bit (0..15)")

    print("\n== Geometry grid sanity ==")
    geo = meta.get("geo_grid", {})
    geo_max = int(geo.get("max", 255))
    gmin = int(poses.min()) if poses.size else 0
    gmax = int(poses.max()) if poses.size else 0
    print(f"[stats] pose grid range=[{gmin},{gmax}], geo_max={geo_max}")
    if gmin < 0 or gmax > geo_max:
        _bad("query_step_pose_grid has values out of [0, geo_max]")
        raise SystemExit(1)
    _ok("query_step_pose_grid within [0, geo_max]")

    # ---- spot-check per-step descriptor slices ----
    print("\n== Spot-check step slices ==")
    # sample up to 10 steps evenly
    if S > 0:
        sample_idx = np.linspace(0, S - 1, num=min(10, S), dtype=int)
        for si in sample_idx:
            a = int(offsets[si])
            b = int(offsets[si + 1])
            if b < a or b > Q:
                _bad(f"step {si}: bad slice [{a},{b}) with Q={Q}")
                raise SystemExit(1)
            # light check: slice shape
            block = desc[a:b]
            if block.ndim != 2 or block.shape[1] != 64:
                _bad(f"step {si}: desc slice shape {block.shape} unexpected")
                raise SystemExit(1)
        _ok(f"Step slices ok for sampled steps: {sample_idx.tolist()}")

    print("\n✅ ALL CHECKS PASSED")

if __name__ == "__main__":
    main()