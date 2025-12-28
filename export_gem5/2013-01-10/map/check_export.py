#!/usr/bin/env python3
import os, json, sys
import numpy as np

DTYPE_MAP = {
    "uint8":  np.uint8,
    "int32":  np.int32,
    "uint32": np.uint32,
    "float32": np.float32,
}

def human(n: int) -> str:
    # bytes -> human
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n}B"

def fail(msg: str):
    print(f"[FAIL] {msg}")
    return False

def ok(msg: str):
    print(f"[ OK ] {msg}")
    return True

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def check_bin(path: str, meta: dict) -> bool:
    """
    Validate:
      - file exists
      - file size matches expected
      - dtype known
      - shape product matches bytes/dtype
    """
    good = True
    if not os.path.isfile(path):
        return fail(f"Missing file: {path}")

    dtype_name = meta["dtype"]
    if dtype_name not in DTYPE_MAP:
        return fail(f"Unknown dtype '{dtype_name}' in {path}")

    dtype = DTYPE_MAP[dtype_name]
    shape = tuple(meta["shape"])
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

    actual_bytes = os.path.getsize(path)

    if "bytes" in meta:
        declared = int(meta["bytes"])
        if declared != expected_bytes:
            good &= fail(f"{os.path.basename(path)}: json bytes={declared} != computed bytes={expected_bytes}")
        else:
            good &= ok(f"{os.path.basename(path)}: json bytes matches computed ({human(expected_bytes)})")

    if actual_bytes != expected_bytes:
        good &= fail(f"{os.path.basename(path)}: file size={actual_bytes} != expected={expected_bytes}")
    else:
        good &= ok(f"{os.path.basename(path)}: file size matches expected ({human(actual_bytes)})")

    # c_order check (informational)
    if meta.get("c_order", True) is not True:
        print(f"[WARN] {os.path.basename(path)}: c_order is not true (your reader must handle this)")

    return good

def mmap_array(path: str, dtype: np.dtype, shape: tuple):
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order="C")

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    jpath = os.path.join(root, "map.json")
    if not os.path.isfile(jpath):
        print("ERROR: map.json not found in current folder.")
        sys.exit(1)

    meta = load_json(jpath)
    bins = meta.get("bins", {})
    if not bins:
        print("ERROR: 'bins' is empty in map.json")
        sys.exit(1)

    print(f"[info] folder: {root}")
    print(f"[info] version: {meta.get('version')}, endianness: {meta.get('endianness')}, desc_bits: {meta.get('desc_bits')}")
    print("")

    all_ok = True

    # 1) Basic per-bin checks
    print("== Basic bin file checks ==")
    for name, b in bins.items():
        f = b["file"]
        path = os.path.join(root, f)
        if not ok(f"Found meta for '{name}' -> {f}"):
            pass
        all_ok &= check_bin(path, b)
    print("")

    # 2) Offsets/postings consistency checks
    print("== IVF postings consistency ==")
    # Required bins
    need = ["ivf_postings_offsets", "ivf_postings_map_ids"]
    for k in need:
        if k not in bins:
            all_ok &= fail(f"Missing '{k}' in map.json bins")
    if all_ok:
        offm = bins["ivf_postings_offsets"]
        pm   = bins["ivf_postings_map_ids"]

        off_path = os.path.join(root, offm["file"])
        p_path   = os.path.join(root, pm["file"])

        offsets = mmap_array(off_path, DTYPE_MAP[offm["dtype"]], tuple(offm["shape"]))
        postings = mmap_array(p_path, DTYPE_MAP[pm["dtype"]], tuple(pm["shape"]))

        Kp1 = offsets.shape[0]
        K = Kp1 - 1
        ok(f"Offsets length = {Kp1} => K={K}")

        # monotonic non-decreasing
        diffs = np.diff(offsets.astype(np.int64))
        if np.any(diffs < 0):
            idx = int(np.where(diffs < 0)[0][0])
            all_ok &= fail(f"Offsets not monotonic at idx {idx}: off[{idx}]={int(offsets[idx])}, off[{idx+1}]={int(offsets[idx+1])}")
        else:
            all_ok &= ok("Offsets are monotonic non-decreasing")

        # offsets[0] == 0
        if int(offsets[0]) != 0:
            all_ok &= fail(f"offsets[0] should be 0 but is {int(offsets[0])}")
        else:
            all_ok &= ok("offsets[0] == 0")

        # offsets[-1] == len(postings)
        if int(offsets[-1]) != postings.shape[0]:
            all_ok &= fail(f"offsets[-1]={int(offsets[-1])} != postings_len={postings.shape[0]}")
        else:
            all_ok &= ok(f"offsets[-1] matches postings length ({postings.shape[0]})")

        # List size stats
        sizes = diffs
        nonempty = int(np.sum(sizes > 0))
        empty = int(np.sum(sizes == 0))
        max_sz = int(np.max(sizes)) if sizes.size else 0
        min_nz = int(np.min(sizes[sizes > 0])) if np.any(sizes > 0) else 0
        avg_sz = float(np.mean(sizes[sizes > 0])) if np.any(sizes > 0) else 0.0

        print(f"[stats] lists: {K}, nonempty: {nonempty}, empty: {empty}")
        print(f"[stats] list size: max={max_sz}, min_nonzero={min_nz}, avg_nonzero={avg_sz:.2f}")

        # postings sanity: map id range
        # sample only to keep fast, but also check min/max
        pmin = int(np.min(postings))
        pmax = int(np.max(postings))
        ok(f"postings map_id range: [{pmin}, {pmax}]")

        # Optional: check for negative ids
        if pmin < 0:
            all_ok &= fail(f"postings contain negative ids (min={pmin})")
        else:
            all_ok &= ok("postings have no negative ids")
    print("")

    # 3) Descriptor quant sanity (q4)
    print("== Descriptor q4 sanity ==")
    if "map_desc_q4" in bins:
        dm = bins["map_desc_q4"]
        d = mmap_array(os.path.join(root, dm["file"]), DTYPE_MAP[dm["dtype"]], tuple(dm["shape"]))
        vmax = int(np.max(d))
        vmin = int(np.min(d))
        ok(f"map_desc_q4 value range: [{vmin}, {vmax}]")
        # Expect 0..15 for 4-bit; allow <=15
        if vmax > 15:
            all_ok &= fail(f"map_desc_q4 has values > 15 (max={vmax}) => not 4-bit packed-in-u8")
        else:
            all_ok &= ok("map_desc_q4 max <= 15 (looks like 4-bit in uint8)")
    else:
        print("[WARN] map_desc_q4 not found in bins; skipping")
    print("")

    # 4) Geometry grid sanity
    print("== Geometry grid sanity ==")
    if "map_geo_grid" in bins:
        gm = bins["map_geo_grid"]
        g = mmap_array(os.path.join(root, gm["file"]), DTYPE_MAP[gm["dtype"]], tuple(gm["shape"]))
        gmin = int(np.min(g))
        gmax = int(np.max(g))
        ok(f"map_geo_grid value range: [{gmin}, {gmax}]")
        # If json has geo_grid.max, compare
        gg = meta.get("geo_grid", {})
        if "max" in gg:
            declared_max = int(gg["max"])
            if gmax > declared_max:
                all_ok &= fail(f"grid data max={gmax} exceeds json geo_grid.max={declared_max} (quantize/clamp mismatch?)")
            else:
                all_ok &= ok(f"grid data max <= geo_grid.max ({declared_max})")
    else:
        print("[WARN] map_geo_grid not found in bins; skipping")
    print("")

    # 5) Centroids / scales sanity
    print("== IVF extra arrays sanity ==")
    for key, expect_shape in [
        ("ivf_centroids", None),        # check it's 2D and second dim is 128 if present
        ("ivf_val_scale", (64,)),
        ("ivf_idf_w", (64,))
    ]:
        if key not in bins:
            print(f"[WARN] {key} not found; skipping")
            continue
        bm = bins[key]
        arr = mmap_array(os.path.join(root, bm["file"]), DTYPE_MAP[bm["dtype"]], tuple(bm["shape"]))
        if expect_shape is not None and tuple(arr.shape) != expect_shape:
            all_ok &= fail(f"{key} shape {arr.shape} != expected {expect_shape}")
        else:
            ok(f"{key} shape ok: {arr.shape}")

        if key == "ivf_centroids":
            if arr.ndim != 2:
                all_ok &= fail(f"ivf_centroids should be 2D but is {arr.ndim}D")
            else:
                # In your json it is (128,128)
                if arr.shape[1] != 128:
                    all_ok &= fail(f"ivf_centroids second dim should be 128 (aug space) but is {arr.shape[1]}")
                else:
                    ok("ivf_centroids second dim == 128 (aug space)")

    print("")
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED (see messages above)")
        sys.exit(2)

if __name__ == "__main__":
    main()