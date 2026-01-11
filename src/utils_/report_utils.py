import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pynclt  

def plot_global_map(globalmapfile):
    data = np.load(globalmapfile)
    x, y = data['polemeans'][:, :2].T
    plt.clf()
    plt.scatter(x, y, s=1, c='b', marker='.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.savefig(globalmapfile[:-4] + '.svg')
    # plt.savefig(globalmapfile[:-4] + '.pgf')
    print("[Report] Map Factors:", data['mapfactors'])

def plot_evaluation_result(sessionname, poserror_mean, poserror_mean_knn, t_plot, files, result_dir):
    """
    Plots positional error and measurement events.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(f'Positional Error and Measurement Timestamps — {sessionname}')

    # ax1: baseline pos error
    ax1.plot(t_plot, poserror_mean, color='blue', label='poserror (mean over runs)')
    ax1.set_ylabel('poserror (m)')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right')

    # ax2: kNN pos error
    ax2.plot(t_plot, poserror_mean_knn, color='green', label='poserror (kNN mean over runs)')
    ax2.set_ylabel('poserror (m)')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='upper right')

    # Collect measurement events
    meas_times_all = []
    n_active_all = []
    for file in files:
        fpath = os.path.join(result_dir, sessionname, file)
        try:
            data = np.load(fpath, allow_pickle=True)
            if 'meas_events' in data.files:
                events = data['meas_events']
                for e in events:
                    try:
                        d = e if isinstance(e, dict) else (e.item() if hasattr(e, 'item') else {})
                        meas_times_all.append(float(d.get('t_now')))
                        n_active_all.append(int(d.get('n_active')))
                    except: pass
        except: pass

    meas_times_all = np.array(meas_times_all)
    n_active_all = np.array(n_active_all)

    # ax3: n_active
    ax3.set_xlabel('timestamp (t_eval)')
    ax3.set_ylabel('n_active')
    ax3.grid(True, linestyle='--', alpha=0.4)
    if meas_times_all.size > 0:
        mask = (meas_times_all >= t_plot[0]) & (meas_times_all <= t_plot[-1])
        ax3.plot(meas_times_all[mask], n_active_all[mask], label='n_active', linewidth=1.2)
        ax3.legend(loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figpath = os.path.join(result_dir, f"{sessionname}_poserror_meas.png")
    plt.savefig(figpath, dpi=300)
    plt.close(fig)
    print(f"[Report] Saved evaluation plot: {figpath}")

def plot_timing_stacked_for_sessions(
    sessions,
    base_dir="nclt",
    mask_measurement_only=True,
    max_bars=3000,
):
    """
    Stacked time series bar chart per session:
      x-axis: timestamp (from meas_events.t_now when available; else step index)
      y-axis: time per step (ms)
      stack:  update_measurement (bottom) + estimate_pose + kNN(total) + other

    Mask: by default only steps with tim_update_measurement_ms > 0.
    Downsampling: if steps > max_bars, group by stride and plot group means.

    Output per session: nclt/{session}/timing_stacked4_{session}.png
    """
    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localization*.npz")))
        if not paths:
            print(f"[timing] skip: no localization*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[timing] skip: failed to load {fpath}: {e}")
            continue

        required = ["tim_total_ms", "tim_update_measurement_ms", "tim_estimate_pose_ms"]
        if not all(k in data.files for k in required):
            print(f"[timing] skip: {fpath} missing required arrays {required}")
            continue

        total    = data["tim_total_ms"].astype(np.float32)
        update   = data["tim_update_measurement_ms"].astype(np.float32)
        estimate = data["tim_estimate_pose_ms"].astype(np.float32)

        # kNN parts are optional; default to zeros when missing
        knn_u = data["tim_knn_update_ms"].astype(np.float32)        if "tim_knn_update_ms" in data.files else np.zeros_like(total)
        knn_m = data["tim_knn_measurement_ms"].astype(np.float32)   if "tim_knn_measurement_ms" in data.files else np.zeros_like(total)
        knn_e = data["tim_knn_estimate_ms"].astype(np.float32)      if "tim_knn_estimate_ms" in data.files else np.zeros_like(total)
        knn = knn_u + knn_m + knn_e
        
        n_active = data["n_active_per_step"].astype(np.float32) if "n_active_per_step" in data.files else None

        # mask: only measurement steps by default
        mask = (update > 0.0) if mask_measurement_only else np.ones_like(update, dtype=bool)
        if not np.any(mask):
            print(f"[timing] skip: no steps after masking in {fpath}")
            continue

        total_m, update_m, estimate_m, knn_msk = total[mask], update[mask], estimate[mask], knn[mask]
        other_m = np.maximum(0.0, total_m - update_m - estimate_m - knn_msk)

        # timestamps: prefer meas_events.t_now if it matches, otherwise use masked indices
        ts = None
        if "meas_events" in data.files:
            try:
                ev = data["meas_events"]
                t_list = []
                for e in ev:
                    try:
                        d = e if isinstance(e, dict) else (e.item() if hasattr(e, "item") else {})
                        t_list.append(float(d.get("t_now")))
                    except Exception:
                        pass
                t_arr = np.array(t_list, dtype=float)
                if t_arr.size == total_m.size:
                    ts = t_arr
            except Exception:
                pass
        if ts is None:
            ts = np.flatnonzero(mask).astype(float)

        # downsample if too many bars (group means)
        n = total_m.size
        stride = int(np.ceil(n / max(1, max_bars)))
        if stride > 1:
            g = (n // stride) * stride

            def group_mean(a):
                return a[:g].reshape(-1, stride).mean(axis=1)

            ts_plot       = group_mean(ts)
            update_plot   = group_mean(update_m)
            estimate_plot = group_mean(estimate_m)
            knn_plot      = group_mean(knn_msk)
            other_plot    = group_mean(other_m)
            count_used    = ts_plot.size
            
            if n_active is not None:
                n_active_plot = group_mean(n_active[mask])
            else:
                n_active_plot = None
        else:
            ts_plot, update_plot, estimate_plot, knn_plot, other_plot = ts, update_m, estimate_m, knn_msk, other_m
            count_used = ts_plot.size
            n_active_plot = n_active[mask] if n_active is not None else None

        # bar width from timestamp spacing
        if ts_plot.size >= 2:
            dt = np.diff(ts_plot).mean()
            width = dt * 0.9 if dt > 0 else 1.0
        else:
            width = 1.0

        # stacked bars: update (bottom) + estimate + knn + other
        fig, ax = plt.subplots(figsize=(12, 5))
        b0 = update_plot
        b1 = b0 + estimate_plot
        b2 = b1 + knn_plot
        ax.bar(ts_plot, update_plot,   width=width, label="update_measurement (ms)")
        ax.bar(ts_plot, estimate_plot, width=width, bottom=b0, label="estimate_pose (ms)")
        ax.bar(ts_plot, knn_plot,      width=width, bottom=b1, label="kNN total (ms)")
        ax.bar(ts_plot, other_plot,    width=width, bottom=b2, label="other (ms)")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("time per step (ms)")
        ax.grid(True, linestyle="--", alpha=0.35)

        if n_active_plot is not None and n_active_plot.size == ts_plot.size:
            ax2 = ax.twinx()
            ax2.plot(ts_plot, n_active_plot, linewidth=1.0, color="k", alpha=0.7, label="n_active")
            ax2.set_ylabel("n_active (count)")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax.legend(loc="upper right")

        plt.tight_layout()
        out_path = os.path.join(sess_dir, f"timing_stacked4_{sess}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[timing] saved: {out_path}  (bars={count_used}, stride={stride})")
        
def report_max_timing_for_sessions(
    sessions,
    base_dir="nclt",
    mask_measurement_only=True,
    max_bars=3000, 
):
    """
    For each session:
      - load nclt/{session}/localization*.npz
      - mask to measurement steps if requested
      - find the step with maximum tim_total_ms
      - print timestamp and breakdown (update/estimate/kNN/other/total) to stdout
    """
    import os, glob
    import numpy as np

    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localization*.npz")))
        if not paths:
            print(f"[max-timing] skip: no localization*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[max-timing] skip: failed to load {fpath}: {e}")
            continue

        required = ["tim_total_ms", "tim_update_measurement_ms", "tim_estimate_pose_ms"]
        if not all(k in data.files for k in required):
            print(f"[max-timing] skip: {fpath} missing required arrays {required}")
            continue

        total    = data["tim_total_ms"].astype(np.float64)
        update   = data["tim_update_measurement_ms"].astype(np.float64)
        estimate = data["tim_estimate_pose_ms"].astype(np.float64)

        # kNN parts are optional
        knn_u = data["tim_knn_update_ms"].astype(np.float64)       if "tim_knn_update_ms" in data.files else np.zeros_like(total)
        knn_m = data["tim_knn_measurement_ms"].astype(np.float64)  if "tim_knn_measurement_ms" in data.files else np.zeros_like(total)
        knn_e = data["tim_knn_estimate_ms"].astype(np.float64)     if "tim_knn_estimate_ms" in data.files else np.zeros_like(total)
        knn = knn_u + knn_m + knn_e

        # mask: measurement-only or all steps
        mask = (update > 0.0) if mask_measurement_only else np.ones_like(update, dtype=bool)
        if not np.any(mask):
            print(f"[max-timing] skip: no steps after masking in {fpath}")
            continue

        total_m, update_m, estimate_m, knn_m = total[mask], update[mask], estimate[mask], knn[mask]
        other_m = np.maximum(0.0, total_m - update_m - estimate_m - knn_m)

        # timestamps: prefer meas_events.t_now if it matches masked length
        ts = None
        if "meas_events" in data.files:
            try:
                ev = data["meas_events"]
                t_list = []
                for e in ev:
                    try:
                        d = e if isinstance(e, dict) else (e.item() if hasattr(e, "item") else {})
                        t_list.append(float(d.get("t_now")))
                    except Exception:
                        pass
                t_arr = np.array(t_list, dtype=float)
                if t_arr.size == total_m.size:
                    ts = t_arr
            except Exception:
                pass
        if ts is None:
            ts = np.flatnonzero(mask).astype(float)  # fallback: masked step indices

        # find argmax on masked total
        k = int(np.argmax(total_m))
        tstamp   = float(ts[k])
        tot_ms   = float(total_m[k])
        upd_ms   = float(update_m[k])
        est_ms   = float(estimate_m[k])
        knn_ms   = float(knn_m[k])
        oth_ms   = max(0.0, tot_ms - upd_ms - est_ms - knn_ms)

        orig_idx = int(np.flatnonzero(mask)[k])

        print(
            f"[max-timing] session={sess}  orig_step={orig_idx}  timestamp={tstamp:.6f}  "
            f"total={tot_ms:.3f} ms  "
            f"update={upd_ms:.3f} ms  estimate={est_ms:.3f} ms  kNN={knn_ms:.3f} ms  other={oth_ms:.3f} ms"
        )
        
def plot_detect_timeline_for_sessions(
    sessions,
    base_dir="nclt",
    max_bars=3000,
    smooth_window=None,
):
    """
    Time-series bar chart of pole-detect time per local map, one figure per session.

    x-axis: timestamp per local map (from 'ts_localmaps' if present; else index)
    y-axis: detection time per local map (ms), drawn as bars
    Downsampling: if number of bars > max_bars, group by stride and plot group means
    Optional smoothing line: running mean with a centered window (smooth_window)

    Input NPZ (first match of 'localmaps*.npz' in the session dir) is expected to contain:
      - tim_detect_ms: float array of shape (N_localmaps,)
      - ts_localmaps:  float array of shape (N_localmaps,)   [optional but recommended]
    """
    def running_mean(a, w):
        if w is None or w <= 1:
            return a
        # centered running mean; trim edges to keep same length as a
        k = w // 2
        c = np.cumsum(np.pad(a, (1, 0), mode="constant"))
        m = (c[w:] - c[:-w]) / float(w)
        # pad to original length
        left  = np.full(k, m[0], dtype=a.dtype)
        right = np.full(a.size - m.size - k, m[-1], dtype=a.dtype)
        return np.concatenate([left, m, right])

    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localmaps*.npz")))
        if not paths:
            print(f"[detect-timeline] skip: no localmaps*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[detect-timeline] skip: failed to load {fpath}: {e}")
            continue

        if "tim_detect_ms" not in data.files:
            print(f"[detect-timeline] skip: {fpath} missing 'tim_detect_ms'")
            continue

        detect = data["tim_detect_ms"].astype(np.float32)
        # Preferred timestamps if present
        if "ts_localmaps" in data.files:
            ts = data["ts_localmaps"].astype(np.float64)
        else:
            # Fallback: use dense indices as x
            ts = np.arange(detect.size, dtype=np.float64)

        if ts.size != detect.size:
            # Size mismatch safety: fall back to indices
            print(f"[detect-timeline] warn: ts length != detect length in {fpath}; fallback to index x-axis")
            ts = np.arange(detect.size, dtype=np.float64)

        n = detect.size
        stride = int(np.ceil(n / max(1, max_bars)))
        if stride > 1:
            # group means to reduce bars
            g = (n // stride) * stride

            def group_mean(a):
                return a[:g].reshape(-1, stride).mean(axis=1)

            ts_plot     = group_mean(ts)
            detect_plot = group_mean(detect)
            count_used  = ts_plot.size
        else:
            ts_plot, detect_plot = ts, detect
            count_used = ts_plot.size

        # Choose a reasonable bar width from timestamp spacing
        if ts_plot.size >= 2:
            dt = np.diff(ts_plot).mean()
            width = dt * 0.9 if dt > 0 else 1.0
        else:
            width = 1.0

        # Optional smoothing line (running mean)
        smooth = running_mean(detect_plot, smooth_window)

        # Plot
        plt.figure(figsize=(12, 4.5))
        plt.bar(ts_plot, detect_plot, width=width, label="detect time (ms)")
        if smooth_window and smooth_window > 1:
            plt.plot(ts_plot, smooth, linewidth=1.25, label=f"running mean (w={smooth_window})")
        title = (f"Pole-detect time over time — {sess}\n"
                 f"N={int(n)} local maps{' (downsampled)' if stride>1 else ''}")
        plt.title(title)
        plt.xlabel("timestamp")
        plt.ylabel("time per local map (ms)")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(loc="upper right")
        plt.tight_layout()

        out_path = os.path.join(sess_dir, f"detect_timeline_{sess}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[detect-timeline] saved: {out_path}  (bars={count_used}, stride={stride})")
        
        
def report_pole_detect_timing_for_sessions(
    sessions,
    base_dir="nclt",
    drop_zeros=True,
    percentiles=(90, 95, 99),
):
    """
    For each session, read nclt/{session}/localmaps_*.npz and print a timing summary
    for tim_detect_ms:
      - N / mean / median / std / min / {percentiles} / max
      - Index of the max in the original array (local map index)
    """
    for sess in sessions:
        sess_dir = os.path.join(base_dir, sess)
        paths = sorted(glob.glob(os.path.join(sess_dir, "localmaps_*.npz")))
        if not paths:
            print(f"[detect-report] skip: no localmaps*.npz in {sess_dir}")
            continue

        fpath = paths[0]
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"[detect-report] skip: failed to load {fpath}: {e}")
            continue

        if "tim_detect_ms" not in data.files:
            print(f"[detect-report] skip: {fpath} has no 'tim_detect_ms'")
            continue

        arr = data["tim_detect_ms"].astype(np.float32)

        # Keep finite values; optionally drop zeros.
        mask = np.isfinite(arr)
        if drop_zeros:
            mask &= (arr > 0.0)

        if not np.any(mask):
            print(f"[detect-report] skip: no valid times after masking in {fpath}")
            continue

        t = arr[mask]
        n = t.size
        mean_v = float(np.mean(t))
        med_v  = float(np.median(t))
        std_v  = float(np.std(t))
        min_v  = float(np.min(t))
        max_v  = float(np.max(t))

        # Map the argmax in the masked array back to the original index.
        idx_in_masked = int(np.argmax(t))
        orig_indices  = np.flatnonzero(mask)
        max_idx_orig  = int(orig_indices[idx_in_masked])

        # Requested percentiles.
        pct_vals = {}
        for p in percentiles:
            try:
                pct_vals[p] = float(np.percentile(t, p))
            except Exception:
                pct_vals[p] = float("nan")

        pcts_str = " ".join([f"p{p}={pct_vals[p]:.2f}ms" for p in percentiles])
        print(
            f"[detect-report] session={sess}  N={n}  mean={mean_v:.2f}ms  median={med_v:.2f}ms  "
            f"std={std_v:.2f}ms  min={min_v:.2f}ms  max={max_v:.2f}ms (map_idx={max_idx_orig})  {pcts_str}"
        )
    