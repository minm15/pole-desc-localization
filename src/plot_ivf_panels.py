import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def nice_bounds(vmin, vmax, step=0.05):
    """Compute clean linear-axis bounds and tick locations (used for e_pos)."""
    lo = math.floor((vmin - 1e-12) / step) * step
    hi = math.ceil((vmax + 1e-12) / step) * step
    ticks, x = [], lo
    for _ in range(1000):
        ticks.append(round(x, 10))
        x += step
        if x > hi + 1e-12:
            break
    return lo, hi, ticks


def main(csv_path: str, out_path: str, linthresh: float = 60.0):
    df = pd.read_csv(csv_path)
    required = {"nprobe", "metric", "session", "nlist", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing column(s): {missing}")

    # X-axis as an equidistant categorical axis
    nlists = sorted(df["nlist"].unique().tolist())
    x_pos = list(range(len(nlists)))
    x_labels = [str(n) for n in nlists]

    # Column order (panels left→right)
    nprobes = sorted(df["nprobe"].unique().tolist())
    sessions = ["2012-03-17", "2012-04-29", "2012-08-20", "2012-11-04", "2013-01-10"]
    sessions = [s for s in sessions if s in set(df["session"])]

    fig, axes = plt.subplots(2, len(nprobes), figsize=(4.6 * len(nprobes), 9.0), squeeze=False)

    # ---------- Top row: Update (symlog + fixed major ticks; disable minor ticks) ----------
    upd = df[df["metric"] == "update"]
    upd_min, upd_max = upd["value"].min(), upd["value"].max()
    ylo_update = max(1e-1, upd_min * 0.6)
    yhi_update = upd_max * 1.15

    # Fixed major ticks including 20/30/40/50/100/200/500/1000; keep doubling if needed
    major_ticks_update = [20, 30, 40, 50, 100, 200, 500, 1000]
    while yhi_update > major_ticks_update[-1]:
        major_ticks_update.append(major_ticks_update[-1] * 2)

    thousand_fmt = mticker.FuncFormatter(lambda x, pos: f"{x:,.0f}")

    for j, nprobe in enumerate(nprobes):
        ax = axes[0, j]
        sub = upd[upd["nprobe"] == nprobe]

        # One polyline per session
        for sess in sessions:
            d = sub[sub["session"] == sess].sort_values("nlist")
            if len(d) != len(nlists):  # skip if any nlist is missing
                continue
            ax.plot(x_pos, d["value"].tolist(), marker="o", linewidth=1.5, label=sess)

        # Labels
        ax.set_xlabel("nlist (categorical)")
        if j == 0:
            ax.set_ylabel("Update time (ms, symlog)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)

        # symlog scale; turn off minor ticks to avoid weird ticks in the linear zone
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
        ax.set_ylim(ylo_update, yhi_update)
        ax.set_yticks(major_ticks_update)
        ax.yaxis.set_major_formatter(thousand_fmt)
        ax.yaxis.set_minor_locator(mticker.NullLocator())  # critical: remove spurious minor ticks
        ax.tick_params(axis="y", which="both", labelleft=True)

        # Grid: majors only
        ax.grid(True, which="major", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.grid(False, which="minor")

        # nprobe badge (top row)
        ax.text(
            0.02,
            0.98,
            f"nprobe={nprobe}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7", boxstyle="round,pad=0.25"),
        )

    # ---------- Bottom row: e_pos (shared limits and ticks; also add badge) ----------
    ep = df[df["metric"] == "e_pos"]
    ep_min, ep_max = ep["value"].min(), ep["value"].max()
    ylo_ep, yhi_ep, y_ticks_ep = nice_bounds(ep_min, ep_max, step=0.05)

    for j, nprobe in enumerate(nprobes):
        ax = axes[1, j]
        sub = ep[ep["nprobe"] == nprobe]

        for sess in sessions:
            d = sub[sub["session"] == sess].sort_values("nlist")
            if len(d) != len(nlists):
                continue
            ax.plot(x_pos, d["value"].tolist(), marker="o", linewidth=1.5, label=sess)

        ax.set_xlabel("nlist (categorical)")
        if j == 0:
            ax.set_ylabel("e_pos")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)

        ax.set_ylim(ylo_ep, yhi_ep)
        ax.set_yticks(y_ticks_ep)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", which="both", labelleft=True)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)

        # nprobe badge (bottom row as well)
        ax.text(
            0.02,
            0.98,
            f"nprobe={nprobe}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7", boxstyle="round,pad=0.25"),
        )

    # Figure-level legend at the very top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(sessions),
            bbox_to_anchor=(0.5, 0.995),
            frameon=True,
        )

    # Leave room for the top legend
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
    fig.savefig(out_path, dpi=170)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2×N IVF panels: Update uses symlog with a linear threshold (no minor ticks); "
                    "e_pos uses shared linear limits; both rows show nprobe badges."
    )
    parser.add_argument("--csv", type=str, default="ivf_summary_sessions_update_epos.csv",
                        help="Input CSV with columns: nprobe, metric, session, nlist, value.")
    parser.add_argument("--out", type=str, default="ivf_panels_symlog_badge.png",
                        help="Output figure path.")
    parser.add_argument("--linthresh", type=float, default=60.0,
                        help="Linear threshold for symlog scale on update plots.")
    args = parser.parse_args()
    main(args.csv, args.out, linthresh=args.linthresh)