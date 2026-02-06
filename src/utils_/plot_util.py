import matplotlib.pyplot as plt
import numpy as np

def save_performance_plot(times_arr, save_path):
    times_ms = np.array(times_arr) * 1000
    mean_time = np.mean(times_ms)
    max_time = np.max(times_ms)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times_ms, color='tab:blue', linewidth=1, alpha=0.8, label='Per-frame Time')
    
    ax.axhline(mean_time, color='tab:red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_time:.2f} ms')
    
    ax.set_title(f'Performance Analysis (Total {len(times_ms)} frames)')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Time (ms)')
    
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.text(0, max_time, f' Max: {max_time:.2f} ms', verticalalignment='bottom', fontsize=8, color='tab:red')

    plt.savefig(save_path)
    print(f"Performance plot saved to: {save_path}")
    plt.close(fig)