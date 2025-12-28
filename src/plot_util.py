import matplotlib.pyplot as plt
import numpy as np

def save_performance_plot(times_arr, save_path):
    """
    繪製執行時間折線圖並存檔。
    
    Args:
        times_arr: 包含每次執行時間的 list 或 numpy array (單位: 秒)
        save_path: 圖片儲存路徑 (包含檔名與副檔名, e.g., .png, .svg)
    """
    # 轉換為 numpy array 並轉成毫秒 (ms) 以便閱讀
    times_ms = np.array(times_arr) * 1000
    mean_time = np.mean(times_ms)
    max_time = np.max(times_ms)

    # 建立一個新的 Figure，避免影響現有的 plt 狀態
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 繪製折線圖
    ax.plot(times_ms, color='tab:blue', linewidth=1, alpha=0.8, label='Per-frame Time')
    
    # 繪製平均線 (紅色虛線)
    ax.axhline(mean_time, color='tab:red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_time:.2f} ms')
    
    # 設定標題與標籤
    ax.set_title(f'Performance Analysis (Total {len(times_ms)} frames)')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Time (ms)')
    
    # 加入 Legend 與 Grid
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 在圖表上方標註最大值，方便快速抓出極端值 (Outliers)
    ax.text(0, max_time, f' Max: {max_time:.2f} ms', verticalalignment='bottom', fontsize=8, color='tab:red')

    # 存檔與關閉
    plt.savefig(save_path)
    print(f"Performance plot saved to: {save_path}")
    plt.close(fig) # 務必關閉 figure 以釋放記憶體